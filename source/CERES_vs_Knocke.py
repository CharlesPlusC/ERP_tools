
import orekit
from orekit.pyhelpers import setup_orekit_curdir

orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()

from orekit import JArray_double
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, OrbitType
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation.analytical.tle import TLEPropagator
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.propagation import SpacecraftState
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.forces.radiation import KnockeRediffusedForceModel, IsotropicRadiationSingleCoefficient

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

from tools.CERES_ERP import CERES_ERP_ForceModel
from tools.utilities import pos_vel_from_orekit_ephem, HCL_diff, jd_to_utc
from tools.data_processing import extract_hourly_ceres_data, extract_hourly_ceres_data ,combine_lw_sw_data
from tools.TLE_tools import twoLE_parse, tle_convert

# Define constants
SATELLITE_MASS = 500.0
PROPAGATION_TIME = 200.0
INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 1000.0
INTEGRATOR_INIT_STEP = 60.0
POSITION_TOLERANCE = 1e-5

def calculate_position_differences(end_state1, end_state2):
    position1 = end_state1.getPVCoordinates().getPosition()
    position2 = end_state2.getPVCoordinates().getPosition()
    return position1.subtract(position2).getNorm()

def compute_hcl_differences(state_vector_data):
    HCL_diffs = {}
    for name in ['CERES ERP', 'Knocke ERP']:
        _, state_vectors = state_vector_data[name]
        _, no_erp_state_vectors = state_vector_data['No ERP']
        H_diffs, C_diffs, L_diffs = HCL_diff(np.array(state_vectors), np.array(no_erp_state_vectors))
        HCL_diffs[name] = (H_diffs, C_diffs, L_diffs)
    return HCL_diffs

def generate_ephemeris_and_extract_data(propagators, start_date, end_date, time_step):
    state_vector_data = {}

    for name, propagator in propagators.items():
        ephemeris = propagator.getEphemerisGenerator().getGeneratedEphemeris()
        times, state_vectors = pos_vel_from_orekit_ephem(ephemeris, start_date, end_date, time_step)
        state_vector_data[name] = (times, state_vectors)

    return state_vector_data


def plot_hcl_differences(hcl_diffs, time_data, titles, colors):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
    
    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.set_xlabel('Time (seconds from start)')
        ax.set_ylabel('Difference (meters)')
        ax.grid(True)

        for name, diffs in hcl_diffs.items():
            ax.plot(time_data, diffs[i], label=f'{name} - No ERP', color=colors[name], linestyle='--')
        ax.legend()

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_position_differences(time_data, differences, labels, title, ylabel):
    plt.figure(figsize=(10, 6))
    for i, diff in enumerate(differences):
        plt.plot(time_data, diff, label=labels[i])
    plt.xlabel('Time (seconds from start)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def setup_propagator(initial_orbit, force_models, positionTolerance):
    tolerances = NumericalPropagator.tolerances(positionTolerance, 
                                            initial_orbit, 
                                            initial_orbit.getType())
    integrator = DormandPrince853Integrator(INTEGRATOR_MIN_STEP, INTEGRATOR_MAX_STEP, 
                                            JArray_double.cast_(tolerances[0]), 
                                            JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(INTEGRATOR_INIT_STEP)
    
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    initialState = SpacecraftState(initial_orbit, SATELLITE_MASS)
    propagator.setInitialState(initialState)

    for model in force_models:
        propagator.addForceModel(model)

    return propagator

def propagate_orbit(propagator, start_date, duration):
    end_state = propagator.propagate(start_date, start_date.shiftedBy(duration))
    return end_state

def main():
    # Dataset path and TLE data
    dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc'
    TLE = "1 58214U 23170J   23345.43674150  .00003150  00000+0  17305-3 0  9997\n2 58214  42.9996 329.1219 0001662 255.3130 104.7534 15.15957346  7032"
    jd_start = 2460069.5000000
    jd_end = jd_start + 1/24
    dt = 60  # Seconds

    # Load data
    data = nc.Dataset(dataset_path)
    ceres_times, _, _, lw_radiation_data, sw_radiation_data = extract_hourly_ceres_data(data)
    combined_radiation_data = combine_lw_sw_data(lw_radiation_data, sw_radiation_data)

    # Convert JD start epoch to UTC and pass to AbsoluteDate
    utc_start = jd_to_utc(jd_start)
    YYYY, MM, DD, H, M, S = [int(utc_start.strftime("%Y")), int(utc_start.strftime("%m")), 
                             int(utc_start.strftime("%d")), int(utc_start.strftime("%H")), 
                             int(utc_start.strftime("%M")), float(utc_start.strftime("%S"))]

    utc = TimeScalesFactory.getUTC()
    TLE_epochDate = AbsoluteDate(YYYY, MM, DD, H, M, S, utc)

    # Convert the initial position and velocity to Keplerian elements
    tle_dict = twoLE_parse(TLE)
    kep_elems = tle_convert(tle_dict)

    # Keplerian parameters
    a, e, i, omega, raan, lv = [float(kep_elems[key]) for key in ['a', 'e', 'i', 'arg_p', 'RAAN', 'true_anomaly']]
    a *= 1000  # Convert to meters

    # Orbit construction
    inertialFrame = FramesFactory.getEME2000()
    initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lv, PositionAngleType.TRUE, inertialFrame, TLE_epochDate, Constants.WGS84_EARTH_MU)

    # Numerical propagator setup
    gravityProvider = GravityFieldFactory.getNormalizedProvider(10, 10)
    gravityModel = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider)

    # Create and configure propagators for different models
    propagator_num = setup_propagator(initialOrbit, [gravityModel], POSITION_TOLERANCE)
    end_state_no_erp = propagate_orbit(propagator_num, TLE_epochDate, jd_end - jd_start)

    ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data)
    propagator_ceres_erp = setup_propagator(initialOrbit, [gravityModel, ceres_erp_force_model], POSITION_TOLERANCE)
    end_state_ceres_erp = propagate_orbit(propagator_ceres_erp, TLE_epochDate, jd_end - jd_start)

    sun = CelestialBodyFactory.getSun()
    spacecraft = IsotropicRadiationSingleCoefficient(10.0, 1.0)
    onedeg_in_rad = np.radians(1.0)
    angularResolution = float(onedeg_in_rad)
    knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
    propagator_knocke = setup_propagator(initialOrbit, [gravityModel, knockeModel], POSITION_TOLERANCE)
    end_state_knocke = propagate_orbit(propagator_knocke, TLE_epochDate, jd_end - jd_start)

    # Calculate differences between end states
    diff_no_erp_ceres = calculate_position_differences(end_state_no_erp, end_state_ceres_erp)
    diff_no_erp_knocke = calculate_position_differences(end_state_no_erp, end_state_knocke)
    diff_ceres_knocke = calculate_position_differences(end_state_ceres_erp, end_state_knocke)

    print("3D diff: No ERP vs CERES ERP:", diff_no_erp_ceres)
    print("3D diff: No ERP vs Knocke ERP:", diff_no_erp_knocke)
    print("3D diff: CERES ERP vs Knocke ERP:", diff_ceres_knocke)

    # Setup for ephemeris generation
    propagators_dict = {
        'CERES ERP': propagator_ceres_erp,
        'Knocke ERP': propagator_knocke,
        'No ERP': propagator_num
    }

    end_date = TLE_epochDate.shiftedBy(jd_end - jd_start)
    time_step = 60.0  # timestep of the ephemeris to be extracted

    # Generate ephemeris and extract state vector data
    state_vector_data = generate_ephemeris_and_extract_data(propagators_dict, TLE_epochDate, end_date, time_step)

    # Compute HCL differences
    HCL_diffs = compute_hcl_differences(state_vector_data)

    # Plot HCL differences
    titles = ['Height Differences Over Time', 'Cross-Track Differences Over Time', 'Along-Track Differences Over Time']
    colors = {'CERES ERP': 'tab:green', 'Knocke ERP': 'tab:orange'}
    plot_hcl_differences(HCL_diffs, state_vector_data['No ERP'][0], titles, colors)

if __name__ == "__main__":
    main()