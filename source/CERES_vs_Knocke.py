
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
from tools.utilities import pos_vel_from_orekit_ephem, HCL_diff, jd_to_utc, keplerian_elements_from_orekit_ephem
from tools.data_processing import extract_hourly_ceres_data, extract_hourly_ceres_data ,combine_lw_sw_data
from tools.TLE_tools import twoLE_parse, tle_convert

# Define constants
SATELLITE_MASS = 500.0
PROPAGATION_TIME = 3600.0 * 6.0
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
    import datetime
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
    plt.tight_layout()
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'output/ERP_prop/{timenow}_HCL_differences.png')
    # plt.show()

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

def main(TLE, sat_name):
    # CERES SYN1Deg Dataset path and TLE data
    dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc'
    jd_start = 2460069.5000000
    # jd_end = jd_start + 1/24
    # dt = 60  # Seconds
    # tle_time = TLE_time(TLE) ##The TLE I have is not actually in the daterange of the CERES dataset I downloaded so not using this now
    utc_start = jd_to_utc(jd_start)

    # Load data
    data = nc.Dataset(dataset_path)
    ceres_times, _, _, lw_radiation_data, sw_radiation_data = extract_hourly_ceres_data(data)
    combined_radiation_data = combine_lw_sw_data(lw_radiation_data, sw_radiation_data)

    # Convert JD start epoch to UTC and pass to AbsoluteDate
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

    # Instantiate the inertial frame where the orbit is defined
    inertialFrame = FramesFactory.getEME2000()

    # Orbit construction as Keplerian
    initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lv, PositionAngleType.TRUE, inertialFrame, TLE_epochDate, Constants.WGS84_EARTH_MU)

    # Set parameters for numerical propagation
    tolerances = NumericalPropagator.tolerances(POSITION_TOLERANCE, initialOrbit, initialOrbit.getType())
    integrator = DormandPrince853Integrator(INTEGRATOR_MIN_STEP, INTEGRATOR_MAX_STEP, JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(INTEGRATOR_INIT_STEP)

    # Initial state
    initialState = SpacecraftState(initialOrbit, SATELLITE_MASS)

    # No ERP Propagation
    propagator_no_erp = NumericalPropagator(integrator)
    propagator_no_erp.setOrbitType(OrbitType.CARTESIAN)
    propagator_no_erp.setInitialState(initialState)
    gravityProvider = GravityFieldFactory.getNormalizedProvider(1, 1)
    propagator_no_erp.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    ephemGen_no_erp = propagator_no_erp.getEphemerisGenerator()  # Get the ephemeris generator
    end_state_no_erp = propagator_no_erp.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(PROPAGATION_TIME))

    # CERES ERP Propagation
    ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data) # pass the time and radiation data to the force model
    propagator_ceres_erp = NumericalPropagator(integrator)
    propagator_ceres_erp.setOrbitType(OrbitType.CARTESIAN)
    propagator_ceres_erp.setInitialState(initialState)
    propagator_ceres_erp.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    propagator_ceres_erp.addForceModel(ceres_erp_force_model)
    ephemGen_CERES = propagator_ceres_erp.getEphemerisGenerator()  # Get the ephemeris generator
    end_state_ceres = propagator_ceres_erp.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(PROPAGATION_TIME))
    rtn_accs = ceres_erp_force_model.rtn_accs
    rtn_times = ceres_erp_force_model.time_data
    scalar_acc_data = ceres_erp_force_model.scalar_acc_data

    # Knocke ERP Propagation
    sun = CelestialBodyFactory.getSun()
    spacecraft = IsotropicRadiationSingleCoefficient(10.0, 1.0)  # area 10 and Cr 1.0
    onedeg_in_rad = np.radians(1.0)
    angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
    knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
    propagator_knocke_erp = NumericalPropagator(integrator)
    propagator_knocke_erp.setOrbitType(OrbitType.CARTESIAN)
    propagator_knocke_erp.setInitialState(initialState)
    propagator_knocke_erp.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    propagator_knocke_erp.addForceModel(knockeModel)
    ephemGen_knocke = propagator_knocke_erp.getEphemerisGenerator()  # Get the ephemeris generator
    end_state_with_knocke = propagator_knocke_erp.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(PROPAGATION_TIME))

    # Calculate the norm of 3D differences between end states
    print("norm of 3D difference between No ERP and CERES end states:", end_state_no_erp.getPVCoordinates().getPosition().subtract(end_state_ceres.getPVCoordinates().getPosition()).getNorm())
    print("norm of 3D difference between No ERP and Knocke end states:", end_state_no_erp.getPVCoordinates().getPosition().subtract(end_state_with_knocke.getPVCoordinates().getPosition()).getNorm())
    print("norm of 3D difference between Knocke and CERES states:", end_state_with_knocke.getPVCoordinates().getPosition().subtract(end_state_ceres.getPVCoordinates().getPosition()).getNorm())

    # Ephemeris generators setup
    ephemeris_generators = {
        'CERES ERP': ephemGen_CERES,
        'Knocke ERP': ephemGen_knocke,
        'No ERP': ephemGen_no_erp
    }

    # Extract state vector data from ephemerides
    state_vector_data = {}
    keplerian_element_data = {}
    for ephem_name, ephem in ephemeris_generators.items():
        ephemeris = ephem.getGeneratedEphemeris()
        end_date = TLE_epochDate.shiftedBy(PROPAGATION_TIME)
        times, state_vectors = pos_vel_from_orekit_ephem(ephemeris, TLE_epochDate, end_date, INTEGRATOR_INIT_STEP)
        times, keplerian_elements = keplerian_elements_from_orekit_ephem(ephemeris, TLE_epochDate, end_date, INTEGRATOR_INIT_STEP, Constants.WGS84_EARTH_MU)
        state_vector_data[ephem_name] = (times, state_vectors)
        keplerian_element_data[ephem_name] = (times, keplerian_elements)

    # Define a list of colors for different ephemeris generators
    colors = ['blue', 'green', 'red', 'purple', 'brown', 'orange', 'pink', 'gray', 'olive', 'cyan']

    # Titles for each subplot
    titles = ['Semi-Major Axis', 'Eccentricity', 'Inclination', 
            'Argument of Perigee', 'Right Ascension of Ascending Node', 'True Anomaly']

    # Plot Keplerian Elements (subplot 3x2) for each propagator
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

    # Iterate over each subplot
    for ax_index, ax in enumerate(axes.flatten()):
        ax.set_title(titles[ax_index])
        ax.set_xlabel('Time (seconds from start)')
        ax.set_ylabel('Value')
        ax.grid(True)

        # Format y-axis to avoid scientific notation
        ax.ticklabel_format(useOffset=False, style='plain')

        # Plot data for each ephemeris generator
        for name_index, (name, keplerian_data) in enumerate(keplerian_element_data.items()):
            times = keplerian_data[0]
            keplerian_elements = keplerian_data[1]

            # Extract the i-th Keplerian element for each time point
            element_values = [element[ax_index] for element in keplerian_elements]

            # Use a different color for each name
            color = colors[name_index % len(colors)]

            # Plot the i-th Keplerian element for the current generator
            ax.plot(times, element_values, label=name, color=color, linestyle='--')

        ax.legend()

    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.tight_layout()

    # Save and show the plot
    import datetime
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'output/ERP_prop/{sat_name}_{timenow}_ERP_Kepels.png')

    # Compute HCL differences
    HCL_diffs = {}
    for name in ['CERES ERP', 'Knocke ERP']:
        _, state_vectors = state_vector_data[name]
        _, no_erp_state_vectors = state_vector_data['No ERP']
        H_diffs, C_diffs, L_diffs = HCL_diff(np.array(state_vectors), np.array(no_erp_state_vectors))
        HCL_diffs[name] = (H_diffs, C_diffs, L_diffs)

    # Plot HCL differences
    titles = [f'Height: {sat_name}', f'Cross-Track: {sat_name}', f'Along-Track: {sat_name}']
    colors = {'CERES ERP': 'tab:green', 'Knocke ERP': 'tab:orange'}
    plot_hcl_differences(HCL_diffs, state_vector_data['No ERP'][0], titles, colors)

    # Plotting
    ceres_times, _ = state_vector_data['CERES ERP']
    r_components = [acc[0] for acc in rtn_accs]
    t_components = [acc[1] for acc in rtn_accs]
    n_components = [acc[2] for acc in rtn_accs]    

    plt.figure(figsize=(12, 6))
    # Plot RTN components
    plt.subplot(2, 1, 1)
    plt.scatter(rtn_times, r_components, label='Radial (R) ', color='xkcd:sky blue', s=2)
    plt.scatter(rtn_times, t_components, label='Transverse (T)', color='xkcd:light red', s=2)
    plt.scatter(rtn_times, n_components, label='Normal (N) ', color='xkcd:light green', s=2)
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    # Plot Scalar Acceleration Data
    plt.subplot(2, 1, 2)
    plt.scatter(rtn_times, scalar_acc_data, label='Scalar Acceleration', color='orange', s=2)
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    import datetime
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'output/ERP_prop/{timenow}_{sat_name}_ERP_acceleration.png')

if __name__ == "__main__":
    #OneWeb TLE
    TLE_OW = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993\n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"
    #Starlink TLE
    TLE_SL= "1 58214U 23170J   23345.43674150  .00003150  00000+0  17305-3 0  9997\n2 58214  42.9996 329.1219 0001662 255.3130 104.7534 15.15957346  7032"
    main(TLE_OW, "OneWeb")
    main(TLE_SL, "Starlink")

    #TODO: # Osculating elements plot