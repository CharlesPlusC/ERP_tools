


from orekit import JArray_double
from orekit.pyhelpers import absolutedate_to_datetime
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, OrbitType
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation.analytical.tle import TLEPropagator
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, NewtonianAttraction
from org.orekit.propagation import SpacecraftState
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.forces.radiation import KnockeRediffusedForceModel, IsotropicRadiationSingleCoefficient

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime

from tools.CERES_ERP import CERES_ERP_ForceModel
from tools.utilities import pos_vel_from_orekit_ephem, HCL_diff, jd_to_utc, keplerian_elements_from_orekit_ephem, extract_acceleration
from source.tools.ceres_data_processing import extract_hourly_ceres_data, extract_hourly_ceres_data ,combine_lw_sw_data
from tools.TLE_tools import twoLE_parse, tle_convert
from tools.plotting import plot_kepels_evolution, plot_hcl_differences

# Define constants
SATELLITE_MASS = 500.0
PROPAGATION_TIME = 3600.0 * 24.0 * 7.0
INTEGRATOR_MIN_STEP = 0.01
INTEGRATOR_MAX_STEP = 120.0
INTEGRATOR_INIT_STEP = 30.0
POSITION_TOLERANCE = 1e-3

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

#TODO: move the plotting functions to the plotting module

def generate_ephemeris_and_extract_data(propagators, start_date, end_date, time_step):
    state_vector_data = {}

    for name, propagator in propagators.items():
        ephemeris = propagator.getEphemerisGenerator().getGeneratedEphemeris()
        times, state_vectors = pos_vel_from_orekit_ephem(ephemeris, start_date, end_date, time_step)
        state_vector_data[name] = (times, state_vectors)

    return state_vector_data

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
    MU = Constants.WGS84_EARTH_MU
    newattr = NewtonianAttraction(MU)
    propagator_no_erp.addForceModel(newattr)
    ephemGen_no_erp = propagator_no_erp.getEphemerisGenerator()  # Get the ephemeris generator
    end_state_no_erp = propagator_no_erp.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(PROPAGATION_TIME))

    # CERES ERP Propagation
    ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data) # pass the time and radiation data to the force model
    propagator_ceres_erp = NumericalPropagator(integrator)
    propagator_ceres_erp.setOrbitType(OrbitType.CARTESIAN)
    propagator_ceres_erp.setInitialState(initialState)
    propagator_ceres_erp.addForceModel(ceres_erp_force_model)
    MU = Constants.WGS84_EARTH_MU
    newattr = NewtonianAttraction(MU)
    propagator_ceres_erp.addForceModel(newattr)
    ephemGen_CERES = propagator_ceres_erp.getEphemerisGenerator()  # Get the ephemeris generator
    end_state_ceres = propagator_ceres_erp.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(PROPAGATION_TIME))
    ceres_rtn_accs = ceres_erp_force_model.rtn_accs
    ceres_rtn_times = ceres_erp_force_model.time_data
    ceres_scalar_acc_data = ceres_erp_force_model.scalar_acc_data

    # Knocke ERP Propagation
    sun = CelestialBodyFactory.getSun()
    spacecraft = IsotropicRadiationSingleCoefficient(10.0, 1.0)  # area 10 and Cr 1.0
    onedeg_in_rad = np.radians(1.0)
    angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
    knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
    propagator_knocke_erp = NumericalPropagator(integrator)
    propagator_knocke_erp.setOrbitType(OrbitType.CARTESIAN)
    propagator_knocke_erp.setInitialState(initialState)
    MU = Constants.WGS84_EARTH_MU
    newattr = NewtonianAttraction(MU)
    propagator_knocke_erp.addForceModel(newattr)
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

    #TODO: this is what I need to save not the derived data

    knocke_accelerations, knocke_rtn_components = extract_acceleration(state_vector_data, TLE_epochDate, SATELLITE_MASS, knockeModel, rtn=True)
    plot_kepels_evolution(keplerian_element_data, sat_name)

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

    ##### ERP Acceleration Plots #####

    TLE_epochDate_datetime = absolutedate_to_datetime(TLE_epochDate)

    # Plotting
    ceres_times, _ = state_vector_data['CERES ERP']
    ceres_r_components = np.array([acc[0] for acc in ceres_rtn_accs])
    ceres_t_components = np.array([acc[1] for acc in ceres_rtn_accs])
    ceres_n_components = np.array([acc[2] for acc in ceres_rtn_accs])   

    # Extracting RTN components from Knocke ERP data
    knocke_r_components = np.array([acc[0] for acc in knocke_rtn_components])
    knocke_t_components = np.array([acc[1] for acc in knocke_rtn_components])
    knocke_n_components = np.array([acc[2] for acc in knocke_rtn_components])

    # Calculate Knocke scalar acceleration data
    knocke_scalar_acc_data = np.array([np.linalg.norm([acc.getX(), acc.getY(), acc.getZ()]) for acc in knocke_accelerations])
    ceres_scalar_acc_data = np.array(ceres_scalar_acc_data)

    # Extract states and times for the Knocke ERP ephemeris
    knocke_states_and_times = state_vector_data['Knocke ERP']
    knocke_times, _ = knocke_states_and_times

    # Convert these times to a format suitable for plotting
    # Assuming the times are in seconds since the TLE_epochDate

    from astropy.time import Time
    knocke_times_julian = [Time(TLE_epochDate_datetime + datetime.timedelta(seconds=duration)).jd for duration in knocke_times]
    start_date = max(min(ceres_rtn_times), min(knocke_times_julian))
    end_date = min(max(ceres_rtn_times), max(knocke_times_julian))


    # Determine max and min for scalar acceleration and RTN acceleration
    max_scalar_acc = max(max(ceres_scalar_acc_data), max(knocke_scalar_acc_data))
    min_scalar_acc = min(min(ceres_scalar_acc_data), min(knocke_scalar_acc_data))

    max_rtn_acc = max(ceres_r_components.max(), ceres_t_components.max(), ceres_n_components.max(),
                    knocke_r_components.max(), knocke_t_components.max(), knocke_n_components.max())
    min_rtn_acc = min(ceres_r_components.min(), ceres_t_components.min(), ceres_n_components.min(),
                    knocke_r_components.min(), knocke_t_components.min(), knocke_n_components.min())


    #save all the data to .npy files in "output/ERP_prop/saved_runs"
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_ceres_r_components.npy', ceres_r_components)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_ceres_t_components.npy', ceres_t_components)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_ceres_n_components.npy', ceres_n_components)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_ceres_scalar_acc_data.npy', ceres_scalar_acc_data)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_ceres_rtn_times.npy', ceres_rtn_times)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_ceres_times.npy', ceres_times)

    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_knocke_r_components.npy', knocke_r_components)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_knocke_t_components.npy', knocke_t_components)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_knocke_n_components.npy', knocke_n_components)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_knocke_scalar_acc_data.npy', knocke_scalar_acc_data)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_knocke_times_julian.npy', knocke_times_julian)

    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_HCL_diffs.npy', HCL_diffs)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_keplerian_element_data.npy', keplerian_element_data)
    np.save(f'output/ERP_prop/saved_runs/{timenow}_{sat_name}_state_vector_data.npy', state_vector_data)

    plt.figure(figsize=(12, 12))

    # Plot RTN components for CERES ERP
    plt.subplot(4, 1, 1)
    plt.scatter(ceres_rtn_times, ceres_r_components, label='CERES Radial (R)', color='xkcd:sky blue',s=2)
    plt.scatter(ceres_rtn_times, ceres_t_components, label='CERES Transverse (T)', color='xkcd:light red',s=2)
    plt.scatter(ceres_rtn_times, ceres_n_components, label='CERES Normal (N)', color='xkcd:light green',s=2)
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('CERES RTN Acceleration Components')
    plt.grid(True)
    plt.legend()
    plt.ylim(min_rtn_acc-0.1*min_rtn_acc, max_rtn_acc+0.1*max_rtn_acc)

    # Plot RTN components for Knocke ERP
    plt.subplot(4, 1, 2)
    plt.scatter(knocke_times_julian, knocke_r_components, label='Knocke Radial (R)', color='xkcd:blue',s=2)
    plt.scatter(knocke_times_julian, knocke_t_components, label='Knocke Transverse (T)', color='xkcd:red',s=2)
    plt.scatter(knocke_times_julian, knocke_n_components, label='Knocke Normal (N)', color='xkcd:green',s=2)
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Knocke RTN Acceleration Components')
    plt.grid(True)
    plt.legend()
    plt.ylim(min_rtn_acc-0.1*min_rtn_acc, max_rtn_acc+0.1*max_rtn_acc)

    # Plot Scalar Acceleration Data for CERES ERP
    plt.subplot(4, 1, 3)
    plt.scatter(ceres_rtn_times, ceres_scalar_acc_data, label='CERES Scalar Acceleration', color='xkcd:purple',s=3)
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('CERES Scalar Acceleration')
    plt.grid(True)
    plt.legend()
    plt.ylim(min_scalar_acc-0.1*min_scalar_acc, max_scalar_acc+0.1*max_scalar_acc)

    # Plot Scalar Acceleration Data for Knocke ERP
    plt.subplot(4, 1, 4)
    plt.scatter(knocke_times_julian, knocke_scalar_acc_data, label='Knocke Scalar Acceleration', color='xkcd:orange',s=3)
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Knocke Scalar Acceleration')
    plt.grid(True)
    plt.legend()
    plt.ylim(min_scalar_acc-0.1*min_scalar_acc, max_scalar_acc+0.1*max_scalar_acc)

    plt.tight_layout()
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Save the plot
    plt.savefig(f'output/ERP_prop/{timenow}_{sat_name}_ERP_acceleration.png')

if __name__ == "__main__":
    #OneWeb TLE
    TLE_OW = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993\n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"
    #Starlink TLE
    TLE_SL= "1 58214U 23170J   23345.43674150  .00003150  00000+0  17305-3 0  9997\n2 58214  42.9996 329.1219 0001662 255.3130 104.7534 15.15957346  7032"
    main(TLE_OW, "OneWeb")
    main(TLE_SL, "Starlink")