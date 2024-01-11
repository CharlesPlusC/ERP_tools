import orekit
from orekit.pyhelpers import setup_orekit_curdir

orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()

from orekit.pyhelpers import datetime_to_absolutedate
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import Constants as orekit_constants
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions, PVCoordinates
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.orbits import CartesianOrbit
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from orekit import JArray_double
from org.orekit.orbits import OrbitType
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, Relativity, NewtonianAttraction
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.models.earth.atmosphere import DTM2000
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants

from tools.utilities import yyyy_mm_dd_hh_mm_ss_to_jd, jd_to_utc, extract_acceleration
from tools.spaceX_ephem_tools import  parse_spacex_datetime_stamps

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

SATELLITE_MASS = 800.0
INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 15.0
POSITION_TOLERANCE = 1e-3 # 1 mm

def configure_force_models(propagator,cr, cross_section,cd, **config_flags):

    # Earth gravity field with degree 64 and order 64
    if config_flags.get('enable_gravity', True):
        MU = Constants.WGS84_EARTH_MU
        newattr = NewtonianAttraction(MU)
        propagator.addForceModel(newattr)

        ### 64x64 gravity model
        gravityProvider = GravityFieldFactory.getNormalizedProvider(64, 64)
        gravityAttractionModel = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider)
        propagator.addForceModel(gravityAttractionModel)

    # Moon and Sun perturbations
    if config_flags.get('enable_third_body', True):
        moon = CelestialBodyFactory.getMoon()
        sun = CelestialBodyFactory.getSun()
        moon_3dbodyattraction = ThirdBodyAttraction(moon)
        sun_3dbodyattraction = ThirdBodyAttraction(sun)
        propagator.addForceModel(moon_3dbodyattraction)
        propagator.addForceModel(sun_3dbodyattraction)

    # Solar radiation pressure
    if config_flags.get('enable_solar_radiation', True):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        cross_section = float(cross_section)
        cr = float(cr)
        isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(cross_section, cr)
        solarRadiationPressure = SolarRadiationPressure(sun, wgs84Ellipsoid, isotropicRadiationSingleCoeff)
        propagator.addForceModel(solarRadiationPressure)

    # Atmospheric drag
    if config_flags.get('enable_atmospheric_drag', True):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        propagator.addForceModel(dragForce)

    # TODO: CERES ERP force model
    # if enable_ceres:
    #     ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data) # pass the time and radiation data to the force model

    return propagator

def propagate_state(start_date, end_date, initial_state_vector, cr=1.5, cd=1.8, cross_section=10.0, **config_flags):
    x, y, z, vx, vy, vz = initial_state_vector
    frame = FramesFactory.getEME2000() # j2000 frame by default
    # Propagation using the configured propagator
    initial_orbit = CartesianOrbit(PVCoordinates(Vector3D(float(x), float(y), float(z)),
                                                Vector3D(float(vx), float(vy), float(vz))),
                                    frame,
                                    datetime_to_absolutedate(start_date),
                                    Constants.WGS84_EARTH_MU)
    
    tolerances = NumericalPropagator.tolerances(POSITION_TOLERANCE, initial_orbit, initial_orbit.getType())
    integrator = DormandPrince853Integrator(INTEGRATOR_MIN_STEP, INTEGRATOR_MAX_STEP, JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(INTEGRATOR_INIT_STEP)
    initialState = SpacecraftState(initial_orbit, SATELLITE_MASS)
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(initialState)

    #TODO: make configure_force_models() not require cr, cd, and cross_section
    configured_propagator = configure_force_models(propagator,cr,cross_section, cd,**config_flags) # configure force models
    final_state = configured_propagator.propagate(datetime_to_absolutedate(end_date))

    pv_coordinates = final_state.getPVCoordinates()
    position = [pv_coordinates.getPosition().getX(), pv_coordinates.getPosition().getY(), pv_coordinates.getPosition().getZ()]
    velocity = [pv_coordinates.getVelocity().getX(), pv_coordinates.getVelocity().getY(), pv_coordinates.getVelocity().getZ()]

    return position + velocity

def std_dev_from_lower_triangular(lower_triangular_data):
    cov_matrix = np.zeros((6, 6))
    row, col = np.tril_indices(6)
    cov_matrix[row, col] = lower_triangular_data
    cov_matrix = cov_matrix + cov_matrix.T - np.diag(cov_matrix.diagonal())
    std_dev = np.sqrt(np.diag(cov_matrix))
    return std_dev

def spacex_ephem_to_df_w_cov(ephem_path: str) -> pd.DataFrame:
    """
    Convert SpaceX ephemeris data, including covariance terms, into a pandas DataFrame.

    Parameters
    ----------
    ephem_path : str
        Path to the ephemeris file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing parsed SpaceX ephemeris data, including covariance terms.
    """
    with open(ephem_path) as f:
        lines = f.readlines()

    # Remove header lines and select every 4th line starting from the first data line
    t_xyz_uvw = lines[4::4]

    # Extract t, x, y, z, u, v, w
    t = [float(i.split()[0]) for i in t_xyz_uvw]
    x = [float(i.split()[1]) for i in t_xyz_uvw]
    y = [float(i.split()[2]) for i in t_xyz_uvw]
    z = [float(i.split()[3]) for i in t_xyz_uvw]
    u = [float(i.split()[4]) for i in t_xyz_uvw]
    v = [float(i.split()[5]) for i in t_xyz_uvw]
    w = [float(i.split()[6]) for i in t_xyz_uvw]

    # Extract the 21 covariance terms (3 lines after each primary data line)
    covariance_data = {f'cov_{i+1}': [] for i in range(21)}
    for i in range(5, len(lines), 4):  # Start from the first covariance line
        cov_lines = lines[i:i+3]  # Get the three lines of covariance terms
        cov_values = ' '.join(cov_lines).split()
        for j, value in enumerate(cov_values):
            covariance_data[f'cov_{j+1}'].append(float(value))

    # Convert timestamps to strings and call parse_spacex_datetime_stamps
    t_str = [str(int(i)) for i in t]  # Convert timestamps to string
    parsed_timestamps = parse_spacex_datetime_stamps(t_str)

    # Calculate Julian Dates for each timestamp
    jd_stamps = np.zeros(len(parsed_timestamps))
    for i in range(len(parsed_timestamps)):
        jd_stamps[i] = yyyy_mm_dd_hh_mm_ss_to_jd(int(parsed_timestamps[i][0]), int(parsed_timestamps[i][1]), 
                                                 int(parsed_timestamps[i][2]), int(parsed_timestamps[i][3]), 
                                                 int(parsed_timestamps[i][4]), int(parsed_timestamps[i][5]), 
                                                 int(parsed_timestamps[i][6]))

    # Initialize lists for averaged position and velocity standard deviations
    sigma_x, sigma_y, sigma_z, sigma_xv, sigma_yv, sigma_zv = [], [], [], [], [], []

    # Calculate averaged standard deviations for each row
    for _, row in pd.DataFrame(covariance_data).iterrows():
        std_devs = std_dev_from_lower_triangular(row.values)
        sigma_x.append(std_devs[0])
        sigma_y.append(std_devs[1])
        sigma_z.append(std_devs[2])
        sigma_xv.append(std_devs[3])
        sigma_yv.append(std_devs[4])
        sigma_zv.append(std_devs[5])

    # Construct the DataFrame with all data
    spacex_ephem_df = pd.DataFrame({
        't': t,
        'x': x,
        'y': y,
        'z': z,
        'xv': u,
        'yv': v,
        'zv': w,
        'JD': jd_stamps,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'sigma_z': sigma_z,
        'sigma_xv': sigma_xv,
        'sigma_yv': sigma_yv,
        'sigma_zv': sigma_zv,
        **covariance_data
    })

    # Multiply all the values except the times by 1000 to convert from km to m
    columns_to_multiply = ['x', 'y', 'z', 'xv', 'yv', 'zv', 
                        'sigma_x', 'sigma_y', 'sigma_z', 
                        'sigma_xv', 'sigma_yv', 'sigma_zv'] + [f'cov_{i+1}' for i in range(21)]

    for col in columns_to_multiply:
        spacex_ephem_df[col] *= 1000

    spacex_ephem_df['hours'] = (spacex_ephem_df['JD'] - spacex_ephem_df['JD'][0]) * 24.0 # hours since first timestamp
    # calculate UTC time by applying jd_to_utc() to each JD value
    spacex_ephem_df['UTC'] = spacex_ephem_df['JD'].apply(jd_to_utc)
    # TODO: I am gaining 3 milisecond per minute in the UTC time. Why?
    return spacex_ephem_df

def rho_i(measured_state, measurement_type='state'):
# maps a state vector to a measurement vector
    if measurement_type == 'state':
        return measured_state
    #TODO: implement other measurement types

def propagate_STM(state_ti, t0, dt, phi_i, force_model_config):

    df_dy = np.zeros((len(state_ti),len(state_ti))) #initialize matrix of partial derivatives (partials at time t0)
    # numerical estimation of partial derivatives
    # get the state at ti and the accelerations at ti
    state_vector_data = (state_ti[0], state_ti[1], state_ti[2], state_ti[3], state_ti[4], state_ti[5])
    epochDate = datetime_to_absolutedate(t0)
    accelerations_t0 = np.zeros(3)
    force_models = []
    if force_model_config.get('enable_gravity', True):
        MU = Constants.WGS84_EARTH_MU
        newattr = NewtonianAttraction(MU)
        force_models.append(newattr)
        gravity_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, newattr)
        gravity_eci_t0 = np.array([gravity_eci_t0[0].getX(), gravity_eci_t0[0].getY(), gravity_eci_t0[0].getZ()])
        accelerations_t0+=gravity_eci_t0

        gravityProvider = GravityFieldFactory.getNormalizedProvider(64, 64)
        gravity_force_model = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider)
        force_models.append(gravity_force_model)
        gravity_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, gravity_force_model)
        gravity_eci_t0 = np.array([gravity_eci_t0[0].getX(), gravity_eci_t0[0].getY(), gravity_eci_t0[0].getZ()])
        accelerations_t0+=gravity_eci_t0

    if force_model_config.get('enable_third_body', True):
        moon = CelestialBodyFactory.getMoon()
        sun = CelestialBodyFactory.getSun()
        moon_3dbodyattraction = ThirdBodyAttraction(moon)
        force_models.append(moon_3dbodyattraction)
        sun_3dbodyattraction = ThirdBodyAttraction(sun)
        force_models.append(sun_3dbodyattraction)
        moon_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, moon_3dbodyattraction)
        moon_eci_t0 = np.array([moon_eci_t0[0].getX(), moon_eci_t0[0].getY(), moon_eci_t0[0].getZ()])
        accelerations_t0+=moon_eci_t0
        sun_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, sun_3dbodyattraction)
        sun_eci_t0 = np.array([sun_eci_t0[0].getX(), sun_eci_t0[0].getY(), sun_eci_t0[0].getZ()])
        accelerations_t0+=sun_eci_t0

    if force_model_config.get('enable_solar_radiation', True):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        cross_section = float(10.0) #TODO: get from state vector
        cr = float(1.5)
        isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(cross_section, cr)
        solarRadiationPressure = SolarRadiationPressure(sun, wgs84Ellipsoid, isotropicRadiationSingleCoeff)
        force_models.append(solarRadiationPressure)
        solar_radiation_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, solarRadiationPressure)
        solar_radiation_eci_t0 = np.array([solar_radiation_eci_t0[0].getX(), solar_radiation_eci_t0[0].getY(), solar_radiation_eci_t0[0].getZ()])
        accelerations_t0+=solar_radiation_eci_t0

    if force_model_config.get('enable_atmospheric_drag', True):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(10.0), float(2.2)) #TODO: get from state vector
        dragForce = DragForce(atmosphere, isotropicDrag)
        force_models.append(dragForce)
        atmospheric_drag_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, dragForce)
        atmospheric_drag_eci_t0 = np.array([atmospheric_drag_eci_t0[0].getX(), atmospheric_drag_eci_t0[0].getY(), atmospheric_drag_eci_t0[0].getZ()])
        accelerations_t0+=atmospheric_drag_eci_t0

    # perturb each state variable by a small amount and get the new accelerations
    perturbation = 0.1 # 10 cm
    
    for i in range(len(state_ti)):
        perturbed_accelerations = np.zeros(3)
        state_ti_perturbed = state_ti.copy()
        state_ti_perturbed[i] += perturbation
        for force_model in force_models:
            acc_perturbed = extract_acceleration(state_ti_perturbed, epochDate, SATELLITE_MASS, force_model)
            acc_perturbed = np.array([acc_perturbed[0].getX(), acc_perturbed[0].getY(), acc_perturbed[0].getZ()])
            perturbed_accelerations+=acc_perturbed
        partial_derivatives = (perturbed_accelerations - accelerations_t0) / perturbation
        
        # Assign partial derivatives to the appropriate submatrix
        if i < 3:  # Position components affect acceleration
            df_dy[3:, i] = partial_derivatives
        else:  # Velocity components affect position
            df_dy[i - 3, i] = 1  # Identity matrix in top-right 3x3 submatrix

    assert np.allclose(df_dy[:3, :3], np.zeros((3, 3)), atol=1e-10), "First 3x3 submatrix is not all zeros"
    assert np.allclose(df_dy[3:, 3:], np.zeros((3, 3)), atol=1e-10), "Last 3x3 submatrix is not all zeros"
    assert np.allclose(df_dy[:3, 3:], np.eye(3), atol=1e-10), "Top right 3x3 submatrix is not the identity matrix"
    assert not np.allclose(df_dy[3:, :3], np.zeros((3, 3)), atol=1e-15), "Bottom left 3x3 submatrix is very small"

    dt_seconds = float(dt.total_seconds())
    phi_t1 = phi_i + df_dy @ phi_i * dt_seconds #STM at time t1

    return phi_t1

def BLS_optimize(observations_df, force_model_config, a_priori_estimate=None):

    # Initialize
    t0 = observations_df['UTC'][0]
    x_bar_0 = np.array(a_priori_estimate[1:7])  # assuming this is [x, y, z, u, v, w] for now. TODO: enable adding other state variables
    state_covs = a_priori_estimate[7:13]
    #make it a diagonal matrix
    state_covs = np.diag(state_covs)

    phi_ti_minus1 = np.identity(6)  # 6x6 identity matrix for 6 state variables
    P_0 = np.array(state_covs, dtype=float)  # Covariance matrix from a priori estimate

    d_rho_d_state = np.eye(6)  # Identity matrix for perfect state measurements
    H_matrix = np.empty((0, 6))  # 6 columns for 6 state variables

    converged = False
    iteration = 1
    max_iterations = 10  # Set a max number of iterations
    weighted_rms_last = np.inf
    convergence_threshold = 0.01 # Define convergence threshold for RMS change
    no_times_diff_increased = 0
    while not converged and iteration < max_iterations:
        print(f"Iteration: {iteration}")
        N = np.zeros(6)  # reset N to zero
        lamda = np.linalg.inv(P_0)  # reset lambda to P_0
        y_all = np.empty((0, 6))  # Initialize residuals array
        ti_minus1 = t0
        state_ti_minus1 = x_bar_0 
        RMSs = []
        for obs, row in observations_df.iterrows():
            print(f"Obs: {obs+1}")
            ti = row['UTC']
            observed_state = row[['x', 'y', 'z', 'xv', 'yv', 'zv']].values
            obs_covariance = np.diag(row[['sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']])
            obs_covariance = np.array(obs_covariance, dtype=float)
            W_i = np.linalg.inv(obs_covariance)  # Inverse of observation covariances

            # Propagate state and STM
            dt = ti - ti_minus1
            state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector =state_ti_minus1, cr=2, cd=1.8, cross_section=10.0,config_flags= force_model_config)
            phi_ti = propagate_STM(state_ti_minus1, ti, dt, phi_ti_minus1, force_model_config)

            # Compute H matrix for this observation
            H_matrix_row = d_rho_d_state @ phi_ti
            H_matrix = np.vstack([H_matrix, H_matrix_row]) # Append row to H matrix
            y_i = observed_state - rho_i(state_ti, 'state')
            y_i = np.array(y_i, dtype=float)
            y_all = np.vstack([y_all, y_i])
            
            # Update lambda and N matrices
            lamda += H_matrix_row.T @ W_i @ H_matrix_row
            N += H_matrix_row.T @ W_i @ y_i

            # Update for next iteration
            ti_minus1 = ti
            state_ti_minus1 = state_ti
            phi_ti_minus1 = phi_ti
            #RMS for this observation is y_i.T @ inv_obs_covariance @ y_i
            RMSs.append(y_i.T @ W_i @ y_i)

        # sum all the RMSs
        RMSs = np.array(RMSs)
        #weighted RMS is sqrt(RMS divided by m. where m is (number of observations * number of state variables))
        weighted_rms = np.sqrt(np.sum(RMSs) / (len(x_bar_0) * len (y_all)))
        print(f"Weighted RMS: {weighted_rms}")

        # Solve normal equations
        xhat = np.linalg.inv(lamda) @ N

        #new figure
        plt.figure()
        plt.plot(observations_df['UTC'], y_all)
        plt.xlabel('UTC')
        plt.ylabel('Residuals (m)')
        #set ylim between -1 and 1
        if len(observations_df) == 120:
            plt.ylim(-15, 15)
        else:
            plt.ylim(-1, 1)
        plt.grid(True)
        plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/tapleyBLS/allforce_iter_{iteration}_#pts{len(observations_df)}.png")
        # plt.show()

        # Check for convergence
        if abs(weighted_rms - weighted_rms_last) < convergence_threshold:
            
            print("Converged!")
            converged = True

        else:
            if weighted_rms > weighted_rms_last:
                no_times_diff_increased += 1
                print(f"RMS increased {no_times_diff_increased} times in a row.")
                if no_times_diff_increased > 3:
                    print("RMS increased 3 times in a row. Stopping iteration.")
                    break
            else:
                no_times_diff_increased = 0 #reset the counter
            weighted_rms_last = weighted_rms
            
            print(f"correction: {xhat}")
            print("state before correction: ", x_bar_0)
            x_bar_0 += xhat  # Update nominal trajectory
            print(f"updated state: {x_bar_0}")

        iteration += 1

    return x_bar_0, np.linalg.inv(lamda)

if __name__ == "__main__":
    spacex_ephem_df = spacex_ephem_to_df_w_cov('external/ephems/starlink/MEME_57632_STARLINK-30309_3530645_Operational_1387262760_UNCLASSIFIED.txt')

    # Initialize state vector from the first point in the SpaceX ephemeris
    # TODO: Can perturb this initial state vector to test convergence later
    initial_X = spacex_ephem_df['x'][0]
    initial_Y = spacex_ephem_df['y'][0]
    initial_Z = spacex_ephem_df['z'][0]
    initial_VX = spacex_ephem_df['xv'][0]
    initial_VY = spacex_ephem_df['yv'][0]
    initial_VZ = spacex_ephem_df['zv'][0]
    initial_sigma_X = spacex_ephem_df['sigma_x'][0]
    initial_sigma_Y = spacex_ephem_df['sigma_y'][0]
    initial_sigma_Z = spacex_ephem_df['sigma_z'][0]
    initial_sigma_XV = spacex_ephem_df['sigma_xv'][0]
    initial_sigma_YV = spacex_ephem_df['sigma_yv'][0]
    initial_sigma_ZV = spacex_ephem_df['sigma_zv'][0]
    cd = 2.2
    cr = 2
    cross_section = 10.0
    initial_t = spacex_ephem_df['UTC'][0]
    a_priori_estimate = np.array([initial_t, initial_X, initial_Y, initial_Z, initial_VX, initial_VY, initial_VZ,
                                  initial_sigma_X, initial_sigma_Y, initial_sigma_Z, initial_sigma_XV, initial_sigma_YV, initial_sigma_ZV,
                                    cr, cd, cross_section])
    #cast all the values except the first one to floats
    a_priori_estimate = np.array([float(i) for i in a_priori_estimate[1:]])
    a_priori_estimate = np.array([initial_t, *a_priori_estimate])

    observations_df = spacex_ephem_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
    observations_df = observations_df.iloc[:120]
    force_model_config =  {'enable_gravity': True, 'enable_third_body': True, 'enable_solar_radiation': True, 'enable_atmospheric_drag': True}

    optimized_state, associated_covariance = BLS_optimize(observations_df, force_model_config, a_priori_estimate)
    print(f"a priori estimate: {a_priori_estimate}")
    print(f"optimized state: {optimized_state}")
    print(f"associated covariance\n: {associated_covariance}")