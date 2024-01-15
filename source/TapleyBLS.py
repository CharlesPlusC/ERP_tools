import orekit
from orekit.pyhelpers import setup_orekit_curdir

orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()

import textwrap

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
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient, KnockeRediffusedForceModel
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
import seaborn as sns
from scipy.integrate import solve_ivp
from matplotlib.colors import SymLogNorm

SATELLITE_MASS = 500.0 #TBC (v1s are 250 , and v2mini are 800 or something?)
INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 15.0
POSITION_TOLERANCE = 1e-3 # 1 mm

def keys_to_string(d):
    """
    Convert the keys of a dictionary to a string with each key on a new line.
    """
    return '\n'.join(d.keys())

def configure_force_models(propagator,cr, cross_section,cd, **config_flags):

    # Earth gravity field with degree 64 and order 64
    if config_flags.get('enable_gravity', False):
        MU = Constants.WGS84_EARTH_MU
        ### monopole gravity model
        newattr = NewtonianAttraction(MU)
        propagator.addForceModel(newattr)

        ### 64x64 gravity model 
        gravityProvider = GravityFieldFactory.getNormalizedProvider(64, 64)#TODO: i think Vishal Ray's paper seems to suggest at least 80x80 would be good- double check
        gravityAttractionModel = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider)
        propagator.addForceModel(gravityAttractionModel)

    # Moon and Sun perturbations
    if config_flags.get('enable_third_body', False):
        moon = CelestialBodyFactory.getMoon()
        sun = CelestialBodyFactory.getSun()
        moon_3dbodyattraction = ThirdBodyAttraction(moon)
        sun_3dbodyattraction = ThirdBodyAttraction(sun)
        propagator.addForceModel(moon_3dbodyattraction)
        propagator.addForceModel(sun_3dbodyattraction)

    # Solar radiation pressure
    if config_flags.get('enable_solar_radiation', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        cross_section = float(cross_section)
        cr = float(cr)
        isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(cross_section, cr)
        solarRadiationPressure = SolarRadiationPressure(sun, wgs84Ellipsoid, isotropicRadiationSingleCoeff)
        propagator.addForceModel(solarRadiationPressure)

    # Atmospheric drag
    if config_flags.get('enable_atmospheric_drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        propagator.addForceModel(dragForce)

    if force_model_config.get('enable_relativity', False):
        relativity = Relativity(Constants.WGS84_EARTH_MU)
        propagator.addForceModel(relativity)

    if force_model_config.get('enable_erp_knocke', False):
        sun = CelestialBodyFactory.getSun()
        spacecraft = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        onedeg_in_rad = np.radians(1.0)
        angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
        knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
        propagator.addForceModel(knockeModel)

    # TODO: CERES ERP force model
    # if enable_ceres:
    #     ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data) # pass the time and radiation data to the force model

    return propagator

def propagate_state(start_date, end_date, initial_state_vector, cr=1.5, cd=1.8, cross_section=10.0, **config_flags):

    if len(initial_state_vector) == 6:
        x, y, z, vx, vy, vz = initial_state_vector
    elif len(initial_state_vector) == 7:
        x, y, z, vx, vy, vz, cd = initial_state_vector
    
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

    configured_propagator = configure_force_models(propagator,cr,cross_section, cd,**config_flags)
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
                        'sigma_xv', 'sigma_yv', 'sigma_zv']

    for col in columns_to_multiply:
        spacex_ephem_df[col] *= 1000

    covariance_columns = [f'cov_{i+1}' for i in range(21)]
    for col in covariance_columns:
        spacex_ephem_df[col] *= 1000**2

    spacex_ephem_df['hours'] = (spacex_ephem_df['JD'] - spacex_ephem_df['JD'][0]) * 24.0 # hours since first timestamp
    spacex_ephem_df['UTC'] = spacex_ephem_df['JD'].apply(jd_to_utc)
    # TODO: I am gaining 3 milisecond per minute in the UTC time. Why?
    return spacex_ephem_df

def rho_i(measured_state, measurement_type='state'):
# maps a state vector to a measurement vector
    if measurement_type == 'state':
        return measured_state
    #TODO: implement other measurement types

def stm_derivative(t, phi_flat, df_dy):
    """
    Computes the derivative of the flattened state transition matrix (STM).

    Parameters:
    - t: Time (not used in this function as the ODE is time-invariant, but required by solve_ivp)
    - phi_flat: Flattened state transition matrix (STM)
    - df_dy: Jacobian of the system dynamics (partial derivatives matrix)

    Returns:
    - Flattened derivative of the STM
    """
    n = int(np.sqrt(len(phi_flat)))
    phi = phi_flat.reshape((n, n))
    phi_dot = df_dy @ phi
    return phi_dot.flatten()

def propagate_STM(state_ti, t0, dt, phi_i, cr, cd, cross_section, **force_model_config):
    df_dy = np.zeros((len(state_ti),len(state_ti)))  # Initialize matrix of partial derivatives
    state_vector_data = (state_ti[0], state_ti[1], state_ti[2], state_ti[3], state_ti[4], state_ti[5])  # x, y, z, xv, yv, zv

    epochDate = datetime_to_absolutedate(t0)
    accelerations_t0 = np.zeros(3)
    force_models = []

    if force_model_config.get('enable_gravity', False):
        
        MU = Constants.WGS84_EARTH_MU
        monopolegrav = NewtonianAttraction(MU)
        force_models.append(monopolegrav)
        monopole_gravity_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, monopolegrav)
        monopole_gravity_eci_t0 = np.array([monopole_gravity_eci_t0[0].getX(), monopole_gravity_eci_t0[0].getY(), monopole_gravity_eci_t0[0].getZ()])
        accelerations_t0+=monopole_gravity_eci_t0

        gravityProvider = GravityFieldFactory.getNormalizedProvider(64, 64)
        gravityfield = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider)
        force_models.append(gravityfield)
        gravityfield_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, gravityfield)
        gravityfield_eci_t0 = np.array([gravityfield_eci_t0[0].getX(), gravityfield_eci_t0[0].getY(), gravityfield_eci_t0[0].getZ()])
        accelerations_t0+=gravityfield_eci_t0

    if force_model_config.get('enable_third_body', False):
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

    if force_model_config.get('enable_solar_radiation', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        cross_section = float(cross_section) 
        cr = float(cr) 
        isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(cross_section, cr)
        solarRadiationPressure = SolarRadiationPressure(sun, wgs84Ellipsoid, isotropicRadiationSingleCoeff)
        force_models.append(solarRadiationPressure)
        solar_radiation_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, solarRadiationPressure)
        solar_radiation_eci_t0 = np.array([solar_radiation_eci_t0[0].getX(), solar_radiation_eci_t0[0].getY(), solar_radiation_eci_t0[0].getZ()])
        accelerations_t0+=solar_radiation_eci_t0

    if force_model_config.get('enable_atmospheric_drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        force_models.append(dragForce)
        atmospheric_drag_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, dragForce)
        atmospheric_drag_eci_t0 = np.array([atmospheric_drag_eci_t0[0].getX(), atmospheric_drag_eci_t0[0].getY(), atmospheric_drag_eci_t0[0].getZ()])
        accelerations_t0+=atmospheric_drag_eci_t0

    if force_model_config.get('enable_relativity', False):
        relativity = Relativity(Constants.WGS84_EARTH_MU)
        force_models.append(relativity)
        relativity_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, relativity)
        relativity_eci_t0 = np.array([relativity_eci_t0[0].getX(), relativity_eci_t0[0].getY(), relativity_eci_t0[0].getZ()])
        accelerations_t0+=relativity_eci_t0

    if force_model_config.get('enable_erp_knocke', False):
        sun = CelestialBodyFactory.getSun()
        spacecraft = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        onedeg_in_rad = np.radians(1.0)
        angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
        knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
        force_models.append(knockeModel)
        knocke_eci_t0 = extract_acceleration(state_vector_data, epochDate, SATELLITE_MASS, knockeModel)
        knocke_eci_t0 = np.array([knocke_eci_t0[0].getX(), knocke_eci_t0[0].getY(), knocke_eci_t0[0].getZ()])
        accelerations_t0+=knocke_eci_t0
    
    for i in range(len(state_ti)):
        state_ti_perturbed = state_ti.copy()
        perturbation = 0.1
        state_ti_perturbed[i] += perturbation
        perturbed_accelerations = np.zeros(3)
        for force_model in force_models:
            if i < 6:  # Use perturbed state vector
                acc_perturbed = extract_acceleration(state_ti_perturbed, epochDate, SATELLITE_MASS, force_model)
            elif i == 6:
                # Recreate drag force model with perturbed Cd
                wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
                msafe = MarshallSolarActivityFutureEstimation(MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES, MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
                sun = CelestialBodyFactory.getSun()
                atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
                perturbed_isotropicDrag = IsotropicDrag(float(cross_section), float(state_ti_perturbed[-1]))                
                perturbed_dragForce = DragForce(atmosphere, perturbed_isotropicDrag)
                force_models[-1] = perturbed_dragForce # Replace the last force model with the perturbed drag force model
                acc_perturbed = extract_acceleration(state_ti_perturbed, epochDate, SATELLITE_MASS, force_model)
            if isinstance(acc_perturbed, np.ndarray): #TODO: hacky fix for the stupid output of extract_acceleration
                acc_perturbed_values = acc_perturbed
            else:
                acc_perturbed_values = np.array([acc_perturbed[0].getX(), acc_perturbed[0].getY(), acc_perturbed[0].getZ()])
            perturbed_accelerations += acc_perturbed_values
        partial_derivatives = (perturbed_accelerations - accelerations_t0) / perturbation

        # Assign partial derivatives to the appropriate submatrix
        if i < 3:  # Position components 
            df_dy[3:6, i] = partial_derivatives
        elif i < 6:  # Velocity components 
            df_dy[3:6, i] = partial_derivatives
            df_dy[i - 3, i] = 1  # Identity matrix in top-right 3x3 submatrix
        elif i == 6:  # Cd component
            df_dy[3:6, 6] = partial_derivatives

    # Flatten the initial STM for integration
    phi_i_flat = phi_i.flatten()

    # Set up time span for integration
    dt_seconds = float(dt.total_seconds())
    t_span = [0, dt_seconds]

    # Use solve_ivp to integrate the STM
    sol = solve_ivp(stm_derivative, t_span, phi_i_flat, args=(df_dy,), method='RK45')
    phi_t1 = sol.y[:, -1].reshape(phi_i.shape)

    return phi_t1

def calculate_cross_correlation_matrix(covariance_matrices):
    """
    Calculate cross-correlation matrices for a list of covariance matrices.

    Args:
    covariance_matrices (list of np.array): List of covariance matrices.

    Returns:
    List of np.array: List of cross-correlation matrices corresponding to each covariance matrix.
    """
    correlation_matrices = []
    for cov_matrix in covariance_matrices:
        # Ensure the matrix is a numpy array
        cov_matrix = np.array(cov_matrix)

        # Diagonal elements (variances)
        variances = np.diag(cov_matrix)

        # Standard deviations (sqrt of variances)
        std_devs = np.sqrt(variances)

        # Initialize correlation matrix
        corr_matrix = np.zeros_like(cov_matrix)

        # Calculate correlation matrix
        for i in range(len(cov_matrix)):
            for j in range(len(cov_matrix)):
                corr_matrix[i, j] = cov_matrix[i, j] / (std_devs[i] * std_devs[j])

        correlation_matrices.append(corr_matrix)

    return correlation_matrices

def OD_BLS(observations_df, force_model_config, a_priori_estimate=None, estimate_drag=False):

    # Initialize
    if estimate_drag==False:
        t0 = observations_df['UTC'].iloc[0]
        x_bar_0 = np.array(a_priori_estimate[1:7])  # x, y, z, u, v, w
        state_covs = a_priori_estimate[7:13]
    elif estimate_drag==True:
        t0 = observations_df['UTC'].iloc[0]
        x_bar_0 = np.array(a_priori_estimate[1:8]) #includes Cd
        state_covs = a_priori_estimate[8:14]

    #make covs a diagonal matrix
    state_covs = np.diag(state_covs)
    phi_ti_minus1 = np.identity(len(x_bar_0))  # n*n identity matrix for n state variables
    P_0 = np.array(state_covs, dtype=float)  # Covariance matrix from a priori estimate
    if estimate_drag:
        P_0 = np.pad(P_0, ((0, 1), (0, 1)), 'constant', constant_values=0)
        # Assign a non-zero variance to the drag coefficient to avoid non-invertible matrix
        initial_cd_variance = 1  # Setting an arbitrary value for now (but still high)
        P_0[-1, -1] = initial_cd_variance

    d_rho_d_state = np.eye(len(x_bar_0))  # Identity matrix: assume perfect state measurements
    H_matrix = np.empty((0, len(x_bar_0)))
    converged = False
    iteration = 1
    max_iterations = 10
    weighted_rms_last = np.inf 
    convergence_threshold = 0.001 
    no_times_diff_increased = 0

    all_residuals = np.empty((0, len(x_bar_0)))
    all_rms = []
    all_xbar_0s = []
    all_covs = []

    while not converged and iteration < max_iterations:
        print(f"Iteration: {iteration}")
        N = np.zeros(len(x_bar_0))  # reset N to zero
        lamda = np.linalg.inv(P_0)  # reset lambda to P_0
        y_all = np.empty((0, len(x_bar_0)))  # Initialize residuals array
        ti_minus1 = t0
        state_ti_minus1 = x_bar_0 
        RMSs = []
        for obs, row in observations_df.iterrows():
            ti = row['UTC']
            print(f"Observation {obs+1} of {len(observations_df)}")
            observed_state = row[['x', 'y', 'z', 'xv', 'yv', 'zv']].values
            #add Cd from the last iteration to the observed state
            if estimate_drag==True:
                observed_state = np.append(observed_state, x_bar_0[-1])
                obs_covariance = np.diag(row[['sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']])
                cd_covariance = 1  # TODO: is this even allowed? just setting a high value for now
                                     # TODO: do i want to reduce the covariance of Cd after each iteration?
                obs_covariance = np.pad(obs_covariance, ((0, 1), (0, 1)), 'constant', constant_values=0)
                obs_covariance[-1, -1] = cd_covariance
            else:
                obs_covariance = np.diag(row[['sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']])

            obs_covariance = np.array(obs_covariance, dtype=float) #convert to numpy array
            W_i = np.linalg.inv(obs_covariance)  # Weight matrix is the inverse of observation covariances
            
            # Propagate state and STM
            dt = ti - ti_minus1
            if estimate_drag==True:
                state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1[:6], cr=1.5, cd=state_ti_minus1[-1], cross_section=10.0, **force_model_config)
                phi_ti = propagate_STM(state_ti_minus1, ti, dt, phi_ti_minus1, cr=1.5, cd=state_ti_minus1[-1], cross_section=10.0, **force_model_config)
            else:
                state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1, cr=1.5, cd=2.2, cross_section=10.0, **force_model_config)
                phi_ti = propagate_STM(state_ti_minus1, ti, dt, phi_ti_minus1, cr=1.5, cd=2.2, cross_section=10.0, **force_model_config)

            # Compute H matrix for this observation
            H_matrix_row = d_rho_d_state @ phi_ti
            H_matrix = np.vstack([H_matrix, H_matrix_row]) # Append row to H matrix
            if estimate_drag==True:
                state_ti = np.append(state_ti, state_ti_minus1[-1]) #add Cd to state_ti
            y_i = observed_state - rho_i(state_ti, 'state')
            y_i = np.array(y_i, dtype=float)
            y_all = np.vstack([y_all, y_i])
            
            # Update lambda and N matrices
            lamda += H_matrix_row.T @ W_i @ H_matrix_row
            N += H_matrix_row.T @ W_i @ y_i

            # Update for next iteration
            ti_minus1 = ti
            state_ti_minus1 = np.array(state_ti)
            phi_ti_minus1 = phi_ti
            RMSs.append(y_i.T @ W_i @ y_i)

        # sum all the RMSs
        RMSs = np.array(RMSs)
        #weighted RMS is sqrt(RMS divided by m. where m is (number of observations * number of state variables))
        weighted_rms = np.sqrt(np.sum(RMSs) / (len(x_bar_0) * len (y_all)))
        print(f"Weighted RMS: {weighted_rms}")
        # Solve normal equations
        xhat = np.linalg.inv(lamda) @ N
        # Check for convergence
        if abs(weighted_rms - weighted_rms_last) < convergence_threshold:
            print("Converged!")
            converged = True

        else:
            if weighted_rms > weighted_rms_last:
                no_times_diff_increased += 1
                print(f"RMS increased {no_times_diff_increased} times in a row.")
                if no_times_diff_increased >= 2:
                    print("RMS increased 2 times in a row. Stopping iteration.")
                    break
                #TODO: make it so that it takes the run with the best RMS if it doesn't converge ?
            else:
                no_times_diff_increased = 0 #reset the counter
            weighted_rms_last = weighted_rms
            
            x_bar_0 += xhat  # Update nominal trajectory
            print(f"Estimated state: {x_bar_0}")

        all_residuals = np.vstack([all_residuals, y_all])
        all_rms.append(weighted_rms)
        all_xbar_0s.append(x_bar_0)
        all_covs.append(np.linalg.inv(lamda))
        iteration += 1

    return x_bar_0, np.linalg.inv(lamda), all_residuals, weighted_rms


if __name__ == "__main__":
    spacex_ephem_df_full = spacex_ephem_to_df_w_cov('external/ephems/starlink/MEME_57632_STARLINK-30309_3530645_Operational_1387262760_UNCLASSIFIED.txt')

    spacex_ephem_df = spacex_ephem_df_full.iloc[2000:2200]

    # Initialize state vector from the first point in the SpaceX ephemeris
    # TODO: Can perturb this initial state vector to test convergence later
    initial_X = spacex_ephem_df['x'].iloc[0]
    initial_Y = spacex_ephem_df['y'].iloc[0]
    initial_Z = spacex_ephem_df['z'].iloc[0]
    initial_VX = spacex_ephem_df['xv'].iloc[0]
    initial_VY = spacex_ephem_df['yv'].iloc[0]
    initial_VZ = spacex_ephem_df['zv'].iloc[0]
    initial_sigma_X = spacex_ephem_df['sigma_x'].iloc[0]
    initial_sigma_Y = spacex_ephem_df['sigma_y'].iloc[0]
    initial_sigma_Z = spacex_ephem_df['sigma_z'].iloc[0]
    initial_sigma_XV = spacex_ephem_df['sigma_xv'].iloc[0]
    initial_sigma_YV = spacex_ephem_df['sigma_yv'].iloc[0]
    initial_sigma_ZV = spacex_ephem_df['sigma_zv'].iloc[0]
    cd = 2.2
    cr = 1.5
    cross_section = 30.0 # from https://lilibots.blogspot.com/2020/04/starlink-satellite-dimension-estimates.html
    initial_t = spacex_ephem_df['UTC'].iloc[0]
    a_priori_estimate = np.array([initial_t, initial_X, initial_Y, initial_Z, initial_VX, initial_VY, initial_VZ, cd,
                                  initial_sigma_X, initial_sigma_Y, initial_sigma_Z, initial_sigma_XV, initial_sigma_YV, initial_sigma_ZV,
                                    ])
    #cast all the values except the first one to floats
    a_priori_estimate = np.array([float(i) for i in a_priori_estimate[1:]])
    a_priori_estimate = np.array([initial_t, *a_priori_estimate])
    
    observations_df_full = spacex_ephem_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
    # obs_lengths_to_test = [10, 20, 35, 50, 75, 100, 120]
    obs_lengths_to_test = [100]
    estimate_drag = True
    force_model_configs = [
        {'enable_gravity': True, 'enable_third_body': True, 'enable_atmospheric_drag': True},
        {'enable_gravity': True, 'enable_third_body': True, 'enable_atmospheric_drag': True, 'enable_solar_radiation': True},
        {'enable_gravity': True, 'enable_third_body': True, 'enable_atmospheric_drag': True, 'enable_solar_radiation': True, 'enable_erp_knocke': True},
        ]

    covariance_matrices = []
    optimized_states = []
    for i, force_model_config in enumerate(force_model_configs):
        if not force_model_config.get('enable_atmospheric_drag', False):
            estimate_drag = False
            raise Warning("Force model doesn't have drag. Setting estimate_drag to False.")
        
        for obs_length in obs_lengths_to_test:
            observations_df = observations_df_full.iloc[:obs_length]
            optimized_states, cov_mats, residuals, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag=estimate_drag)
            
            #find the index of the minimum RMS
            min_RMS_index = np.argmin(RMSs)
            optimized_state = optimized_states[min_RMS_index]
            covariance_matrix = cov_mats[min_RMS_index]
            final_RMS = RMSs[min_RMS_index]
            residuals_final = residuals[min_RMS_index]

            covariance_matrices.append(covariance_matrix)

            plt.figure()
            plt.scatter(observations_df['UTC'], residuals_final[:,0], s=3, label='x', c = "xkcd:blue")
            plt.scatter(observations_df['UTC'], residuals_final[:,1], s=3, label='y', c = "xkcd:green")
            plt.scatter(observations_df['UTC'], residuals_final[:,2], s=3, label='z', c = "xkcd:red")
            plt.scatter(observations_df['UTC'], residuals_final[:,3], s=3, label='xv', c = "xkcd:purple")
            plt.scatter(observations_df['UTC'], residuals_final[:,4], s=3, label='yv', c = "xkcd:orange")
            plt.scatter(observations_df['UTC'], residuals_final[:,5], s=3, label='zv', c = "xkcd:yellow")
            plt.title("Residuals (O-C) for final BLS iteration")
            plt.xlabel("Observation time (UTC)")
            plt.xticks(rotation=45)
            plt.ylabel("Residual (m)")
            plt.text(0.05, 0.95, f"Weighted RMS: {final_RMS:.2f}", transform=plt.gca().transAxes)
            force_model_keys_str = keys_to_string(force_model_config)
            wrapped_text = textwrap.fill(force_model_keys_str, 40)
            plt.text(0.05, 0.90, f"Force model:\n{wrapped_text}", transform=plt.gca().transAxes)
            if len(observations_df) <= 60:
                plt.ylim(-2,2)
            elif len(observations_df) <= 100:
                plt.ylim(-15,15)
            plt.grid(True)
            plt.legend(['x', 'y', 'z', 'xv', 'yv', 'zv'])
            save_to = f"output/OD_BLS/Tapley/estimation_experiment/fmodel_{i}_#pts_{len(observations_df)}.png"
            if estimate_drag:
                save_to = f"output/OD_BLS/Tapley/estimation_experiment/estim_drag_fmodel_{i}_#pts_{len(observations_df)}_.png"
            plt.savefig(save_to)

            #plot histograms of residuals
            plt.figure()
            plt.hist(residuals_final[:,0], bins=20, label='x', color="xkcd:blue")
            plt.hist(residuals_final[:,1], bins=20, label='y', color="xkcd:green")
            plt.hist(residuals_final[:,2], bins=20, label='z', color="xkcd:red")
            plt.hist(residuals_final[:,3], bins=20, label='xv', color="xkcd:purple")
            plt.hist(residuals_final[:,4], bins=20, label='yv', color="xkcd:orange")
            plt.hist(residuals_final[:,5], bins=20, label='zv', color="xkcd:yellow")
            plt.title("Residuals (O-C) for final BLS iteration")
            plt.xlabel("Residual (m)")
            plt.ylabel("Frequency")
            force_model_keys_str = keys_to_string(force_model_config)
            wrapped_text = textwrap.fill(force_model_keys_str, 40)
            plt.text(0.05, 0.90, f"Force model:\n{wrapped_text}", transform=plt.gca().transAxes)
            plt.legend(['x', 'y', 'z', 'xv', 'yv', 'zv'])
            plt.savefig(f"output/OD_BLS/Tapley/estimation_experiment/hist_force_model_{i}_#pts_{len(observations_df)}.png")

            #ECI covariance matrix
            if estimate_drag:
                labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel', 'Cd']
            else:
                labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel']
            log_norm = SymLogNorm(linthresh=1e-10, vmin=covariance_matrix.min(), vmax=covariance_matrix.max())
            plt.figure(figsize=(8, 7))
            sns.heatmap(covariance_matrix, annot=True, fmt=".3e", xticklabels=labels, yticklabels=labels, cmap="viridis", norm=log_norm)
            plt.title(f"No. obs:{len(observations_df)}, force model:{i}")
            force_model_keys_str = keys_to_string(force_model_config)
            wrapped_text = textwrap.fill(force_model_keys_str, 40)
            plt.text(0.05, 0.90, f"Force model:\n{wrapped_text}", transform=plt.gca().transAxes)
            save_to = f"output/OD_BLS/Tapley/estimation_experiment/covMat_#pts_{len(observations_df)}_config{i}.png"
            if estimate_drag:
                save_to = f"output/OD_BLS/Tapley/estimation_experiment/covMat_#pts_{len(observations_df)}_config{i}_estim_drag.png"
            plt.savefig(save_to)
    
    relative_differences = []
    for j in range(1, len(covariance_matrices)):
        difference_matrix = covariance_matrices[j] - covariance_matrices[j-1]
        if estimate_drag:
            labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel', 'Cd']
        else:
            labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel']
        # Plotting the difference as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(difference_matrix, annot=True, fmt=".3e",xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
        plt.title(f'Difference in Covariance Matrix: Run {j} vs Run {j-1}')
        plt.savefig(f'output/OD_BLS/Tapley/estimation_experiment/diff_covmat_run_{j}_vs_{j-1}.png')

    correlation_matrices =  calculate_cross_correlation_matrix(covariance_matrices)
    for j in range(len(correlation_matrices)):
        # Plotting the correlation matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrices[j], annot=True, fmt=".3f", cmap="coolwarm", center=0)
        plt.title(f'Correlation Matrix: Run {j}')
        plt.xlabel('Components')
        plt.ylabel('Components')
        plt.savefig(f'output/OD_BLS/Tapley/estimation_experiment/corr_covmat_run_{j}.png')

    # Plot a bar chart of the Cd values
    plt.figure()
    bars = plt.bar(np.arange(len(optimized_states)), [i[-1] for i in optimized_states])

    # Adding labels on each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', va='bottom', ha='center')

    plt.xticks(np.arange(len(optimized_states)), [f"Run {i}" for i in range(len(optimized_states))])
    plt.title("Cd values estimated by BLS under different force models")
    plt.savefig(f"output/OD_BLS/Tapley/estimation_experiment/Cd_values_#fmodels_{len(optimized_states)}_#pts_{len(observations_df)}.png")
