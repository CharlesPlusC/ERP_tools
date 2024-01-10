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
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, Relativity
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

SATELLITE_MASS = 800.0
INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 15.0
POSITION_TOLERANCE = 1e-3 # 1 mm

def configure_force_models(propagator,cr, cross_section,cd, **config_flags):
    # Earth gravity field with degree 64 and order 64
    if config_flags.get('enable_gravity', True):
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

    # Relativity
    if config_flags.get('enable_relativity', True):
        relativity = Relativity(orekit_constants.EIGEN5C_EARTH_MU)
        propagator.addForceModel(relativity)

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

def propagate_state(start_date, end_date, initial_state_vector, cr=None, cd=None, cross_section=None, **config_flags):
    
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

def propagate_STM(state_ti, dt, phi_i):

    df_dy = np.zeros((len(state_ti),len(state_ti))) #initialize matrix of partial derivatives (partials at time t0)
    # numerical estimation of partial derivatives
    # get the state at ti and the accelerations at ti
    state_vector_data = tuple(state_ti[0], state_ti[1], state_ti[2], state_ti[3], state_ti[4], state_ti[5])
    t0 = state_ti[0]
    gravity_eci = extract_acceleration(state_vector_data, TLE_epochDate, SATELLITE_MASS, HolmesFeatherstoneAttractionModel)

    # perturb each state variable by a small amount and get the new accelerations
    # new accelerations - old accelerations / small amount = partial derivative

    # check -> the first 3*3 should just be 0s
    # check -> the last 3*3 should just be 0s
    # check -> the top right hand 3*3 should be the identity matrix
    # check -> the bottom left hand 3*3 should is where the partial derivatives are changing

    phi_t1 = phi_i + df_dy * phi_i * dt #STM at time t1

    # get d_rho/d_state -> for perfect state measurements this is just ones (include for compelteness)
    # check -> dimensions of d_rho/d_state are 1 x len(state_t)

    # d_rho/d_state * phi_t1 = one row in the H matrix
    pass

def BLS_optimize(observations_df, force_model_config, a_priori_estimate=None):
    """
    Batch Least Squares orbit determination algorithm.

    Parameters
    ----------
    state_vector : np.array
        Initial state vector. Must be in the form [t, x, y, z, u, v, w, cd].
    observations_df : pd.DataFrame
        Observations dataframe. Must have columns: ['UTC', 'x', 'y', 'z', 'u', 'v', 'w', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv'].
    force_model_config : dict
        Dictionary containing force model configuration parameters.
    t0 : float
        Initial time of orbit determination.
    tfinal : float
        Final time of orbit determination.
    a_priori_estimate : np.array, optional
        A priori state estimate. The default is None.

    Returns
    -------
    None.

    """
    
    #observations must be in the form of a pandas dataframe with columns:
    #   t, x, y, z, u, v, w, sigma_x, sigma_y, sigma_z, sigma_u, sigma_v, sigma_w
    assert isinstance(observations_df, pd.DataFrame), "observations must be a pandas dataframe"
    assert len(observations_df.columns) == 13, "observations dataframe must have 13 columns"
    required_obs_cols = ['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']
    assert all([col in observations_df.columns for col in required_obs_cols]), f"observations dataframe must have columns: {required_obs_cols}"
    
    ### 1) initialize iteration
    iteration = 1
    t0 = observations_df['UTC'][0]
    ti_minus1 = t0
    ti = ti_minus1
    print(f"ti_minus1: {ti_minus1}")
    print(f"t0: {t0}")
    state_t0 = observations_df[['x', 'y', 'z', 'xv', 'yv', 'zv']].iloc[0].values
    state_ti_minus1 = state_t0
    phi_t0 = np.identity(len(state_t0)) #STM
    phi_ti_minus1 = phi_t0 

    #from a priori estimate
    x_bar_0 = np.array(a_priori_estimate[1:7])
    a_priori_sigmas = a_priori_estimate[7:13]
    P_0 = (np.diag(np.array(a_priori_sigmas)))
    P_0_inv = np.linalg.inv(P_0)
    N = np.linalg.inv(P_0) * x_bar_0

    ### 2) read next observation (corresping to time ti)
    observation_time = observations_df['UTC'][1]
    observed_state = observations_df[['x', 'y', 'z', 'xv', 'yv', 'zv']].iloc[1].values
    obs_covariance = observations_df[['sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']].iloc[1].values

    # in Tapley notation
    ti = observation_time
    yi = observed_state
    ri = obs_covariance

    ### 3) propagate state and STM from ti_minus1 to ti
    #Start with the state at t0 and propagate to t1
    state_t0 = propagate_state(start_date=ti_minus1, end_date=t0, initial_state_vector=state_ti_minus1, cr=cr, cd=cd, cross_section=cross_section, **force_model_config)
    print("propagated state from t-1 to t0")
    print(f"difference between propagated state and state at t0: {(state_t0 - observed_state)}")

    phi_ti = propagate_STM(phi_ti_minus1, ti)

    ### 4) compute H-matrix
        # H-tilde is the observation-state mapping matrix
        H_tilde_i = dg/d_state #where g is the observation-state relationship along the reference trajectory
        d_rho/d_state * phi_t1 = one row in the H matrix 
        y_i = yi - rho_i(state_ti) #residual (except for perfect state measurements this is just yi)
        H_i = H_tilde_i * phi_ti
        lamda = lamda + H_i.T * np.linalg.inv(ri) * H_i
        N = N + H_i.T * np.linalg.inv(ri) * y_i

    ### 5) Time check
        if ti < tfinal:
            i = i + 1
            ti_minus1 = ti
            state_ti_minus1 = state_ti
            phi_ti_minus1 = phi_ti
            # go to step 2
        if ti>=tfinal:
            # solve normal equations
            # lamda_xhat = N
            lamda_0 = H_i.T * np.linalg.inv(ri) * H_i + np.linalg.inv(P_0)
            xhat_0 = np.linalg.inv(lamda_0) * N
            P_0 = np.linalg.inv(lamda)

    ### 6) convergence check
        residuals = yi - H_i * xhat_0
        if np.linalg.norm(lamda_xhat - lamda_xhat_old) < 0.01:
            break
        else:
            #update nominal trajectory
            state_t0 = state_t0 + lamda_xhat
            x_bar_0 = x_bar_0 - lamda_xhat

            # use original value of P_0

            # go to step 1

    pass

if __name__ == "__main__":
    spacex_ephem_df = spacex_ephem_to_df_w_cov('external/ephems/starlink/MEME_57632_STARLINK-30309_3530645_Operational_1387262760_UNCLASSIFIED.txt')

    # Initialize state vector from the first point in the SpaceX ephemeris
    # TODO: Can perturb this initial state vector to test convergence later
    initial_X = spacex_ephem_df['x'][0]*1000
    initial_Y = spacex_ephem_df['y'][0]*1000
    initial_Z = spacex_ephem_df['z'][0]*1000
    initial_VX = spacex_ephem_df['xv'][0]*1000
    initial_VY = spacex_ephem_df['yv'][0]*1000
    initial_VZ = spacex_ephem_df['zv'][0]*1000
    initial_sigma_X = spacex_ephem_df['sigma_x'][0]*1000
    initial_sigma_Y = spacex_ephem_df['sigma_y'][0]*1000
    initial_sigma_Z = spacex_ephem_df['sigma_z'][0]*1000
    initial_sigma_XV = spacex_ephem_df['sigma_xv'][0]*1000
    initial_sigma_YV = spacex_ephem_df['sigma_yv'][0]*1000
    initial_sigma_ZV = spacex_ephem_df['sigma_zv'][0]*1000
    cd = 2.2
    cr = 1.5
    cross_section = 10.0
    initial_t = spacex_ephem_df['UTC'][0]
    a_priori_estimate = np.array([initial_t, initial_X, initial_Y, initial_Z, initial_VX, initial_VY, initial_VZ,
                                  initial_sigma_X, initial_sigma_Y, initial_sigma_Z, initial_sigma_XV, initial_sigma_YV, initial_sigma_ZV,
                                    cr, cd, cross_section])
    #cast all the values except the first one to floats
    a_priori_estimate = np.array([float(i) for i in a_priori_estimate[1:]])

    observations_df = spacex_ephem_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]

    force_model_config =  {'enable_gravity': True, 'enable_third_body': True, 'enable_solar_radiation': True, 'enable_atmospheric_drag': True}

    BLS_optimize(observations_df, force_model_config, a_priori_estimate)