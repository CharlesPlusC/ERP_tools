import orekit
from orekit.pyhelpers import setup_orekit_curdir

orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()

from tools.spaceX_ephem_tools import  parse_spacex_datetime_stamps
from tools.utilities import yyyy_mm_dd_hh_mm_ss_to_jd, jd_to_utc
from tools.CERES_ERP import CERES_ERP_ForceModel

import pandas as pd
import numpy as np
import scipy
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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

def std_dev_from_lower_triangular(lower_triangular_data):
    cov_matrix = np.zeros((6, 6))
    row, col = np.tril_indices(6)
    cov_matrix[row, col] = lower_triangular_data
    cov_matrix = cov_matrix + cov_matrix.T - np.diag(cov_matrix.diagonal())
    std_dev = np.sqrt(np.diag(cov_matrix))
    return std_dev

def pos_vel_from_orekit_ephem(ephemeris, initial_date, end_date, step):
    times = []
    state_vectors = []  # Store position and velocity vectors

    current_date = initial_date
    while current_date.compareTo(end_date) <= 0:
        state = ephemeris.propagate(current_date)
        position = state.getPVCoordinates().getPosition().toArray()
        velocity = state.getPVCoordinates().getVelocity().toArray()
        state_vector = np.concatenate([position, velocity])  # Combine position and velocity

        times.append(current_date.durationFrom(initial_date))
        state_vectors.append(state_vector)

        current_date = current_date.shiftedBy(step)

    return times, state_vectors

def generate_ephemeris_and_extract_data(propagator, start_date, end_date, time_step):

    ephemeris = propagator.getEphemerisGenerator().getGeneratedEphemeris()
    times, state_vectors = pos_vel_from_orekit_ephem(ephemeris, start_date, end_date, time_step)

    return (times, state_vectors)

def propagate_state_using_propagator(propagator, start_date, end_date, initial_state_vector, frame):

    x, y, z, vx, vy, vz = initial_state_vector
    initial_orbit = CartesianOrbit(PVCoordinates(Vector3D(float(x), float(y), float(z)),
                                                Vector3D(float(vx), float(vy), float(vz))),
                                    frame,
                                    start_date,
                                    Constants.WGS84_EARTH_MU)
    

    initial_state = SpacecraftState(initial_orbit)
    propagator.setInitialState(initial_state)
    final_state = propagator.propagate(end_date)

    pv_coordinates = final_state.getPVCoordinates()
    position = [pv_coordinates.getPosition().getX(), pv_coordinates.getPosition().getY(), pv_coordinates.getPosition().getZ()]
    velocity = [pv_coordinates.getVelocity().getX(), pv_coordinates.getVelocity().getY(), pv_coordinates.getVelocity().getZ()]

    return position + velocity

def configure_force_models(propagator,cr, cd, cross_section, enable_gravity=True, enable_third_body=True,
                        enable_solar_radiation=True, enable_relativity=True, enable_atmospheric_drag=True, enable_ceres=True):
    # Earth gravity field with degree 64 and order 64
    if enable_gravity:
        gravityProvider = GravityFieldFactory.getNormalizedProvider(64, 64)
        gravityAttractionModel = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider)
        propagator.addForceModel(gravityAttractionModel)

    # Moon and Sun perturbations
    if enable_third_body:
        moon = CelestialBodyFactory.getMoon()
        sun = CelestialBodyFactory.getSun()
        moon_3dbodyattraction = ThirdBodyAttraction(moon)
        sun_3dbodyattraction = ThirdBodyAttraction(sun)
        propagator.addForceModel(moon_3dbodyattraction)
        propagator.addForceModel(sun_3dbodyattraction)

    # Solar radiation pressure
    if enable_solar_radiation:
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        cross_section = float(cross_section)
        cr = float(cr)
        isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(cross_section, cr)
        solarRadiationPressure = SolarRadiationPressure(sun, wgs84Ellipsoid, isotropicRadiationSingleCoeff)
        propagator.addForceModel(solarRadiationPressure)

    # Relativity
    if enable_relativity:
        relativity = Relativity(orekit_constants.EIGEN5C_EARTH_MU)
        propagator.addForceModel(relativity)

    # Atmospheric drag
    if enable_atmospheric_drag:
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

def create_symmetric_corr_matrix(lower_triangular_data):
    # Create the symmetric covariance matrix
    cov_matrix = np.zeros((6, 6))
    row, col = np.tril_indices(6)
    cov_matrix[row, col] = lower_triangular_data
    cov_matrix = cov_matrix + cov_matrix.T - np.diag(cov_matrix.diagonal())

    # Convert to correlation matrix
    std_dev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = np.divide(cov_matrix, std_dev[:, None])
    corr_matrix = np.divide(corr_matrix, std_dev[None, :])
    np.fill_diagonal(corr_matrix, 1)  # Fill diagonal with 1s for self-correlation
    return corr_matrix

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
    sigma_xs = []
    sigma_ys = []
    sigma_zs = []
    sigma_us = []
    sigma_vs = []
    sigma_ws = []

    # Calculate averaged standard deviations for each row
    for _, row in pd.DataFrame(covariance_data).iterrows():
        std_devs = std_dev_from_lower_triangular(row.values)
        sigma_xs.append(std_devs[0])
        sigma_ys.append(std_devs[1])
        sigma_zs.append(std_devs[2])
        sigma_us.append(std_devs[3])
        sigma_vs.append(std_devs[4])
        sigma_ws.append(std_devs[5])

    # Construct the DataFrame with all data
    spacex_ephem_df = pd.DataFrame({
        't': t,
        'x': x,
        'y': y,
        'z': z,
        'u': u,
        'v': v,
        'w': w,
        'JD': jd_stamps,
        'sigma_xs': sigma_xs,
        'sigma_ys': sigma_ys,
        'sigma_zs': sigma_zs,
        'sigma_us': sigma_us,
        'sigma_vs': sigma_vs,
        'sigma_ws': sigma_ws,
        **covariance_data
    })

    spacex_ephem_df['hours'] = (spacex_ephem_df['JD'] - spacex_ephem_df['JD'][0]) * 24.0 # hours since first timestamp
    # calculate UTC time by applying jd_to_utc() to each JD value
    spacex_ephem_df['UTC'] = spacex_ephem_df['JD'].apply(jd_to_utc)
    # TODO: I am gaining 3 milisecond per minute in the UTC time. Why?

    return spacex_ephem_df

def main():

    spacex_ephem_dfwcov = spacex_ephem_to_df_w_cov('external/ephems/starlink/MEME_57632_STARLINK-30309_3530645_Operational_1387262760_UNCLASSIFIED.txt')
    #convert the MEME coordinates to GCRF coordinates
    SATELLITE_MASS = 800.0
    INTEGRATOR_MIN_STEP = 0.001
    INTEGRATOR_MAX_STEP = 15.0
    INTEGRATOR_INIT_STEP = 15.0
    POSITION_TOLERANCE = 1e-5

    sat_list = {    
    'STARLINK-30309': {
        'norad_id': 57632,  # For Space-Track TLE queries
        'cospar_id': '2023-122A',  # For laser ranging data queries
        'sic_id': '000',  # For writing in CPF files
        'mass': 800.0, # kg; v2 mini
        'cross_section': 10.0, # m2; TODO: get proper value
        'cd': 1.5, # TODO: compute proper value
        'cr': 2.2  # TODO: compute proper value
                    }
    }

    sc_name = 'STARLINK-30309'  # Change the name to select a different satellite in the dict

    j2000 = FramesFactory.getEME2000()
    eci = j2000

    # Set the initial conditions (manually taken from SpaceX ephemeris)
    odDate = datetime(2023, 12, 19, 6, 45, 42, 00000)
    Orbit0_epoch = datetime_to_absolutedate(odDate)

    # Initialize state vector from the first point in the SpaceX ephemeris
    initial_X = spacex_ephem_dfwcov['x'][0]*1000
    initial_Y = spacex_ephem_dfwcov['y'][0]*1000
    initial_Z = spacex_ephem_dfwcov['z'][0]*1000
    initial_VX = spacex_ephem_dfwcov['u'][0]*1000
    initial_VY = spacex_ephem_dfwcov['v'][0]*1000
    initial_VZ = spacex_ephem_dfwcov['w'][0]*1000
    state_vector = np.array([initial_X, initial_Y, initial_Z, initial_VX, initial_VY, initial_VZ])
    
    Orbit0_ECI = CartesianOrbit(PVCoordinates(Vector3D(float(state_vector[0]), float(state_vector[1]), float(state_vector[2])),
                                            Vector3D(float(state_vector[3]), float(state_vector[4]), float(state_vector[5]))),
                                eci,
                                Orbit0_epoch,
                                Constants.WGS84_EARTH_MU)

    configurations = [
        {'enable_gravity': True, 'enable_third_body': False, 'enable_solar_radiation': False, 'enable_atmospheric_drag': False},
        {'enable_gravity': True, 'enable_third_body': True, 'enable_solar_radiation': False, 'enable_atmospheric_drag': False},
        {'enable_gravity': True, 'enable_third_body': True, 'enable_solar_radiation': True, 'enable_atmospheric_drag': False},
        {'enable_gravity': True, 'enable_third_body': True, 'enable_solar_radiation': True, 'enable_atmospheric_drag': True},
    ]

    propagators = []
    for config in configurations:
        initialOrbit = Orbit0_ECI
        tolerances = NumericalPropagator.tolerances(POSITION_TOLERANCE, initialOrbit, initialOrbit.getType())
        integrator = DormandPrince853Integrator(INTEGRATOR_MIN_STEP, INTEGRATOR_MAX_STEP, JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
        integrator.setInitialStepSize(INTEGRATOR_INIT_STEP)
        initialState = SpacecraftState(initialOrbit, SATELLITE_MASS)
        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(OrbitType.CARTESIAN)
        propagator.setInitialState(initialState)
        configured_propagator = configure_force_models(propagator,sat_list[sc_name]['cr'], sat_list[sc_name]['cd'],
                                                              sat_list[sc_name]['cross_section'], **config)
        propagators.append(configured_propagator)


    max_iterations = 15
    points_to_use = 15  # Number of observations to use
    convergence_threshold = 0.05

    # Initialize dictionaries for storing results
    Delta_xs_dict = {}
    Residuals_dict = {}

    num_propagators = len(propagators)
    num_iterations = max_iterations
    num_timesteps = points_to_use
    num_components = 3  # X, Y, Z

    # Initialize 4D lists to store observed and propagated positions
    all_itx_observed_positions = np.zeros((num_propagators, num_iterations, num_timesteps, num_components))
    all_itx_propagated_positions = np.zeros((num_propagators, num_iterations, num_timesteps, num_components))
    
    cov_matx_dict = {}
    true_state = np.array([spacex_ephem_dfwcov['x'][0]*1000, spacex_ephem_dfwcov['y'][0]*1000, spacex_ephem_dfwcov['z'][0]*1000, spacex_ephem_dfwcov['u'][0]*1000, spacex_ephem_dfwcov['v'][0]*1000, spacex_ephem_dfwcov['w'][0]*1000])
    for idx, configured_propagator in enumerate(propagators):
        propagator = configured_propagator
        apriori_state_vector = state_vector.copy()
        estimation_errors = []

        # Initial covariance matrix (assumed to be diagonal with large values)
        Pk = np.eye(6) * 1e8

        for iteration in range(max_iterations):
            print(f'Iteration {iteration + 1}')

            # Initializing matrices for the iteration
            total_residuals_vector = np.zeros((0, 1))
            total_design_matrix = np.zeros((0, 6))  # 6 parameters to estimate
            total_observation_cov_matrix = np.zeros((0, 0))

            for i, row in spacex_ephem_dfwcov.head(points_to_use).iterrows():
                measurement_epoch = datetime_to_absolutedate(row['UTC'].to_pydatetime())
                propagated_state = propagate_state_using_propagator(propagator, Orbit0_epoch, measurement_epoch, apriori_state_vector, frame=eci)
                observed_state = np.array([row['x']*1000, row['y']*1000, row['z']*1000, row['u']*1000, row['v']*1000, row['w']*1000])
                
                # Storing the observed and propagated positions for each timestep
                all_itx_observed_positions[idx, iteration, i, :] = observed_state[:3]
                all_itx_propagated_positions[idx, iteration, i, :] = propagated_state[:3]

                # Compute residual (Observed - Computed)
                residual = observed_state - propagated_state
                residuals_vector = residual.reshape(-1, 1)
                total_residuals_vector = np.vstack([total_residuals_vector, residuals_vector])

                # Construct observation covariance matrix for this point
                sigma_vec = np.array([row['sigma_xs'], row['sigma_ys'], row['sigma_zs'], row['sigma_us'], row['sigma_vs'], row['sigma_ws']]) * 1000
                observation_cov_matrix = np.diag(sigma_vec ** 2)
                
                total_observation_cov_matrix = scipy.linalg.block_diag(total_observation_cov_matrix, observation_cov_matrix)

                # Sensitivity matrix (partial derivatives), here assumed as identity
                design_matrix = np.identity(6)
                total_design_matrix = np.vstack([total_design_matrix, design_matrix])

            print(f'Magnitude of pos residuals before update: {np.linalg.norm(total_residuals_vector[:3])}')
            print(f'Magnitude of vel residuals before update: {np.linalg.norm(total_residuals_vector[3:])}')
            # Weight matrix (inverse of observation covariance matrix)
            Wk = np.linalg.inv(total_observation_cov_matrix)

            # Least squares calculations
            HtWH = total_design_matrix.T @ Wk @ total_design_matrix
            HtWY = total_design_matrix.T @ Wk @ total_residuals_vector

            # Update state vector
            delta_X = np.linalg.inv(HtWH) @ HtWY
            apriori_state_vector += delta_X.flatten()
            Delta_xs_dict[iteration] = delta_X.flatten()            
            # Calculate and store the covariance matrix for the current iteration
            estimation_error = apriori_state_vector - true_state
            estimation_errors.append(estimation_error)

            # Check for convergence
            if np.linalg.norm(delta_X) < convergence_threshold:
                print("Convergence achieved.")
                break

        estimation_errors_matrix = np.array(estimation_errors)
        covariance_matrix = np.cov(estimation_errors_matrix.T)
        cov_matx_dict[idx] = covariance_matrix

        # Store the residuals for this configuration
        Residuals_dict[idx] = total_residuals_vector
    
    def covariance_to_correlation(cov_matrix):
        """Convert a covariance matrix to a correlation matrix."""
        d = np.sqrt(np.diag(cov_matrix))
        d_inv = np.reciprocal(d, where=d!=0)  # Avoid division by zero
        corr_matrix = cov_matrix * np.outer(d_inv, d_inv)
        return corr_matrix

    num_matrices = len(cov_matx_dict)
    labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel']

    # Convert covariance matrices to correlation matrices
    for idx, cov_matrix in cov_matx_dict.items():
        corr_matrix = covariance_to_correlation(cov_matrix)
        cov_matx_dict[idx] = corr_matrix

    # Plotting
    plt.figure(figsize=(8 * num_matrices, 7))
    for idx, corr_matrix in cov_matx_dict.items():
        plt.subplot(1, num_matrices, idx + 1)
        plt.rcParams.update({'font.size': 8})  # Set font size
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="viridis")
        plt.title(f'Correlation Matrix: {idx}')

    plt.tight_layout()
    plt.show()

    # Convert lists to numpy arrays for easier handling
    all_itx_observed_positions = np.array(all_itx_observed_positions)
    all_itx_propagated_positions = np.array(all_itx_propagated_positions)

    # Assuming all_itx_observed_positions and all_itx_propagated_positions are already filled with the correct data
    num_propagators = all_itx_observed_positions.shape[0]
    num_iterations = all_itx_observed_positions.shape[1]

    # Using the 'Dark2' colormap
    colormap = plt.get_cmap('Dark2')(np.linspace(0, 1, num_iterations))

    # Total number of subplots
    total_subplots = num_propagators * 3

    fig, axes = plt.subplots(num_propagators, 3, figsize=(15, 5 * num_propagators), sharex=True)

    # Initialize lists to hold the max and min y-values for each column
    max_y_values = [float('-inf')] * 3
    min_y_values = [float('inf')] * 3

    # First pass to determine the max and min y-values for each column
    for propagator_idx in range(num_propagators):
        for i, component in enumerate(['X', 'Y', 'Z']):
            ax = axes[propagator_idx, i]
            for iteration in range(num_iterations):
                observed_positions = all_itx_observed_positions[propagator_idx, iteration, :, i]
                propagated_positions = all_itx_propagated_positions[propagator_idx, iteration, :, i]
                difference = observed_positions - propagated_positions
                ax.plot(difference, label=f'Iter {iteration + 1}', color=colormap[iteration])
            
            current_min, current_max = ax.get_ylim()
            max_y_values[i] = max(max_y_values[i], current_max)
            min_y_values[i] = min(min_y_values[i], current_min)

    # Second pass to set uniform y-axis limits for each column
    for i in range(3):
        for ax in axes[:, i]:
            ax.set_ylim(min_y_values[i], max_y_values[i])
            ax.grid(True)

    # Setting titles, labels, and legend
    for i, component in enumerate(['X', 'Y', 'Z']):
        axes[0, i].set_title(f'{component} Component')
    for propagator_idx in range(num_propagators):
        axes[propagator_idx, 1].set_xlabel('Observation')
        axes[propagator_idx, 0].set_ylabel(f'meters (m)')

    # Create a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 1), title='Iterations')

    plt.show()

    # Plotting the histogram of residuals for each force model
    plt.figure(figsize=(8, 6))
    for idx, residuals in Residuals_dict.items():
        plt.hist(residuals, bins=50, label=f'Model {idx}')
    plt.title(f'Residuals - Observations: {points_to_use}')
    plt.xlabel('Residual (m)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # # Plotting the magnitude of delta_X for each iteration in its own line but all on the same plot
    # plt.figure(figsize=(8, 6))
    # for iteration in range(max_iterations):
    #     delta_X = Delta_xs_dict[iteration]
    #     plt.plot(np.linalg.norm(delta_X), label=f'Iter {iteration + 1}')
    # plt.title('Magnitude of Delta_X')
    # plt.xlabel('Iteration #')
    # plt.ylabel('Magnitude of Delta_X (m)')
    # plt.legend()

if __name__ == '__main__':
    main()