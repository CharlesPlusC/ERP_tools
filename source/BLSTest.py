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

def propagate_state_using_propagator(propagator, start_date, end_date, initial_state_vector, frame, cr, cross_section, **config_flags):
    # Extract cd from the state vector
    cd = initial_state_vector[-1]  # Assuming cd is the last element in the state vector

    # Configure the force models with the current cd value
    configured_propagator = configure_force_models(propagator, cr, cross_section, cd, **config_flags)

    # Propagation using the configured propagator
    x, y, z, vx, vy, vz = initial_state_vector[:-1]  # Exclude cd from the state vector for orbit initialization
    initial_orbit = CartesianOrbit(PVCoordinates(Vector3D(float(x), float(y), float(z)),
                                                Vector3D(float(vx), float(vy), float(vz))),
                                    frame,
                                    start_date,
                                    Constants.WGS84_EARTH_MU)

    initial_state = SpacecraftState(initial_orbit)
    configured_propagator.setInitialState(initial_state)
    final_state = configured_propagator.propagate(end_date)

    pv_coordinates = final_state.getPVCoordinates()
    position = [pv_coordinates.getPosition().getX(), pv_coordinates.getPosition().getY(), pv_coordinates.getPosition().getZ()]
    velocity = [pv_coordinates.getVelocity().getX(), pv_coordinates.getVelocity().getY(), pv_coordinates.getVelocity().getZ()]

    return position + velocity + [cd]  # Include cd in the returned state

def configure_force_models(propagator,cr, cross_section,cd_var, **config_flags):
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
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd_var))
        dragForce = DragForce(atmosphere, isotropicDrag)
        propagator.addForceModel(dragForce)

    # TODO: CERES ERP force model
    # if enable_ceres:
    #     ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data) # pass the time and radiation data to the force model

    return propagator

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
    SATELLITE_MASS = 800.0
    INTEGRATOR_MIN_STEP = 0.001
    INTEGRATOR_MAX_STEP = 15.0
    INTEGRATOR_INIT_STEP = 15.0
    POSITION_TOLERANCE = 1e-3

    sat_list = {    
    'STARLINK-30309': {
        'norad_id': 57632,  # For Space-Track TLE queries
        'cospar_id': '2023-122A',  # For laser ranging data queries
        'sic_id': '000',  # For writing in CPF files
        'mass': 800.0, # kg; v2 mini
        'cross_section': 10.0, # m2; TODO: get proper value
        'cd': 2.2, # TODO: compute proper value
        'cr': 1.5  # TODO: compute proper value
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
    initial_cd = 2.2  # TODO: get this from BSTAR?
    state_vector = np.array([initial_X, initial_Y, initial_Z, initial_VX, initial_VY, initial_VZ, initial_cd])
    
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
        configured_propagator = configure_force_models(propagator,sat_list[sc_name]['cr'],sat_list[sc_name]['cross_section'], float(state_vector[-1]),**config)
        propagators.append(configured_propagator)

    max_iterations = 10
    points_to_use = 15  # Number of observations to use
    convergence_threshold = 1e-6 # Convergence threshold for delta_X

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
    3
    for idx, configured_propagator in enumerate(propagators):
        propagator = configured_propagator
        apriori_state_vector = state_vector.copy()

        for iteration in range(max_iterations):
            print(f'BLS Iteration {iteration + 1}')

            # Initializing matrices for the iteration
            total_residuals_vector = np.zeros((0, 1))
            total_design_matrix = np.zeros((0, len(state_vector)))  # no parameters to estimate
            total_observation_cov_matrix = np.zeros((0, 0))

            for i, row in spacex_ephem_dfwcov.head(points_to_use).iterrows():
                print(f'Observation {i + 1}')
                measurement_epoch = datetime_to_absolutedate(row['UTC'].to_pydatetime())
                observed_state = np.array([row['x']*1000, row['y']*1000, row['z']*1000, row['u']*1000, row['v']*1000, row['w']*1000, sat_list[sc_name]['cd']])
                propagated_state = propagate_state_using_propagator(propagator, Orbit0_epoch, measurement_epoch, apriori_state_vector, frame=eci, cr=sat_list[sc_name]['cr'], cross_section=sat_list[sc_name]['cross_section'], **config)
                
                # Storing the observed and propagated positions for each timestep
                all_itx_observed_positions[idx, iteration, i, :] = observed_state[:3]
                all_itx_propagated_positions[idx, iteration, i, :] = propagated_state[:3]

                # (Observed - Computed)
                residual = observed_state - propagated_state
                residuals_vector = residual.reshape(-1, 1)
                total_residuals_vector = np.vstack([total_residuals_vector, residuals_vector])

                # Construct observation covariance matrix for this point
                sigma_vec = np.array([row['sigma_xs'], row['sigma_ys'], row['sigma_zs'], row['sigma_us'], row['sigma_vs'], row['sigma_ws'], 1e8]) * 1000 #NOTE: added 1e8 for Cd
                observation_cov_matrix = np.diag(sigma_vec ** 2)
                total_observation_cov_matrix = scipy.linalg.block_diag(total_observation_cov_matrix, observation_cov_matrix)

                design_matrix_observation = np.zeros((len(apriori_state_vector), len(apriori_state_vector)))
                perturbation = 1e-2  # 1 cm perturbation #NOTE: this being applied to pos, vel and Cd
                # Numerically determine design matrix for this observation
                for j in range(len(apriori_state_vector)): 
                    print(f'Performing perturbation {j + 1}')
                    perturbed_state_vector = apriori_state_vector.copy()
                    perturbed_state_vector[j] += perturbation

                    # Re-configure the propagator if we are adjusting the drag coefficient
                    if j == len(apriori_state_vector) - 1:
                        propagator = configure_force_models(propagator, sat_list[sc_name]['cr'], sat_list[sc_name]['cross_section'], perturbed_state_vector[j], **config)

                    # Propagate using the perturbed state vector
                    perturbed_propagated_state = propagate_state_using_propagator(propagator, Orbit0_epoch, measurement_epoch, perturbed_state_vector, eci, sat_list[sc_name]['cr'], sat_list[sc_name]['cross_section'], **config)

                    # Calculate partial derivatives for each observation component
                    for k in range(len(perturbed_propagated_state)):
                        design_matrix_observation[k, j] = (perturbed_propagated_state[k] - propagated_state[k]) / perturbation

                total_design_matrix = np.vstack([total_design_matrix, design_matrix_observation])

            Residuals_dict[idx] = total_residuals_vector # store residuals
            print(f'Magnitude of pos residuals: {np.linalg.norm(total_residuals_vector[:3])}')
            print(f'Magnitude of vel residuals: {np.linalg.norm(total_residuals_vector[3:])}')
            print(f'Cd residual: {np.linalg.norm(total_residuals_vector[-1])}')
            
            # Weight matrix (inverse of observation covariance matrix)
            Wk = np.linalg.inv(total_observation_cov_matrix)
            HtWH = total_design_matrix.T @ Wk @ total_design_matrix
            HtWY = total_design_matrix.T @ Wk @ total_residuals_vector

            # Update state vector
            delta_X = np.linalg.inv(HtWH) @ HtWY
            apriori_state_vector += delta_X.flatten()
            Delta_xs_dict[iteration] = delta_X.flatten()

            # Check for convergence
            print("norm of delta_X: ", np.linalg.norm(delta_X))
            if np.linalg.norm(delta_X) < convergence_threshold:
                print("Convergence thresh. reached")
                break

            #check max iterations
            if iteration == max_iterations - 1:
                break

        # Calculate a priori covariance matrix
        a_priori_cov_matrix = np.linalg.inv(total_design_matrix.T @ Wk @ total_design_matrix)

        # Calculate sigma_zero_squared
        num_observations = len(total_residuals_vector)
        num_parameters = len(a_priori_cov_matrix)
        sigma_zero_squared = (total_residuals_vector.T @ Wk @ total_residuals_vector) / (num_observations - num_parameters)

        # Calculate a posteriori covariance matrix (Marek's method)
        a_posteriori_cov_matrix = a_priori_cov_matrix * sigma_zero_squared

        # Plotting the a posteriori covariance matrix
        plt.figure(figsize=(8, 7))
        plt.rcParams.update({'font.size': 8})  # Set font size
        labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel', 'cd']
        sns.heatmap(a_posteriori_cov_matrix, annot=True, fmt=".2e", xticklabels=labels, yticklabels=labels, cmap="viridis")
        plt.title('A Posteriori Covariance Matrix')
        plt.tight_layout()
        plt.show()

########## Convergence Plots ##########
    # Convert lists to numpy arrays for easier handling
    all_itx_observed_positions = np.array(all_itx_observed_positions)
    all_itx_propagated_positions = np.array(all_itx_propagated_positions)

    # Assuming all_itx_observed_positions and all_itx_propagated_positions are already filled with the correct data
    num_propagators = all_itx_observed_positions.shape[0]
    num_iterations = all_itx_observed_positions.shape[1]

    # Using the 'Dark2' colormap
    colormap = plt.get_cmap('nipy_spectral')(np.linspace(0, 1, num_iterations))
    fig, axes = plt.subplots(num_propagators, 1, figsize=(15, 5 * num_propagators), sharex=True)

    # Initialize max and min values for y-axis limits
    max_norm_diff = float('-inf')
    min_norm_diff = float('inf')

    # Loop through each propagator to find the overall max and min norm differences
    for propagator_idx in range(num_propagators):
        for iteration in range(num_iterations):
            observed_positions = all_itx_observed_positions[propagator_idx, iteration, :, :]
            propagated_positions = all_itx_propagated_positions[propagator_idx, iteration, :, :]
            difference = observed_positions - propagated_positions
            norm_difference = np.linalg.norm(difference, axis=1)
            max_norm_diff = max(max_norm_diff, np.max(norm_difference))
            min_norm_diff = min(min_norm_diff, np.min(norm_difference))

    # Loop through each propagator again and plot the norm of the positional differences
    for propagator_idx in range(num_propagators):
        ax = axes[propagator_idx] if num_propagators > 1 else axes
        for iteration in range(num_iterations):
            observed_positions = all_itx_observed_positions[propagator_idx, iteration, :, :]
            propagated_positions = all_itx_propagated_positions[propagator_idx, iteration, :, :]
            difference = observed_positions - propagated_positions
            norm_difference = np.linalg.norm(difference, axis=1)
            ax.plot(range(num_timesteps), norm_difference, label=f'Iter {iteration + 1}', color=colormap[iteration])

        ax.set_ylim(min_norm_diff, max_norm_diff)
        #log the y axis
        ax.set_title(f'Force Model #{propagator_idx + 1}')
        # add the final position error to the plot as text
        ax.text(0.05, 0.9, f'Final Position Error: {norm_difference[-1]:.5e} m', transform=ax.transAxes)
        ax.set_xlabel('Observation')
        ax.set_ylabel('Norm Difference (m)')
        ax.grid(True)

    # Create a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels() if num_propagators > 1 else axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.7), title='Iterations')

    plt.tight_layout()
    plt.show()

########## Convergence Plots ##########

    # Assuming all_itx_observed_positions and all_itx_propagated_positions are defined
    num_propagators = all_itx_observed_positions.shape[0]
    num_iterations = all_itx_observed_positions.shape[1]
    num_timesteps = all_itx_observed_positions.shape[2]

    # Using the 'nipy_spectral' colormap
    colormap = plt.get_cmap('nipy_spectral')(np.linspace(0, 1, num_iterations))

    fig, axes = plt.subplots(num_propagators, 3, figsize=(15, 5 * num_propagators), sharex=True)

    # Adjust axes array for single propagator case
    if num_propagators == 1:
        axes = [axes]

    for propagator_idx in range(num_propagators):
        for i, component in enumerate(['X', 'Y', 'Z']):
            ax = axes[propagator_idx][i]
            for iteration in range(num_iterations):
                observed_positions = all_itx_observed_positions[propagator_idx, iteration, :, i]
                propagated_positions = all_itx_propagated_positions[propagator_idx, iteration, :, i]
                difference = observed_positions - propagated_positions
                ax.plot(np.arange(num_timesteps), difference, label=f'Iter {iteration + 1}', color=colormap[iteration])
            ax.grid(True)
            ax.set_ylabel(f'{component} Difference (m)')
            if propagator_idx == num_propagators - 1:
                ax.set_xlabel('Time Step')

    # Set titles for the top row
    if num_propagators > 1:
        for i, component in enumerate(['X', 'Y', 'Z']):
            axes[0][i].set_title(f'{component} Component')
    else:
        for i, component in enumerate(['X', 'Y', 'Z']):
            axes[i].set_title(f'{component} Component')

    # Create a single legend for the entire figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), title='Iterations')

    plt.tight_layout()
    plt.show()

    #same plot as above except make a new suboplot for each force model
    plt.figure(figsize=(8, 6))
    for idx, residuals in Residuals_dict.items():
        plt.subplot(2,2,idx+1)
        plt.hist(residuals, bins=50, label=f'Model {idx}')
        plt.grid(True)
        plt.title(f'Residuals - Observations: {points_to_use}')
        plt.xlabel('Residual (m)')
        plt.ylabel('Frequency')
        plt.legend()
    plt.show()

if __name__ == '__main__':
    main()