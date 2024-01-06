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
from scipy.linalg import block_diag
from datetime import datetime

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

# Function to create a symmetric correlation matrix from lower triangular covariance elements
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

    Delta_xs_dict = {}
    Residuals_dict = {}
    import scipy
    max_iterations = 10
    points_to_use = 60  # Number of observations to use
    convergence_threshold = 0.1

    # Initialize dictionaries for storing results
    Delta_xs_dict = {}
    Residuals_dict = {}

    for idx, configured_propagator in enumerate(propagators[0:1]):
        propagator = configured_propagator
        apriori_state_vector = state_vector.copy()

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

            # Weight matrix (inverse of observation covariance matrix)
            Wk = np.linalg.inv(total_observation_cov_matrix)

            # Least squares calculations
            HtWH = total_design_matrix.T @ Wk @ total_design_matrix
            HtWY = total_design_matrix.T @ Wk @ total_residuals_vector

            # Update state vector
            delta_X = np.linalg.inv(HtWH) @ HtWY
            print("state before update: ", apriori_state_vector)
            apriori_state_vector += delta_X.flatten()
            print("state after update: ", apriori_state_vector)
            print("position difference remaining: ", np.linalg.norm(apriori_state_vector[:3] - observed_state[:3]))

            # Store the change in state vector for each iteration
            Delta_xs_dict[iteration] = delta_X
            print(f'Delta_X: {delta_X.flatten()}')
            # Check for convergence
            if np.linalg.norm(delta_X) < convergence_threshold:
                print("Convergence achieved.")
                break
            else:
                print("Convergence not achieved.")
                # print("residuals: ", total_residuals_vector)

        # Store the residuals for this configuration
        Residuals_dict[idx] = total_residuals_vector

    import matplotlib.pyplot as plt
    # Plotting the histogram of residuals
    plt.figure(figsize=(8, 6))
    for idx, residuals in Residuals_dict.items():
        plt.hist(residuals, bins=100, label=f'Model {idx}')
    plt.title(f'Residuals - Observations: {points_to_use}')
    plt.xlabel('Residual (m)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    #plot a histogram of the residuals
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    for idx, total_residuals_vector in Residuals_dict.items():
        plt.hist(total_residuals_vector, bins=100)
    plt.title(f'Residuals - Observations:{points_to_use}, force Model: {idx}')
    #add number of residuals as text
    plt.xlabel('Residual(m)')
    plt.ylabel('Frequency')
    plt.show()

    # #plot each configuration's Delta_xs
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8,6))
    # for idx, Delta_xs in Delta_xs_dict.items():
    #     plt.plot(Delta_xs, label=f'config{idx}')
    # plt.title(f'Convergence of Delta_x')
    # plt.xlabel('Iteration')
    # plt.ylabel('Delta_x')
    # plt.legend()
    # plt.show()

            # # plot the x,y,z residuals
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(8,6))
            # plt.scatter(epochs_of_residuals, pos_norm_residuals, s=3, marker='o', color = 'red')
            # plt.title(f'Position Residuals - Observations:{points_to_use}, config: {idx}')
            # plt.xlabel('Date')
            # plt.ylabel('Position Residual(m)')
            # plt.show()

#TODO: get covariance matrix at each iteration and for each force model
#TODO: now we are assuming perfect initial conditions and perfect measurements.
        # # Store the estimated state
        # estimated_positions_dict[(points_to_use, idx)] = state_vector[:3]  # Position
        # estimated_velocities_dict[(points_to_use, idx)] = state_vector[3:]  # Velocity


            #residuals = np.array(observedValue) - np.array(estimatedValue)
            #normal matrix = design matrix transpose * weight matrix * design matrix
            #rhs vector = design matrix transpose * weight matrix * residuals
            #state correction = normal matrix inverse * rhs vector
            #initial state = initial state + state correction
            #if np.linalg.norm(state correction) < convergence threshold:
            #   break



#             date_start = datetime_to_absolutedate(startCollectionDate).shiftedBy(-86400.0)
#             date_end = datetime_to_absolutedate(odDate).shiftedBy(86400.0)

#             estimatedPropagator = estimatedPropagatorArray[0]
#             print(f"Estimated propagator: {estimatedPropagator}")
#             print(f"Final estimated parameters for configuration {idx}:")
#             print(estimatedPropagator.getInitialState().getOrbit())
#             estpos = estimatedPropagator.getInitialState().getOrbit().getPVCoordinates().getPosition()
#             estvel = estimatedPropagator.getInitialState().getOrbit().getPVCoordinates().getVelocity()

#             #key is pts to use and force model config
#             key = (points_to_use, idx)
#             estimated_positions_dict[key] = estpos
#             estimated_velocities_dict[key] = estvel

#             # estimated_params = estimatedPropagator.getInitialState().getOrbit()
#             estimatedInitialState = estimatedPropagator.getInitialState()
#             actualOdDate = estimatedInitialState.getDate()
#             estimatedPropagator.resetInitialState(estimatedInitialState)
#             estimatedgenerator = estimatedPropagator.getEphemerisGenerator()
#             estimatedPropagator.propagate(date_start, date_end)
#             bounded_propagator = estimatedgenerator.getGeneratedEphemeris()

#             lvlh = LocalOrbitalFrame(eci, LOFType.LVLH, bounded_propagator, 'LVLH')
#             covMat_eci_java = estimator.getPhysicalCovariances(1.0e-12)
#             eci2lvlh_frozen = eci.getTransformTo(lvlh, actualOdDate).freeze()
#             jacobianDoubleArray = JArray_double2D(6, 6)
#             eci2lvlh_frozen.getJacobian(CartesianDerivativesFilter.USE_PV, jacobianDoubleArray)
#             jacobian = Array2DRowRealMatrix(jacobianDoubleArray)
#             covMat_lvlh_java = jacobian.multiply(covMat_eci_java.multiply(jacobian.transpose()))

#             covarianceMat_eci = np.matrix([covMat_eci_java.getRow(iRow)
#                                         for iRow in range(0, covMat_eci_java.getRowDimension())])
#             covarianceMat_lvlh = np.matrix([covMat_lvlh_java.getRow(iRow)
#                                             for iRow in range(0, covMat_lvlh_java.getRowDimension())])
            
#             import seaborn as sns
#             import matplotlib.pyplot as plt
#             #ECI covariance matrix
#             labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel']
#             from matplotlib.colors import SymLogNorm
#             log_norm = SymLogNorm(linthresh=1e-10, vmin=covarianceMat_eci.min(), vmax=covarianceMat_eci.max())
#             plt.figure(figsize=(8, 7))
#             sns.heatmap(covarianceMat_eci, annot=True, fmt=".3e", xticklabels=labels, yticklabels=labels, cmap="viridis", norm=log_norm)
#             #add title containing points_to_use and configuration
#             plt.title(f"No. obs:{points_to_use}, force model:{idx}")
#             plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/covMat_eci/covMat_eci_pts{points_to_use}_config{idx}.png")

#             #LVLH covariance matrix
#             labels = ['H_pos', 'C_pos', 'L_pos', 'H_vel', 'C_vel', 'L_vel']
#             from matplotlib.colors import SymLogNorm
#             log_norm = SymLogNorm(linthresh=1e-10, vmin=covarianceMat_lvlh.min(), vmax=covarianceMat_lvlh.max())
#             plt.figure(figsize=(8, 7))
#             sns.heatmap(covarianceMat_lvlh, annot=True, fmt=".3e", xticklabels=labels, yticklabels=labels, cmap="viridis", norm=log_norm)
#             #add title containing points_to_use and configuration
#             plt.title(f"No. obs:{points_to_use}, force model:{idx}")
#             plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/covMat_lvlh/covMatlvlh_pts{points_to_use}_config{idx}.png")
#             # plt.show()

#             # Create and Plot the Correlation Matrix
#             plt.figure(figsize=(8, 7))
#             lower_triangular_data = covarianceMat_lvlh[np.tril_indices_from(covarianceMat_lvlh)]
#             corr_matrix = create_symmetric_corr_matrix(lower_triangular_data)
#             sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
#             plt.title(f"Correlation Matrix - points_to_use = {points_to_use}, configuration = {idx}")
#             plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/corrMat_lvlh/corrMat_lvlh_{points_to_use}_{idx}.png")

#             propagatorParameters   = estimator.getPropagatorParametersDrivers(True)
#             measurementsParameters = estimator.getMeasurementsParametersDrivers(True)

#             lastEstimations = estimator.getLastEstimations()
#             valueSet = lastEstimations.values()
#             estimatedMeasurements = valueSet.toArray()
#             keySet = lastEstimations.keySet()
#             realMeasurements = keySet.toArray()

#             from org.orekit.estimation.measurements import EstimatedMeasurement

#             # Assuming that each measurement has 6 elements (3 for position, 3 for velocity)
#             columns = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']
#             pv_residuals = pd.DataFrame(columns=columns)

#             for estMeas, realMeas in zip(estimatedMeasurements, realMeasurements):
#                 estMeas = EstimatedMeasurement.cast_(estMeas)
#                 estimatedValue = estMeas.getEstimatedValue()
#                 pyDateTime = absolutedate_to_datetime(estMeas.date)
                
#                 if PV.instance_(realMeas):
#                     observedValue = PV.cast_(realMeas).getObservedValue()
#                     # Compute residuals and convert JArray to numpy array
#                     residuals = np.array(observedValue) - np.array(estimatedValue)
#                     # Ensure residuals are a 1D array with the correct length
#                     residuals = residuals.ravel()[:len(columns)]
#                     # Assign the residuals to the DataFrame
#                     pv_residuals.loc[pyDateTime] = residuals

#             # Check the first few rows of the DataFrame
#             # Plotting - Adjust this part based on the specific data you want to plot
#             import matplotlib.pyplot as plt
#             plt.figure(figsize=(8,6))
#             # plot x,y,z residuals
#             # mean_pos_residuals = pv_residuals[['pos_x', 'pos_y', 'pos_z']].mean(axis=1)
#             # mean_vel_residuals = pv_residuals[['vel_x', 'vel_y', 'vel_z']].mean(axis=1)
#             plt.scatter(pv_residuals.index, pv_residuals['pos_x'], s=3, marker='o', color = 'red')
#             plt.scatter(pv_residuals.index, pv_residuals['pos_y'], s=3, marker='o', color = 'green')
#             plt.scatter(pv_residuals.index, pv_residuals['pos_z'], s=3, marker='o', color = 'blue')
#             plt.legend(['x', 'y', 'z'])
#             plt.title(f'Position Residuals - Observations:{points_to_use}, config: {idx}')
#             plt.xlabel('Date')
#             plt.ylabel('Position Residual(m)')
#             if points_to_use == 60:
#                 plt.ylim(-2, 2)
#             elif points_to_use == 120:
#                 plt.ylim(-15, 15)
#             elif points_to_use == 180:
#                 plt.ylim(-20, 20)
#             plt.grid(True)
#             plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/residuals/pos_res_pts{points_to_use}_config{idx}.png")
#             # plt.show()

#             #close all figures
#             plt.close('all')

#     import matplotlib.patches as mpatches
#     # Create subplots for x, y, and z positions
#     fig, axs = plt.subplots(3, 1, figsize=(8, 6))

#     # Initialize a list to store legend handles
#     legend_handles = []

#     # Iterate over the estimated positions
#     for key, position in estimated_positions_dict.items():
#         points_to_use, force_model_idx = key
#         x, y, z = position.getX(), position.getY(), position.getZ()

#         # Plot x, y, z on their respective subplots
#         axs[0].scatter(points_to_use, x, c=f'C{force_model_idx}')
#         axs[1].scatter(points_to_use, y, c=f'C{force_model_idx}')
#         axs[2].scatter(points_to_use, z, c=f'C{force_model_idx}')

#         # Create legend handles (only if not already created for this force_model_idx)
#         if force_model_idx not in [h.get_label() for h in legend_handles]:
#             legend_handles.append(mpatches.Patch(color=f'C{force_model_idx}', label=f'FM {force_model_idx}'))

#     # Set titles and labels for each subplot
#     axs[0].set_title('Estimated X Positions')
#     axs[0].set_ylabel('X Position (m)')

#     axs[1].set_title('Estimated Y Positions')
#     axs[1].set_ylabel('Y Position (m)')

#     axs[2].set_title('Estimated Z Positions')
#     axs[2].set_xlabel('Observations (points_to_use)')
#     axs[2].set_ylabel('Z Position (m)')

#     # Apply grid to all subplots
#     for ax in axs:
#         ax.grid(True)

#     # Add a single legend to the figure
#     fig.legend(handles=legend_handles, loc='upper right')

#     # Save the figure
#     # plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
#     plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/estimated_positions.png")

#     # for idx, data in results.items():
#     #     plt.plot(data['ranges_used'], data['out_of_plane'], label=f'Config {idx}')

#     # plt.title('Mean Position Standard Deviation vs Points Used')
#     # plt.xlabel('Points Used')
#     # plt.ylabel('Mean Position Std (m)')

#     # plt.legend()
#     # plt.grid(True)
#     # plt.savefig('output/cov_heatmaps/starlink_fitting_test/mean_pos_std_vs_points_used.png')
#     # # plt.show()

if __name__ == '__main__':
    main()