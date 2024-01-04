import orekit
from orekit.pyhelpers import setup_orekit_curdir

orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()

from tools.spaceX_ephem_tools import  parse_spacex_datetime_stamps
from tools.utilities import yyyy_mm_dd_hh_mm_ss_to_jd, jd_to_utc

import pandas as pd
import numpy as np
from spacetrack import SpaceTrackClient
import getpass
from datetime import datetime
from datetime import timedelta

from orekit.pyhelpers import datetime_to_absolutedate, absolutedate_to_datetime
from org.orekit.estimation.measurements import PV, ObservableSatellite
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import Constants as orekit_constants
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.propagation.analytical.tle import TLE
from org.orekit.attitudes import NadirPointing
from org.orekit.propagation.analytical.tle import SGP4
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder
from org.orekit.frames import LocalOrbitalFrame, LOFType
from org.orekit.utils import CartesianDerivativesFilter
from orekit.pyhelpers import JArray_double2D
from org.hipparchus.linear import Array2DRowRealMatrix
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, Relativity
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.models.earth.atmosphere import DTM2000
from org.orekit.forces.drag import DragForce, IsotropicDrag

from org.orekit.propagation.conversion import NumericalPropagatorBuilder
from org.orekit.orbits import PositionAngleType


def std_dev_from_lower_triangular(lower_triangular_data):
    cov_matrix = np.zeros((6, 6))
    row, col = np.tril_indices(6)
    cov_matrix[row, col] = lower_triangular_data
    cov_matrix = cov_matrix + cov_matrix.T - np.diag(cov_matrix.diagonal())
    std_dev = np.sqrt(np.diag(cov_matrix))
    return std_dev

def configure_force_models(propagatorBuilder,cr, cd, cross_section, enable_gravity=True, enable_third_body=True,
                        enable_solar_radiation=True, enable_relativity=True, enable_atmospheric_drag=True):
    # Earth gravity field with degree 64 and order 64
    if enable_gravity:
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
        ecef = itrf
        gravityProvider = GravityFieldFactory.getNormalizedProvider(64, 64)
        gravityAttractionModel = HolmesFeatherstoneAttractionModel(ecef, gravityProvider)
        propagatorBuilder.addForceModel(gravityAttractionModel)

    # Moon and Sun perturbations
    if enable_third_body:
        moon = CelestialBodyFactory.getMoon()
        sun = CelestialBodyFactory.getSun()
        moon_3dbodyattraction = ThirdBodyAttraction(moon)
        sun_3dbodyattraction = ThirdBodyAttraction(sun)
        propagatorBuilder.addForceModel(moon_3dbodyattraction)
        propagatorBuilder.addForceModel(sun_3dbodyattraction)

    # Solar radiation pressure
    if enable_solar_radiation:
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
        ecef = itrf
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(ecef)
        cross_section = float(cross_section)
        cr = float(cr)
        isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(cross_section, cr)
        solarRadiationPressure = SolarRadiationPressure(sun, wgs84Ellipsoid, isotropicRadiationSingleCoeff)
        propagatorBuilder.addForceModel(solarRadiationPressure)

    # Relativity
    if enable_relativity:
        relativity = Relativity(orekit_constants.EIGEN5C_EARTH_MU)
        propagatorBuilder.addForceModel(relativity)

    # Atmospheric drag
    if enable_atmospheric_drag:
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
        ecef = itrf
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(ecef)
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        propagatorBuilder.addForceModel(dragForce)

    return propagatorBuilder

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
    sigma_positions = []
    sigma_velocities = []

    # Calculate averaged standard deviations for each row
    for _, row in pd.DataFrame(covariance_data).iterrows():
        std_devs = std_dev_from_lower_triangular(row.values)
        sigma_position = np.mean(std_devs[:3])  # Assuming I can average uncertainty in x, y, z
        sigma_velocity = np.mean(std_devs[3:])  # Assuming I can average uncertainty in u, v, w
        sigma_positions.append(sigma_position)
        sigma_velocities.append(sigma_velocity)

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
        'sigma_pos': sigma_positions,
        'sigma_vel': sigma_velocities,
        **covariance_data
    })

    spacex_ephem_df['hours'] = (spacex_ephem_df['JD'] - spacex_ephem_df['JD'][0]) * 24 # hours since first timestamp
    # calculate UTC time by applying jd_to_utc() to each JD value
    spacex_ephem_df['UTC'] = spacex_ephem_df['JD'].apply(jd_to_utc)
    # TODO: I am gaining 3 milisecond per minute in the UTC time. Why?

    return spacex_ephem_df

from org.orekit.estimation.leastsquares import PythonBatchLSObserver
class BLSObserver(PythonBatchLSObserver):
    def __init__(self):
        super().__init__()
        self.all_estimations = []

    def evaluationPerformed(self, itCounts, 
           evCounts, orbits, orbParams, propParams, 
           measParams, provider, lspEval):
        print(f"iteration counts: {itCounts}")
        print(f"evaluation counts: {evCounts}")
        print(f"estimated orbital parameters: {orbParams.getDrivers()}")

    def returnAllEstimations(self):
        return self.all_estimations

def main():
    ephem_path = '/Users/charlesc/Documents/GitHub/ERP_tools/external/ephems/starlink/MEME_57632_STARLINK-30309_3530645_Operational_1387262760_UNCLASSIFIED.txt'
    spacex_ephem_dfwcov = spacex_ephem_to_df_w_cov(ephem_path)

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

    odDate = datetime(2023, 12, 19, 7, 45, 42, 00000)
    collectionDuration = 1 * 1/24 * 1/60 * 120 # 120 minutes
    startCollectionDate = odDate + timedelta(days=-collectionDuration)

    #Get TLE for first guess
    # Space-Track
    identity_st = input('Enter SpaceTrack username')
    password_st = getpass.getpass(prompt='Enter SpaceTrack password for account {}'.format(identity_st))
    st = SpaceTrackClient(identity=identity_st, password=password_st)
    rawTle = st.tle(norad_cat_id=sat_list[sc_name]['norad_id'], epoch='<{}'.format(odDate), orderby='epoch desc', limit=1, format='tle')
    print("rawTle: ", rawTle)
    tleLine1 = rawTle.split('\n')[0]
    tleLine2 = rawTle.split('\n')[1]

    # Orbit propagator parameters
    prop_min_step = 0.001 # s
    prop_max_step = 25.0 # s
    prop_position_error = 0.01 # m

    # Estimator parameters
    estimator_position_scale = 1.0 # m
    estimator_convergence_thres = 0.01 # m
    estimator_max_iterations = 100
    estimator_max_evaluations = 100
    gcrf = FramesFactory.getGCRF()
    # Selecting frames to use for OD
    eci = gcrf
    orekitTle = TLE(tleLine1, tleLine2)
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
    ecef = itrf
    wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(ecef)
    nadirPointing = NadirPointing(eci, wgs84Ellipsoid)
    sgp4Propagator = SGP4(orekitTle, nadirPointing, sat_list[sc_name]['mass'])
    tleInitialState = sgp4Propagator.getInitialState()
    # tleEpoch = tleInitialState.getDate()
    tleOrbit_TEME = tleInitialState.getOrbit()
    tlePV_ECI = tleOrbit_TEME.getPVCoordinates(eci)
    tleOrbit_ECI = CartesianOrbit(tlePV_ECI, eci, wgs84Ellipsoid.getGM())
    integratorBuilder = DormandPrince853IntegratorBuilder(prop_min_step, prop_max_step, prop_position_error)
    propagatorBuilder = NumericalPropagatorBuilder(tleOrbit_ECI,
                                                integratorBuilder, PositionAngleType.MEAN, estimator_position_scale)
    propagatorBuilder.setMass(sat_list[sc_name]['mass'])
    propagatorBuilder.setAttitudeProvider(nadirPointing)
    propagatorBuilders = []
    configurations = [
        {'enable_gravity': True, 'enable_third_body': False, 'enable_solar_radiation': False, 'enable_atmospheric_drag': False},
        {'enable_gravity': True, 'enable_third_body': True, 'enable_solar_radiation': False, 'enable_atmospheric_drag': False},
        {'enable_gravity': True, 'enable_third_body': True, 'enable_solar_radiation': True, 'enable_atmospheric_drag': False},
        {'enable_gravity': True, 'enable_third_body': True, 'enable_solar_radiation': True, 'enable_atmospheric_drag': True},
    ]

    for config in configurations:
        propagatorBuilder = NumericalPropagatorBuilder(tleOrbit_ECI, integratorBuilder, PositionAngleType.MEAN, estimator_position_scale)
        propagatorBuilder.setMass(sat_list[sc_name]['mass'])
        propagatorBuilder.setAttitudeProvider(nadirPointing)
        configured_propagatorBuilder = configure_force_models(propagatorBuilder,sat_list[sc_name]['cr'], sat_list[sc_name]['cd'],
                                                              sat_list[sc_name]['cross_section'], **config)
        propagatorBuilders.append(configured_propagatorBuilder)

    estimated_positions = []
    estimated_velocities = []
    for idx, configured_propagatorBuilder in enumerate(propagatorBuilders):
        for points_to_use in range(60, 130, 60):
            # Reset the initial state
            tleOrbit_ECI = CartesianOrbit(tlePV_ECI, eci, wgs84Ellipsoid.getGM())
            propagatorBuilder = NumericalPropagatorBuilder(
                tleOrbit_ECI,
                integratorBuilder, PositionAngleType.MEAN, estimator_position_scale
            )
            propagatorBuilder.setMass(sat_list[sc_name]['mass'])
            propagatorBuilder.setAttitudeProvider(nadirPointing)
            # Reapply force model configurations
            configured_propagatorBuilder = configure_force_models(propagatorBuilder, sat_list[sc_name]['cr'], sat_list[sc_name]['cd'],
                                                                sat_list[sc_name]['cross_section'], **configurations[idx])

            # Reset and configure the estimator
            matrixDecomposer = QRDecomposer(1e-12)
            optimizer = GaussNewtonOptimizer(matrixDecomposer, False)
            estimator = BatchLSEstimator(optimizer, configured_propagatorBuilder)
            estimator.setParametersConvergenceThreshold(estimator_convergence_thres)
            estimator.setMaxIterations(estimator_max_iterations)
            estimator.setMaxEvaluations(estimator_max_evaluations)
            estimator.setObserver(BLSObserver())

            # Add measurements for the current number of points to use
            for _, row in spacex_ephem_dfwcov.head(points_to_use).iterrows():
                # existing code to add measurement
                date = datetime_to_absolutedate((row['UTC']).to_pydatetime())
                position = Vector3D(row['x']*1000, row['y']*1000, row['z']*1000)
                velocity = Vector3D(row['u']*1000, row['v']*1000, row['w']*1000)
                sigmaPosition = row['sigma_pos']
                sigmaVelocity = row['sigma_vel']
                baseWeight = 1.0
                observableSatellite = ObservableSatellite(0)
                orekitPV = PV(date, position, velocity, sigmaPosition, sigmaVelocity, baseWeight, observableSatellite)
                estimator.addMeasurement(orekitPV)

            # Perform estimation
            print(f"#Observables: {points_to_use}\nForce Model Config:{idx}")
            estimatedPropagatorArray = estimator.estimate()

            date_start = datetime_to_absolutedate(startCollectionDate).shiftedBy(-86400.0)
            date_end = datetime_to_absolutedate(odDate).shiftedBy(86400.0)

            estimatedPropagator = estimatedPropagatorArray[0]
            print(f"Estimated propagator: {estimatedPropagator}")
            print(f"Final estimated parameters for configuration {idx}:")
            print(estimatedPropagator.getInitialState().getOrbit())
            estpos = estimatedPropagator.getInitialState().getOrbit().getPVCoordinates().getPosition()
            estvel = estimatedPropagator.getInitialState().getOrbit().getPVCoordinates().getVelocity()
            estimated_positions.append(estpos)
            estimated_velocities.append(estvel)

            # estimated_params = estimatedPropagator.getInitialState().getOrbit()
            estimatedInitialState = estimatedPropagator.getInitialState()
            actualOdDate = estimatedInitialState.getDate()
            estimatedPropagator.resetInitialState(estimatedInitialState)
            estimatedgenerator = estimatedPropagator.getEphemerisGenerator()
            estimatedPropagator.propagate(date_start, date_end)
            bounded_propagator = estimatedgenerator.getGeneratedEphemeris()

            lvlh = LocalOrbitalFrame(eci, LOFType.LVLH, bounded_propagator, 'LVLH')
            covMat_eci_java = estimator.getPhysicalCovariances(1.0e-12)
            eci2lvlh_frozen = eci.getTransformTo(lvlh, actualOdDate).freeze()
            jacobianDoubleArray = JArray_double2D(6, 6)
            eci2lvlh_frozen.getJacobian(CartesianDerivativesFilter.USE_PV, jacobianDoubleArray)
            jacobian = Array2DRowRealMatrix(jacobianDoubleArray)
            covMat_lvlh_java = jacobian.multiply(covMat_eci_java.multiply(jacobian.transpose()))

            covarianceMat_eci = np.matrix([covMat_eci_java.getRow(iRow)
                                        for iRow in range(0, covMat_eci_java.getRowDimension())])
            covarianceMat_lvlh = np.matrix([covMat_lvlh_java.getRow(iRow)
                                            for iRow in range(0, covMat_lvlh_java.getRowDimension())])
            
            import seaborn as sns
            import matplotlib.pyplot as plt
            #ECI covariance matrix
            labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel']
            from matplotlib.colors import SymLogNorm
            log_norm = SymLogNorm(linthresh=1e-10, vmin=covarianceMat_eci.min(), vmax=covarianceMat_eci.max())
            plt.figure(figsize=(8, 7))
            sns.heatmap(covarianceMat_eci, annot=True, fmt=".3e", xticklabels=labels, yticklabels=labels, cmap="viridis", norm=log_norm)
            #add title containing points_to_use and configuration
            plt.title(f"No. obs:{points_to_use}, force model:{idx}")
            plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/covMat_eci/covMat_eci_pts{points_to_use}_config{idx}.png")

            #LVLH covariance matrix
            labels = ['H_pos', 'C_pos', 'L_pos', 'H_vel', 'C_vel', 'L_vel']
            from matplotlib.colors import SymLogNorm
            log_norm = SymLogNorm(linthresh=1e-10, vmin=covarianceMat_lvlh.min(), vmax=covarianceMat_lvlh.max())
            plt.figure(figsize=(8, 7))
            sns.heatmap(covarianceMat_lvlh, annot=True, fmt=".3e", xticklabels=labels, yticklabels=labels, cmap="viridis", norm=log_norm)
            #add title containing points_to_use and configuration
            plt.title(f"No. obs:{points_to_use}, force model:{idx}")
            plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/covMat_lvlh/covMatlvlh_pts{points_to_use}_config{idx}.png")
            # plt.show()

            # Create and Plot the Correlation Matrix
            plt.figure(figsize=(8, 7))
            lower_triangular_data = covarianceMat_lvlh[np.tril_indices_from(covarianceMat_lvlh)]
            corr_matrix = create_symmetric_corr_matrix(lower_triangular_data)
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
            plt.title(f"Correlation Matrix - points_to_use = {points_to_use}, configuration = {idx}")
            plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/corrMat_lvlh/corrMat_lvlh_{points_to_use}_{idx}.png")

            propagatorParameters   = estimator.getPropagatorParametersDrivers(True)
            measurementsParameters = estimator.getMeasurementsParametersDrivers(True)

            lastEstimations = estimator.getLastEstimations()
            valueSet = lastEstimations.values()
            estimatedMeasurements = valueSet.toArray()
            keySet = lastEstimations.keySet()
            realMeasurements = keySet.toArray()

            from org.orekit.estimation.measurements import EstimatedMeasurement

            # Assuming that each measurement has 6 elements (3 for position, 3 for velocity)
            columns = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']
            pv_residuals = pd.DataFrame(columns=columns)

            for estMeas, realMeas in zip(estimatedMeasurements, realMeasurements):
                estMeas = EstimatedMeasurement.cast_(estMeas)
                estimatedValue = estMeas.getEstimatedValue()
                pyDateTime = absolutedate_to_datetime(estMeas.date)
                
                if PV.instance_(realMeas):
                    observedValue = PV.cast_(realMeas).getObservedValue()
                    # Compute residuals and convert JArray to numpy array
                    residuals = np.array(observedValue) - np.array(estimatedValue)
                    # Ensure residuals are a 1D array with the correct length
                    residuals = residuals.ravel()[:len(columns)]
                    # Assign the residuals to the DataFrame
                    pv_residuals.loc[pyDateTime] = residuals

            # Check the first few rows of the DataFrame
            # Plotting - Adjust this part based on the specific data you want to plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,6))
            # plot x,y,z residuals
            # mean_pos_residuals = pv_residuals[['pos_x', 'pos_y', 'pos_z']].mean(axis=1)
            # mean_vel_residuals = pv_residuals[['vel_x', 'vel_y', 'vel_z']].mean(axis=1)
            plt.scatter(pv_residuals.index, pv_residuals['pos_x'], s=3, marker='o', color = 'red')
            plt.scatter(pv_residuals.index, pv_residuals['pos_y'], s=3, marker='o', color = 'green')
            plt.scatter(pv_residuals.index, pv_residuals['pos_z'], s=3, marker='o', color = 'blue')
            plt.legend(['x', 'y', 'z'])
            plt.title(f'Position Residuals - Observations:{points_to_use}, config: {idx}')
            plt.xlabel('Date')
            plt.ylabel('Position Residual(m)')
            if points_to_use == 60:
                plt.ylim(-2, 2)
            elif points_to_use == 120:
                plt.ylim(-15, 15)
            plt.grid(True)
            plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/residuals/pos_res_pts{points_to_use}_config{idx}.png")
            # plt.show()

            #close all figures
            plt.close('all')

    #plot estimated positions (of the form [[{-6,882,907.389700828; -866,487.9305192033; 15,796.9627339977}], [{-6,882,907.389700828; -866,487.9305192033; 15,796.9627339977}], [{-6,882,907.389700828; -866,487.9305192033; 15,796.9627339977}], [{-6,882,907.389700828; -866,487.9305192033; 15,796.9627339977}], [{-6,882,907.389700828; -866,487.9305192033; 15,796.9627339977}], [{-6,882,907.389700828; -866,487.9305192033; 15,796.9627339977}], [{-6,882,907.389700828; -866,487.9305192033; 15,796.9627339977}], [{-6,882,907.389700828; -866,487.9305192033; 15,796.9627339977}]])
    plt.figure(figsize=(8,6))
    for idx, pos in enumerate(estimated_positions):
        plt.scatter(idx, pos.getY(), s=3, marker='o', color = 'red', label='x')
        plt.scatter(idx, pos.getX(), s=3, marker='o', color = 'green', label='y')
        plt.scatter(idx, pos.getZ(), s=3, marker='o', color = 'blue', label='z')
    plt.legend(['x', 'y', 'z'])
    plt.title(f'Estimated Positions - Observations:{points_to_use}')
    plt.xlabel('Run')
    plt.ylabel('Position (m)')
    plt.grid(True)

    plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/estimated_positions.png")

    #now plot the difference between each estimated position and the first estimated position
    plt.figure(figsize=(8,6))
    for idx, pos in enumerate(estimated_positions):
        plt.scatter(idx, pos.getY()-estimated_positions[0].getY(), s=3, marker='o', color = 'red', label='x')
        plt.scatter(idx, pos.getX()-estimated_positions[0].getX(), s=3, marker='o', color = 'green', label='y')
        plt.scatter(idx, pos.getZ()-estimated_positions[0].getZ(), s=3, marker='o', color = 'blue', label='z')
    plt.legend(['x', 'y', 'z'])
    plt.title(f'Estimated Positions - Observations:{points_to_use}')
    plt.xlabel('Run')
    plt.ylabel('Position (m)')
    plt.grid(True)

    plt.savefig(f"output/cov_heatmaps/starlink_fitting_test/estimated_positions_diff.png")





    # for idx, data in results.items():
    #     plt.plot(data['ranges_used'], data['out_of_plane'], label=f'Config {idx}')

    # plt.title('Mean Position Standard Deviation vs Points Used')
    # plt.xlabel('Points Used')
    # plt.ylabel('Mean Position Std (m)')

    # plt.legend()
    # plt.grid(True)
    # plt.savefig('output/cov_heatmaps/starlink_fitting_test/mean_pos_std_vs_points_used.png')
    # # plt.show()

if __name__ == '__main__':
    main()