import orekit
from orekit.pyhelpers import setup_orekit_curdir

orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()
import textwrap

from orekit.pyhelpers import datetime_to_absolutedate
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions, PVCoordinates
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.orbits import CartesianOrbit
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from orekit import JArray_double
from org.orekit.orbits import OrbitType
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, Relativity, NewtonianAttraction
from org.orekit.forces import BoxAndSolarArraySpacecraft
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient, KnockeRediffusedForceModel
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants
from org.orekit.models.earth.atmosphere.data import JB2008SpaceEnvironmentData
from org.orekit.models.earth.atmosphere import JB2008
from org.orekit.data import DataSource
from org.orekit.time import TimeScalesFactory   

from tools.utilities import extract_acceleration, keys_to_string, download_file_url,get_boxwing_config, calculate_cross_correlation_matrix, get_satellite_info
from tools.spaceX_ephem_tools import spacex_ephem_to_df_w_cov
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.ceres_data_processing import extract_hourly_ceres_data, extract_hourly_ceres_data ,combine_lw_sw_data
from tools.CERES_ERP import CERES_ERP_ForceModel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import netCDF4 as nc
from scipy.integrate import solve_ivp
from matplotlib.colors import SymLogNorm
from matplotlib.gridspec import GridSpec

INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 15.0
POSITION_TOLERANCE = 1e-3 # 1 mm

# Download SOLFSMY and DTCFILE files for JB2008 model
solfsmy_file = download_file_url("https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT", "external/jb08_inputs/SOLFSMY.TXT")
dtcfile_file = download_file_url("https://sol.spacenvironment.net/JB2008/indices/DTCFILE.TXT", "external/jb08_inputs/DTCFILE.TXT")

# Create DataSource instances
solfsmy_data_source = DataSource(solfsmy_file)
dtcfile_data_source = DataSource(dtcfile_file)

#load CERES dataset, combine longwave and shortwave, extract the associated times in UTC format
ceres_dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc'
data = nc.Dataset(ceres_dataset_path)
ceres_times, _, _, lw_radiation_data, sw_radiation_data = extract_hourly_ceres_data(data)
combined_radiation_data = combine_lw_sw_data(lw_radiation_data, sw_radiation_data)

def configure_force_models(propagator,cr,cross_section,cd, **config_flags):

    if config_flags.get('gravity', False):
        MU = Constants.WGS84_EARTH_MU
        newattr = NewtonianAttraction(MU)
        propagator.addForceModel(newattr)

        ### 120x120 gravity model 
        gravityProvider = GravityFieldFactory.getNormalizedProvider(120, 120)
        gravityAttractionModel = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, False), gravityProvider)
        propagator.addForceModel(gravityAttractionModel)

    # Moon and Sun perturbations
    if config_flags.get('3BP', False):
        moon = CelestialBodyFactory.getMoon()
        sun = CelestialBodyFactory.getSun()
        moon_3dbodyattraction = ThirdBodyAttraction(moon)
        sun_3dbodyattraction = ThirdBodyAttraction(sun)
        propagator.addForceModel(moon_3dbodyattraction)
        propagator.addForceModel(sun_3dbodyattraction)

    # Solar radiation pressure
    if config_flags.get('SRP', False):
        isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        solarRadiationPressure = SolarRadiationPressure(CelestialBodyFactory.getSun(), earth, isotropicRadiationSingleCoeff)
        propagator.addForceModel(solarRadiationPressure)

    if config_flags.get("boxwing_srp", False):

        x_length = float(5)
        y_length = float(2)
        z_length = float(2)
        solar_array_area = float(10)
        solar_array_axis = Vector3D(float(0), float(1), float(0))  # Y-axis unit vector in spacecraft body frame
        drag_coeff = float(2.2)
        lift_ratio = float(0.0)
        absorption_coeff = float(0.7)
        reflection_coeff = float(0.2)
        sun = CelestialBodyFactory.getSun()
        rotation_rate = float(0.0)
        box_and_solar_array = BoxAndSolarArraySpacecraft(x_length, 
                                                        y_length, 
                                                        z_length, 
                                                        sun, 
                                                        solar_array_area, 
                                                        solar_array_axis,
                                                        drag_coeff, 
                                                        absorption_coeff,
                                                        rotation_rate, 
                                                        reflection_coeff)


        solar_radiation_pressure = SolarRadiationPressure(sun, wgs84Ellipsoid, box_and_solar_array)

    if force_model_config.get('knocke_erp', False):
        sun = CelestialBodyFactory.getSun()
        spacecraft = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        onedeg_in_rad = np.radians(1.0)
        angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
        knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
        propagator.addForceModel(knockeModel)

    # Atmospheric drag
    if config_flags.get('drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False))

        jb08_data = JB2008SpaceEnvironmentData(solfsmy_data_source,
                                            dtcfile_data_source)

        utc = TimeScalesFactory.getUTC()
        atmosphere = JB2008(jb08_data, sun, wgs84Ellipsoid, utc)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        propagator.addForceModel(dragForce)

    if config_flags.get('ceres_erp', False):
        ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data, mass, cross_section) # pass the time and radiation data to the force model
        propagator.addForceModel(ceres_erp_force_model)

    return propagator

def propagate_state(start_date, end_date, initial_state_vector, cr, cd, cross_section, mass, **config_flags):

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
    initialState = SpacecraftState(initial_orbit, float(mass))
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(initialState)

    configured_propagator = configure_force_models(propagator,cr,cross_section, cd,**config_flags)
    final_state = configured_propagator.propagate(datetime_to_absolutedate(end_date))

    pv_coordinates = final_state.getPVCoordinates()
    position = [pv_coordinates.getPosition().getX(), pv_coordinates.getPosition().getY(), pv_coordinates.getPosition().getZ()]
    velocity = [pv_coordinates.getVelocity().getX(), pv_coordinates.getVelocity().getY(), pv_coordinates.getVelocity().getZ()]

    return position + velocity

def rho_i(measured_state, measurement_type='state'):
    # maps a state vector to a measurement vector
    if measurement_type == 'state':
        return measured_state
    #TODO: implement other measurement types

def propagate_STM(state_ti, t0, dt, phi_i, cr, cd, cross_section,mass, estimate_drag=False, **force_model_config):

    df_dy_size = 7 if estimate_drag else 6
    df_dy = np.zeros((df_dy_size, df_dy_size))

    state_vector_data = state_ti[:6]  # x, y, z, xv, yv, zv
    epochDate = datetime_to_absolutedate(t0)
    accelerations_t0 = np.zeros(3)
    force_models = []

    if force_model_config.get('gravity', False):
        MU = Constants.WGS84_EARTH_MU
        monopolegrav = NewtonianAttraction(MU)
        force_models.append(monopolegrav)
        monopole_gravity_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, monopolegrav)
        monopole_gravity_eci_t0 = np.array([monopole_gravity_eci_t0[0].getX(), monopole_gravity_eci_t0[0].getY(), monopole_gravity_eci_t0[0].getZ()])
        accelerations_t0+=monopole_gravity_eci_t0

        gravityProvider = GravityFieldFactory.getNormalizedProvider(120,120)
        gravityfield = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, False), gravityProvider)
        force_models.append(gravityfield)
        gravityfield_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, gravityfield)
        gravityfield_eci_t0 = np.array([gravityfield_eci_t0[0].getX(), gravityfield_eci_t0[0].getY(), gravityfield_eci_t0[0].getZ()])
        accelerations_t0+=gravityfield_eci_t0

    if force_model_config.get('3BP', False):
        moon = CelestialBodyFactory.getMoon()
        sun = CelestialBodyFactory.getSun()
        moon_3dbodyattraction = ThirdBodyAttraction(moon)
        force_models.append(moon_3dbodyattraction)
        sun_3dbodyattraction = ThirdBodyAttraction(sun)
        force_models.append(sun_3dbodyattraction)

        moon_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, moon_3dbodyattraction)
        moon_eci_t0 = np.array([moon_eci_t0[0].getX(), moon_eci_t0[0].getY(), moon_eci_t0[0].getZ()])
        accelerations_t0+=moon_eci_t0

        sun_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, sun_3dbodyattraction)
        sun_eci_t0 = np.array([sun_eci_t0[0].getX(), sun_eci_t0[0].getY(), sun_eci_t0[0].getZ()])
        accelerations_t0+=sun_eci_t0

    if force_model_config.get('SRP', False):
        isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        solarRadiationPressure = SolarRadiationPressure(CelestialBodyFactory.getSun(), earth, isotropicRadiationSingleCoeff)
        force_models.append(solarRadiationPressure)
        solar_radiation_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, solarRadiationPressure)
        solar_radiation_eci_t0 = np.array([solar_radiation_eci_t0[0].getX(), solar_radiation_eci_t0[0].getY(), solar_radiation_eci_t0[0].getZ()])
        accelerations_t0+=solar_radiation_eci_t0

    if force_model_config.get('knocke_erp', False):
        sun = CelestialBodyFactory.getSun()
        spacecraft = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        onedeg_in_rad = np.radians(1.0)
        angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
        knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
        force_models.append(knockeModel)
        knocke_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, knockeModel)
        knocke_eci_t0 = np.array([knocke_eci_t0[0].getX(), knocke_eci_t0[0].getY(), knocke_eci_t0[0].getZ()])
        accelerations_t0+=knocke_eci_t0

    if force_model_config.get('ceres_erp', False):
        ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data, mass, cross_section) # pass the time and radiation data to the force model
        force_models.append(ceres_erp_force_model)
        ceres_erp_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, ceres_erp_force_model)
        ceres_erp_eci_t0 = np.array([ceres_erp_eci_t0[0].getX(), ceres_erp_eci_t0[0].getY(), ceres_erp_eci_t0[0].getZ()])
        accelerations_t0+=ceres_erp_eci_t0

    ###NOTE: this force model has to stay last in the if-loop (see below)
    if force_model_config.get('drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False))
        jb08_data = JB2008SpaceEnvironmentData(solfsmy_data_source,
                                            dtcfile_data_source)
        utc = TimeScalesFactory.getUTC()
        atmosphere = JB2008(jb08_data, sun, wgs84Ellipsoid, utc)

        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        force_models.append(dragForce)
        atmospheric_drag_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, dragForce)
        atmospheric_drag_eci_t0 = np.array([atmospheric_drag_eci_t0[0].getX(), atmospheric_drag_eci_t0[0].getY(), atmospheric_drag_eci_t0[0].getZ()])
        accelerations_t0+=atmospheric_drag_eci_t0

    state_perturbation = 0.1
    cd_perturbation = 1e-4
    variables_to_perturb = df_dy_size

    for i in range(variables_to_perturb):
        perturbed_accelerations = np.zeros(3)
        state_ti_perturbed = state_ti.copy()
        
        if i < 6:
            state_ti_perturbed[i] += state_perturbation
        elif i == 6:
            # Perturb drag coefficient and re-instantiate drag model and atmosphere
            cd_perturbed = cd + cd_perturbation
            # Re-instantiate required objects for drag force model
            wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False))
            
            jb08_data = JB2008SpaceEnvironmentData(solfsmy_data_source, dtcfile_data_source)
            utc = TimeScalesFactory.getUTC()
            atmosphere = JB2008(jb08_data, sun, wgs84Ellipsoid, utc)
            isotropicDrag = IsotropicDrag(float(cross_section), float(cd_perturbed))
            dragForce = DragForce(atmosphere, isotropicDrag)
            force_models[-1] = dragForce  # Update the drag force model

        for force_model in force_models:
            acc_perturbed = extract_acceleration(state_ti_perturbed, epochDate, mass, force_model)
            if isinstance(acc_perturbed, np.ndarray): #deal with stupid output of extract_acceleration
                acc_perturbed_values = acc_perturbed
            else:
                acc_perturbed_values = np.array([acc_perturbed[0].getX(), acc_perturbed[0].getY(), acc_perturbed[0].getZ()])
            perturbed_accelerations += acc_perturbed_values

        current_perturbation = cd_perturbation if i == 6 else state_perturbation
        partial_derivatives = (perturbed_accelerations - accelerations_t0) / current_perturbation

        # Assign partial derivatives
        if i < 6:  # State variables
            df_dy[3:6, i] = partial_derivatives
            if i >= 3:
                df_dy[i - 3, i] = 1  # Identity matrix for velocity
        elif i == 6:  # Drag coefficient
            df_dy[3:6, 6] = partial_derivatives  # Drag coefficient partials

    # Propagate State Transition Matrix (STM)
    dt_seconds = float(dt.total_seconds())
    # Define time span and initial condition
    t_span = [0, dt_seconds]  # Start and end times
    initial_condition = phi_i.flatten()  # Flatten if phi_i
    # Solve the differential equation
    result = solve_ivp(lambda t, y: (df_dy @ y.reshape(phi_i.shape)).flatten(), t_span, initial_condition, method='RK45')
    # Extract the final state and reshape back to original shape
    phi_t1 = result.y[:, -1].reshape(phi_i.shape)

    return phi_t1

def OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag, max_patience=1, box_wing_config=None):

    print(f"boxwing config in ODBLS: {box_wing_config}")

    t0 = observations_df['UTC'].iloc[0]
    x_bar_0 = np.array(a_priori_estimate[1:7])  # x, y, z, u, v, w
    state_covs = a_priori_estimate[7:13]
    cd = a_priori_estimate[-4]
    cr = a_priori_estimate[-3]
    cross_section = a_priori_estimate[-2]
    mass = a_priori_estimate[-1]
    print(f"Initial state: {(x_bar_0)}")
    print(f"Initial state covariances: {(state_covs)}")
    print(f"Initial Cd: {cd}")
    print(f" Cr: {cr}")
    print(f" cross section: {cross_section}")
    print(f" mass: {mass}")
    if estimate_drag:
        print(f"Estimating drag coefficient. Initial value: {cd}")
        x_bar_0 = np.append(x_bar_0, cd)

    state_covs = np.diag(state_covs)
    phi_ti_minus1 = np.identity(len(x_bar_0))  # n*n identity matrix for n state variables
    P_0 = np.array(state_covs, dtype=float)  # Covariance matrix from a priori estimate

    if estimate_drag:
        P_0 = np.pad(P_0, ((0, 1), (0, 1)), 'constant', constant_values=0)
        # Assign a non-zero variance to the drag coefficient to avoid non-invertible matrix
        initial_cd_variance = 1  # Setting an arbitrary value for now (but still high)
        P_0[-1, -1] = initial_cd_variance

    d_rho_d_state = np.eye(len(x_bar_0))
    H_matrix = np.empty((0, len(x_bar_0)))
    converged = False
    iteration = 1
    max_iterations = 5
    weighted_rms_last = np.inf 
    convergence_threshold = 0.001 
    no_times_diff_increased = 0

    all_residuals = []
    all_rms = []
    all_xbar_0s = []
    all_covs = []

    while not converged and iteration < max_iterations:
        print(f"Iteration: {iteration}")
        N = np.zeros(len(x_bar_0))
        lamda = np.linalg.inv(P_0)
        y_all = np.empty((0, len(x_bar_0)))
        ti_minus1 = t0
        state_ti_minus1 = x_bar_0 
        RMSs = []
        for _, row in observations_df.iterrows():
            ti = row['UTC']
            observed_state = row[['x', 'y', 'z', 'xv', 'yv', 'zv']].values
            if estimate_drag:
                observed_state = np.append(observed_state, x_bar_0[-1]) #add drag coefficient to the observed state
                obs_covariance = np.diag(row[['sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']])
                cd_covariance = 1  # TODO: is this even allowed? just setting a high value for now
                                     # TODO: do i want to reduce the covariance of Cd after each iteration?
                obs_covariance = np.pad(obs_covariance, ((0, 1), (0, 1)), 'constant', constant_values=0)
                obs_covariance[-1, -1] = cd_covariance
            else:
                obs_covariance = np.diag(row[['sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']])
            obs_covariance = np.array(obs_covariance, dtype=float)
            W_i = np.linalg.inv(obs_covariance)
            
            # Propagate state and STM
            dt = ti - ti_minus1
            if estimate_drag:
                state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1[:6], cr=cr, cd=state_ti_minus1[-1], cross_section=cross_section,mass=mass, **force_model_config)
                phi_ti = propagate_STM(state_ti_minus1[:6], ti, dt, phi_ti_minus1, cr=cr, cd=state_ti_minus1[-1], cross_section=cross_section,mass=mass,estimate_drag=True, **force_model_config)
            else:
                state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass, **force_model_config)
                phi_ti = propagate_STM(state_ti_minus1, ti, dt, phi_ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass, **force_model_config)

            # Compute H matrix and residuals for this observation
            H_matrix_row = d_rho_d_state @ phi_ti
            H_matrix = np.vstack([H_matrix, H_matrix_row])
            if estimate_drag:
                state_ti = np.append(state_ti, state_ti_minus1[-1])
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

        RMSs = np.array(RMSs)
        weighted_rms = np.sqrt(np.sum(RMSs) / (len(x_bar_0) * len (y_all)))
        print(f"RMS: {weighted_rms}")
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
                if no_times_diff_increased >= max_patience:
                    print("Stopping iteration.")
                    break
            else:
                no_times_diff_increased = 0
            weighted_rms_last = weighted_rms
            x_bar_0 += xhat  # Update nominal trajectory
            print("updated x_bar_0: ", x_bar_0)

        all_residuals.append(y_all)
        all_rms.append(weighted_rms)
        all_xbar_0s.append(x_bar_0)
        all_covs.append(np.linalg.inv(lamda))
        iteration += 1

    return all_xbar_0s, all_covs, all_residuals, all_rms

def format_array(arr, precision=3):
    """Format numpy array elements to strings with specified precision."""
    return np.array2string(arr, precision=precision, separator=', ', suppress_small=True)

if __name__ == "__main__":
    sat_names_to_test = ["TerraSAR-X","TanDEM-X", "GRACE-FO-A", "GRACE-FO-B"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        # sat_name = "STARLINK-47633"
        # ephemeris_df = spacex_ephem_to_df_w_cov('external/ephems/starlink/MEME_47633_STARLINK-2009_0291157_Operational_1359287880_UNCLASSIFIED.txt')
        sat_info = get_satellite_info(sat_name)
        cd = sat_info['cd']
        cr = sat_info['cr']
        cross_section = sat_info['cross_section']
        mass = sat_info['mass']
        ephemeris_df = ephemeris_df.iloc[::5, :]#return only every 5th row
        initial_X = ephemeris_df['x'].iloc[0]
        initial_Y = ephemeris_df['y'].iloc[0]
        initial_Z = ephemeris_df['z'].iloc[0]
        initial_VX = ephemeris_df['xv'].iloc[0]
        initial_VY = ephemeris_df['yv'].iloc[0]
        initial_VZ = ephemeris_df['zv'].iloc[0]
        initial_sigma_X = ephemeris_df['sigma_x'].iloc[0]
        initial_sigma_Y = ephemeris_df['sigma_y'].iloc[0]
        initial_sigma_Z = ephemeris_df['sigma_z'].iloc[0]
        initial_sigma_XV = ephemeris_df['sigma_xv'].iloc[0]
        initial_sigma_YV = ephemeris_df['sigma_yv'].iloc[0]
        initial_sigma_ZV = ephemeris_df['sigma_zv'].iloc[0]
        initial_t = ephemeris_df['UTC'].iloc[0]
        a_priori_estimate = np.array([initial_t, initial_X, initial_Y, initial_Z, initial_VX, initial_VY, initial_VZ,
                                    initial_sigma_X, initial_sigma_Y, initial_sigma_Z, initial_sigma_XV, initial_sigma_YV, initial_sigma_ZV,
                                    cd, cr , cross_section, mass])
        a_priori_estimate = np.array([float(i) for i in a_priori_estimate[1:]]) #cast to float for compatibility with Orekit functions
        a_priori_estimate = np.array([initial_t, *a_priori_estimate])
        
        observations_df_full = ephemeris_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
        obs_lengths_to_test = [35, 70, 105, 140]
        estimate_drag = False
        force_model_configs = [
            # {'gravity': True},
            # {'gravity': True, '3BP': True},
            # {'gravity': True, '3BP': True, 'drag': True},
            {'gravity': True, '3BP': True, 'drag': True, 'boxwing_srp': True},
            {'gravity': True, '3BP': True, 'drag': True, 'SRP': True, 'ceres_erp': True},
            {'gravity': True, '3BP': True, 'drag': True, 'SRP': True, 'knocke_erp': True}]

        covariance_matrices = []
        optimized_states = []
        for i, force_model_config in enumerate(force_model_configs):
            if not force_model_config.get('drag', False):
                estimate_drag = False
                print(f"Force model doesn't have drag. Setting estimate_drag to {estimate_drag}.")
            if force_model_config.get('boxwing_srp', False) or force_model_config.get('boxwing_drag', False):
                boxwing_info = get_boxwing_config(sat_name)
                print(f"boxwing force model requested. Using boxwing info\n: {boxwing_info}")
            else:
                boxwing_info = None

            for obs_length in obs_lengths_to_test:
                observations_df = observations_df_full.iloc[:obs_length]
                optimized_states, cov_mats, residuals, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag, max_patience=1, box_wing_config=boxwing_info)
                #save each run as a set of .npy files in its own folder with the datetimestamp and the force model config, number of observations, whether drag was estimated in the title
                #save the optimized states, covariance matrices, residuals, RMSs

                # Save data and find the index of the minimum RMS
                date_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                folder_path = "output/OD_BLS/Tapley/saved_runs"
                output_folder = f"{folder_path}/{date_now}_fmodel_{i}_#pts_{len(observations_df)}_estdrag_{estimate_drag}"
                os.makedirs(output_folder)
                np.save(f"{output_folder}/optimized_states.npy", optimized_states)
                np.save(f"{output_folder}/cov_mats.npy", cov_mats)
                np.save(f"{output_folder}/residuals.npy", residuals)
                np.save(f"{output_folder}/RMSs.npy", RMSs)

                min_RMS_index = np.argmin(RMSs)
                optimized_state = optimized_states[min_RMS_index]
                covariance_matrix = cov_mats[min_RMS_index]
                final_RMS = RMSs[min_RMS_index]
                residuals_final = residuals[min_RMS_index]
                # Creating a large figure for combined plots
                fig = plt.figure(figsize=(12, 9))
                sns.set(style="whitegrid")
                gs = GridSpec(4, 4, figure=fig)

                # Scatter plots for position and velocity residuals
                ax1 = fig.add_subplot(gs[0, :])
                ax2 = fig.add_subplot(gs[1, :])

                # Position residuals plot
                sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,0], ax=ax1, color="xkcd:blue", s=10, label='x')
                sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,1], ax=ax1, color="xkcd:green", s=10, label='y')
                sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,2], ax=ax1, color="xkcd:red", s=10, label='z')
                ax1.set_ylabel("Position Residual (m)")
                ax1.set_xlabel("Observation time (UTC)")
                ax1.legend()

                # Velocity residuals plot
                sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,3], ax=ax2, color="xkcd:purple", s=10, label='u')
                sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,4], ax=ax2, color="xkcd:orange", s=10, label='v')
                sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,5], ax=ax2, color="xkcd:yellow", s=10, label='w')
                ax2.set_ylabel("Velocity Residual (m/s)")
                ax2.set_xlabel("Observation time (UTC)")
                ax2.legend()

                # Histograms for position and velocity residuals
                ax3 = fig.add_subplot(gs[2, :2])
                ax4 = fig.add_subplot(gs[2, 2:])

                sns.histplot(residuals_final[:,0:3], bins=20, ax=ax3, palette=["xkcd:blue", "xkcd:green", "xkcd:red"], legend=False)
                ax3.set_xlabel("Position Residual (m)")
                ax3.set_ylabel("Frequency")
                ax3.legend(['x', 'y', 'z'])

                sns.histplot(residuals_final[:,3:6], bins=20, ax=ax4, palette=["xkcd:purple", "xkcd:orange", "xkcd:yellow"], legend=False)
                ax4.set_xlabel("Velocity Residual (m/s)")
                ax4.set_ylabel("Frequency")
                ax4.legend(['u', 'v', 'w'])

                # Format initial and optimized states with 3 significant figures
                # apriori2sf = np.round(a_priori_estimate[1:-3], 3)
                # optimized_state_2sf = np.round(optimized_state, 3)
                formatted_initial_state = format_array(a_priori_estimate)
                formatted_optimized_state = format_array(optimized_state)
                # Table for force model configuration, initial state, and final estimated state
                ax5 = fig.add_subplot(gs[3, :])
                force_model_data = [['Force Model Config', str(force_model_config)],
                                    ['Initial State', formatted_initial_state],
                                    ['Final Estimated State', formatted_optimized_state]]
                table = plt.table(cellText=force_model_data, colWidths=[0.5 for _ in force_model_data[0]], loc='center', cellLoc='left')
                ax5.axis('off')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.0, 1.7)

                #Overall title with spacecraft name
                plt.suptitle(f"Residuals (O-C) for best BLS iteration. \nRMS: {final_RMS:.3f} \n{sat_name}", y=0.95, fontsize=16)

                # Adjust layout
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3)

                # Save the combined plot
                #make a folder for the spacecraft if it doesn't exist
                sat_name_folder = f"output/OD_BLS/Tapley/combined_plots/{sat_name}"
                if not os.path.exists(sat_name_folder):
                    os.makedirs(sat_name_folder)
                #now make a folder for the number of observations if it doesn't exist
                obs_length_folder = f"{sat_name_folder}/{len(observations_df)}"
                if not os.path.exists(obs_length_folder):
                    os.makedirs(obs_length_folder)
                plt.savefig(f"{obs_length_folder}/fmodel_{i}_estdrag_{estimate_drag}.png")
                plt.close()

    #         date_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #         folder_path = "output/OD_BLS/Tapley/saved_runs"
    #         output_folder = f"{folder_path}/{date_now}_fmodel_{i}_#pts_{len(observations_df)}_estdrag_{estimate_drag}"
    #         os.makedirs(output_folder)
    #         np.save(f"{output_folder}/optimized_states.npy", optimized_states)
    #         np.save(f"{output_folder}/cov_mats.npy", cov_mats)
    #         np.save(f"{output_folder}/residuals.npy", residuals)
    #         np.save(f"{output_folder}/RMSs.npy", RMSs)

    #         #find the index of the minimum RMS
    #         min_RMS_index = np.argmin(RMSs)
    #         optimized_state = optimized_states[min_RMS_index]
    #         covariance_matrix = cov_mats[min_RMS_index]
    #         final_RMS = RMSs[min_RMS_index]
    #         residuals_final = residuals[min_RMS_index]
    #         covariance_matrices.append(covariance_matrix)

    #         # Create two subplots: one for position, one for velocity
    #         fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    #         # Position residuals plot
    #         axs[0].scatter(observations_df['UTC'], residuals_final[:,0], s=3, label='x', c="xkcd:blue")
    #         axs[0].scatter(observations_df['UTC'], residuals_final[:,1], s=3, label='y', c="xkcd:green")
    #         axs[0].scatter(observations_df['UTC'], residuals_final[:,2], s=3, label='z', c="xkcd:red")
    #         axs[0].set_ylabel("Position Residual (m)")
    #         # if estimate_drag:
    #         #     axs[0].set_ylim(-3, 3)
    #         # else:
    #         #     axs[0].set_ylim(-10, 10)
    #         axs[0].legend(['x', 'y', 'z'])
    #         axs[0].grid(True)

    #         # Velocity residuals plot
    #         axs[1].scatter(observations_df['UTC'], residuals_final[:,3], s=3, label='xv', c="xkcd:purple")
    #         axs[1].scatter(observations_df['UTC'], residuals_final[:,4], s=3, label='yv', c="xkcd:orange")
    #         axs[1].scatter(observations_df['UTC'], residuals_final[:,5], s=3, label='zv', c="xkcd:yellow")
    #         axs[1].set_xlabel("Observation time (UTC)")
    #         axs[1].set_ylabel("Velocity Residual (m/s)")
    #         # if estimate_drag:
    #         #     axs[1].set_ylim(-3e-3, 3e-3)
    #         # else:
    #         #     axs[1].set_ylim(-10e-10, 10e-10)
    #         axs[1].legend(['xv', 'yv', 'zv'])
    #         axs[1].grid(True)

    #         # Shared title, rotation of x-ticks, and force model text
    #         plt.suptitle(f"Residuals (O-C) for final BLS iteration. \nRMS: {final_RMS:.3f}")
    #         #
    #         plt.xticks(rotation=45)
    #         force_model_keys_str = keys_to_string(force_model_config)
    #         wrapped_text = textwrap.fill(force_model_keys_str, 20)
    #         axs[1].text(0.8, -0.2, f"Force model:\n{wrapped_text}", 
    #                     transform=axs[1].transAxes,
    #                     fontsize=10, 
    #                     verticalalignment='bottom',
    #                     bbox=dict(facecolor='white', alpha=0.3))

    #         # Save the figure
    #         save_to = f"output/OD_BLS/Tapley/estimation_experiment/new_fmodel_{i}_#pts_{len(observations_df)}.png"
    #         if estimate_drag:
    #             save_to = f"output/OD_BLS/Tapley/estimation_experiment/estim_drag_fmodel_{i}_#pts_{len(observations_df)}_.png"
    #         plt.tight_layout()
    #         plt.savefig(save_to)

    #         #plot histograms of residuals
    #         plt.figure()
    #         plt.hist(residuals_final[:,0], bins=20, label='x', color="xkcd:blue")
    #         plt.hist(residuals_final[:,1], bins=20, label='y', color="xkcd:green")
    #         plt.hist(residuals_final[:,2], bins=20, label='z', color="xkcd:red")
    #         plt.hist(residuals_final[:,3], bins=20, label='xv', color="xkcd:purple")
    #         plt.hist(residuals_final[:,4], bins=20, label='yv', color="xkcd:orange")
    #         plt.hist(residuals_final[:,5], bins=20, label='zv', color="xkcd:yellow")
    #         plt.title("Residuals (O-C) for final BLS iteration")
    #         plt.xlabel("Residual (m)")
    #         plt.ylabel("Frequency")
    #         force_model_keys_str = keys_to_string(force_model_config)
    #         wrapped_text = textwrap.fill(force_model_keys_str, 20)
    #         plt.text(0.8, 0.2, f"Force model:\n{wrapped_text}", 
    #                 transform=plt.gca().transAxes,
    #                 fontsize=10, 
    #                 verticalalignment='bottom',
    #                 bbox=dict(facecolor='white', alpha=0.5))
    #         plt.legend(['x', 'y', 'z', 'xv', 'yv', 'zv'])
    #         plt.savefig(f"output/OD_BLS/Tapley/estimation_experiment/hist_force_model_{i}_#pts_{len(observations_df)}.png")

    #         #ECI covariance matrix
    #         if estimate_drag:
    #             labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel', 'Cd']
    #         else:
    #             labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel']
    #         log_norm = SymLogNorm(linthresh=1e-10, vmin=covariance_matrix.min(), vmax=covariance_matrix.max())
    #         plt.figure(figsize=(8, 6))
    #         sns.heatmap(covariance_matrix, annot=True, fmt=".3e", xticklabels=labels, yticklabels=labels, cmap="viridis", norm=log_norm)
    #         plt.title(f"No. obs:{len(observations_df)}, force model:{i}")
    #         save_to = f"output/OD_BLS/Tapley/estimation_experiment/covMat_#pts_{len(observations_df)}_config{i}.png"
    #         if estimate_drag:
    #             save_to = f"output/OD_BLS/Tapley/estimation_experiment/covMat_#pts_{len(observations_df)}_config{i}_estim_drag.png"
    #         plt.savefig(save_to)
    
    # relative_differences = []
    # for j in range(1, len(covariance_matrices)):
    #     difference_matrix = covariance_matrices[j] - covariance_matrices[j-1]
    #     if estimate_drag:
    #         labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel', 'Cd']
    #     else:
    #         labels = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel']
    #     # Plotting the difference as a heatmap
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(difference_matrix, annot=True, fmt=".3e",xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
    #     plt.title(f'Difference in Covariance Matrix: Run {j} vs Run {j-1}')
    #     plt.savefig(f'output/OD_BLS/Tapley/estimation_experiment/diff_covmat_run_{j}_vs_{j-1}.png')

    # correlation_matrices =  calculate_cross_correlation_matrix(covariance_matrices)
    # for j in range(len(correlation_matrices)):
    #     # Plotting the correlation matrix as a heatmap
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(correlation_matrices[j], annot=True, fmt=".3f", cmap="coolwarm", center=0)
    #     plt.title(f'Correlation Matrix: Run {j}')
    #     plt.xlabel('Components')
    #     plt.ylabel('Components')
    #     plt.savefig(f'output/OD_BLS/Tapley/estimation_experiment/corr_covmat_run_{j}.png')

    # # Plot a bar chart of the final Cd values
    # if estimate_drag:
    #     plt.figure()
    #     bars = plt.bar(np.arange(len(optimized_states)), [i[-1] for i in optimized_states])

    #     # Adding labels on each bar
    #     for bar in bars:
    #         yval = bar.get_height()
    #         plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', va='bottom', ha='center')

    #     plt.xticks(np.arange(len(optimized_states)), [f"Run {i}" for i in range(len(optimized_states))])
    #     plt.title("Cd values estimated by BLS under different force models")
    #     plt.savefig(f"output/OD_BLS/Tapley/estimation_experiment/Cd_values_#fmodels_{len(optimized_states)}_#pts_{len(observations_df)}.png")

    #     #plot a linegraph of the Cd values at each iteration
    #     plt.figure()
    #     for optimized_state in optimized_states:
    #         plt.plot(np.arange(len(optimized_state)), optimized_state[-1], label=f"Run {i}")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Cd")
    #     plt.title("Cd values at each iteration")
    #     plt.legend()
    #     plt.savefig(f"output/OD_BLS/Tapley/estimation_experiment/Cd_values_iter_#fmodels_{len(optimized_states)}_#pts_{len(observations_df)}.png")

#TODO: make it easier to switch atmospheric density models
#TODO: consider using a dictionary to pass the state vector to avoid all this awkward slicing of arrays that is more error prone
#TODO: pass satellite mass as a part of the state vector