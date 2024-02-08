import orekit
from orekit.pyhelpers import setup_orekit_curdir

orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()
import textwrap

from orekit.pyhelpers import datetime_to_absolutedate
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.attitudes import NadirPointing
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions, PVCoordinates
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.orbits import CartesianOrbit
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from orekit import JArray_double
from org.orekit.orbits import OrbitType
from org.orekit.forces.gravity.potential import GravityFieldFactory, TideSystem
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, SolidTides, OceanTides, ThirdBodyAttraction, Relativity, NewtonianAttraction
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

from tools.utilities import build_boxwing, HCL_diff, extract_acceleration, keys_to_string, download_file_url,build_boxwing, calculate_cross_correlation_matrix, get_satellite_info, pos_vel_from_orekit_ephem, keplerian_elements_from_orekit_ephem
from tools.spaceX_ephem_tools import spacex_ephem_to_df_w_cov
from tools.sp3_2_ephemeris import sp3_ephem_to_df
# from tools.ceres_data_processing import extract_hourly_ceres_data, extract_hourly_ceres_data ,combine_lw_sw_data
# from tools.CERES_ERP import CERES_ERP_ForceModel
from tools.plotting import combined_residuals_plot

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
INTEGRATOR_INIT_STEP = 60.0
POSITION_TOLERANCE = 1e-2 # 1 cm

# Download SOLFSMY and DTCFILE files for JB2008 model
solfsmy_file = download_file_url("https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT", "external/jb08_inputs/SOLFSMY.TXT")
dtcfile_file = download_file_url("https://sol.spacenvironment.net/JB2008/indices/DTCFILE.TXT", "external/jb08_inputs/DTCFILE.TXT")

# Create DataSource instances
solfsmy_data_source = DataSource(solfsmy_file)
dtcfile_data_source = DataSource(dtcfile_file)

# # load CERES dataset, combine longwave and shortwave, extract the associated times in UTC format
# ceres_dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc'
# data = nc.Dataset(ceres_dataset_path)
# ceres_times, _, _, lw_radiation_data, sw_radiation_data = extract_hourly_ceres_data(data)
# combined_radiation_data = combine_lw_sw_data(lw_radiation_data, sw_radiation_data)

def configure_force_models(propagator,cr,cross_section,cd,boxwing, **config_flags):

    if config_flags.get('gravity', False):
        MU = Constants.WGS84_EARTH_MU
        newattr = NewtonianAttraction(MU)
        propagator.addForceModel(newattr)

        ### 120x120 gravity model 
        gravityProvider = GravityFieldFactory.getNormalizedProvider(120, 120)
        gravityAttractionModel = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, False), gravityProvider)
        propagator.addForceModel(gravityAttractionModel)

    if config_flags.get('3BP', False):
        moon = CelestialBodyFactory.getMoon()
        sun = CelestialBodyFactory.getSun()
        moon_3dbodyattraction = ThirdBodyAttraction(moon)
        sun_3dbodyattraction = ThirdBodyAttraction(sun)
        propagator.addForceModel(moon_3dbodyattraction)
        propagator.addForceModel(sun_3dbodyattraction)

    if force_model_config.get('solid_tides', False):
        central_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
        ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        mu = Constants.WGS84_EARTH_MU
        tidesystem = TideSystem.ZERO_TIDE
        iersConv = IERSConventions.IERS_2010
        ut1scale = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, True)
        sun = CelestialBodyFactory.getSun()
        moon = CelestialBodyFactory.getMoon()
        solid_tides_sun = SolidTides(central_frame, ae, mu, tidesystem, iersConv, ut1scale, sun)
        solid_tides_moon = SolidTides(central_frame, ae, mu, tidesystem, iersConv, ut1scale, moon)
        propagator.addForceModel(solid_tides_sun)
        propagator.addForceModel(solid_tides_moon)

    if force_model_config.get('ocean_tides', False):
        central_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
        ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        mu = Constants.WGS84_EARTH_MU
        ocean_tides = OceanTides(central_frame, ae, mu, 4, 4, IERSConventions.IERS_2010, TimeScalesFactory.getUT1(IERSConventions.IERS_2010, False))
        propagator.addForceModel(ocean_tides)

    if config_flags.get('SRP', False):
        if boxwing:
            radiation_sensitive = boxwing
        else:
            radiation_sensitive = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        earth_ellipsoid =  OneAxisEllipsoid(Constants.IERS2010_EARTH_EQUATORIAL_RADIUS, Constants.IERS2010_EARTH_FLATTENING, FramesFactory.getITRF(IERSConventions.IERS_2010, False))
        solarRadiationPressure = SolarRadiationPressure(CelestialBodyFactory.getSun(), earth_ellipsoid, radiation_sensitive)
        solarRadiationPressure.addOccultingBody(CelestialBodyFactory.getMoon(), Constants.MOON_EQUATORIAL_RADIUS)
        propagator.addForceModel(solarRadiationPressure)

    if force_model_config.get('knocke_erp', False):
        sun = CelestialBodyFactory.getSun()
        spacecraft = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        onedeg_in_rad = np.radians(1.0)
        angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
        knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
        propagator.addForceModel(knockeModel)

    if force_model_config.get('relativity', False):
        relativity = Relativity(Constants.WGS84_EARTH_MU)
        propagator.addForceModel(relativity)

    # if config_flags.get('ceres_erp', False):
    #     ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data, mass, cross_section, cr)
    #     propagator.addForceModel(ceres_erp_force_model)

    if config_flags.get('drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False))
        jb08_data = JB2008SpaceEnvironmentData(solfsmy_data_source,
                                            dtcfile_data_source)

        utc = TimeScalesFactory.getUTC()
        atmosphere = JB2008(jb08_data, sun, wgs84Ellipsoid, utc)
        if boxwing:
            drag_sensitive = boxwing
        else:
            drag_sensitive = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, drag_sensitive)
        propagator.addForceModel(dragForce)
    return propagator

def propagate_state(start_date, end_date, initial_state_vector, cr, cd, cross_section, mass,boxwing, **config_flags):

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
    eci = FramesFactory.getGCRF()
    nadirPointing = NadirPointing(eci, ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False)))
    propagator.setAttitudeProvider(nadirPointing)

    configured_propagator = configure_force_models(propagator,cr,cross_section,cd,boxwing,**config_flags)
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

    if force_model_config.get('solid_tides', False):
        central_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
        ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        mu = Constants.WGS84_EARTH_MU
        tidesystem = TideSystem.ZERO_TIDE
        iersConv = IERSConventions.IERS_2010
        ut1scale = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, False)
        sun = CelestialBodyFactory.getSun()
        moon = CelestialBodyFactory.getMoon()
        solid_tides_moon = SolidTides(central_frame, ae, mu, tidesystem, iersConv, ut1scale, moon)
        force_models.append(solid_tides_moon)
        solid_tides_sun = SolidTides(central_frame, ae, mu, tidesystem, iersConv, ut1scale, sun)
        force_models.append(solid_tides_sun)
        solid_tides_moon_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, solid_tides_moon)
        solid_tides_moon_eci_t0 = np.array([solid_tides_moon_eci_t0[0].getX(), solid_tides_moon_eci_t0[0].getY(), solid_tides_moon_eci_t0[0].getZ()])
        accelerations_t0+=solid_tides_moon_eci_t0
        solid_tides_sun_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, solid_tides_sun)
        solid_tides_sun_eci_t0 = np.array([solid_tides_sun_eci_t0[0].getX(), solid_tides_sun_eci_t0[0].getY(), solid_tides_sun_eci_t0[0].getZ()])
        accelerations_t0+=solid_tides_sun_eci_t0

    if force_model_config.get('ocean_tides', False):
        central_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
        ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        mu = Constants.WGS84_EARTH_MU
        ocean_tides = OceanTides(central_frame, ae, mu, 4, 4, IERSConventions.IERS_2010, TimeScalesFactory.getUT1(IERSConventions.IERS_2010, False))
        force_models.append(ocean_tides)
        ocean_tides_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, ocean_tides)
        ocean_tides_eci_t0 = np.array([ocean_tides_eci_t0[0].getX(), ocean_tides_eci_t0[0].getY(), ocean_tides_eci_t0[0].getZ()])
        accelerations_t0+=ocean_tides_eci_t0

    if force_model_config.get('SRP', False):
        if boxwing:
            radiation_sensitive = boxwing
        else:
            radiation_sensitive = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        earth_ellipsoid =  OneAxisEllipsoid(Constants.IERS2010_EARTH_EQUATORIAL_RADIUS, Constants.IERS2010_EARTH_FLATTENING, FramesFactory.getITRF(IERSConventions.IERS_2010, False))
        solarRadiationPressure = SolarRadiationPressure(CelestialBodyFactory.getSun(), earth_ellipsoid, radiation_sensitive)
        solarRadiationPressure.addOccultingBody(CelestialBodyFactory.getMoon(), Constants.MOON_EQUATORIAL_RADIUS)
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

    # if force_model_config.get('ceres_erp', False):
    #     ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data, mass, cross_section, cr) # pass the time and radiation data to the force model
    #     force_models.append(ceres_erp_force_model)
    #     ceres_erp_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, ceres_erp_force_model)
    #     ceres_erp_eci_t0 = np.array([ceres_erp_eci_t0[0].getX(), ceres_erp_eci_t0[0].getY(), ceres_erp_eci_t0[0].getZ()])
    #     accelerations_t0+=ceres_erp_eci_t0

    if force_model_config.get('relativity', False):
        relativity = Relativity(Constants.WGS84_EARTH_MU)
        force_models.append(relativity)
        relativity_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, relativity)
        relativity_eci_t0 = np.array([relativity_eci_t0[0].getX(), relativity_eci_t0[0].getY(), relativity_eci_t0[0].getZ()])
        accelerations_t0+=relativity_eci_t0

    ###NOTE: this force model has to stay last in the if-loop (see below)
    if force_model_config.get('drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False))
        jb08_data = JB2008SpaceEnvironmentData(solfsmy_data_source,
                                            dtcfile_data_source)
        utc = TimeScalesFactory.getUTC()
        atmosphere = JB2008(jb08_data, sun, wgs84Ellipsoid, utc)

        if boxwing:
            drag_sensitive = boxwing
        else:
            drag_sensitive = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, drag_sensitive)
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
            if boxwing:
                drag_sensitive = boxwing
            else:
                drag_sensitive = IsotropicDrag(float(cross_section), float(cd_perturbed))
            dragForce = DragForce(atmosphere, drag_sensitive)
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
    t_span = [0, dt_seconds]
    initial_condition = phi_i.flatten()
    result = solve_ivp(lambda t, y: (df_dy @ y.reshape(phi_i.shape)).flatten(), t_span, initial_condition, method='RK45')
    phi_t1 = result.y[:, -1].reshape(phi_i.shape)

    return phi_t1

def OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag, max_patience=1, boxwing=None):

    t0 = observations_df['UTC'].iloc[0]
    x_bar_0 = np.array(a_priori_estimate[1:7])  # x, y, z, u, v, w
    state_covs = a_priori_estimate[7:13]
    cd = a_priori_estimate[-4]
    cr = a_priori_estimate[-3]
    cross_section = a_priori_estimate[-2]
    mass = a_priori_estimate[-1]
    
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
                state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1[:6], cr=cr, cd=state_ti_minus1[-1], cross_section=cross_section,mass=mass,boxwing=boxwing, **force_model_config)
                phi_ti = propagate_STM(state_ti_minus1[:6], ti, dt, phi_ti_minus1, cr=cr, cd=state_ti_minus1[-1], cross_section=cross_section,mass=mass,estimate_drag=True,boxwing=boxwing, **force_model_config)
            else:
                state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass,boxwing=boxwing, **force_model_config)
                phi_ti = propagate_STM(state_ti_minus1, ti, dt, phi_ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass,boxwing=boxwing, **force_model_config)

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

def generate_config_name(config_dict, arc_number):
    config_keys = '+'.join(key for key, value in config_dict.items() if value)
    return f"arc{arc_number}_{config_keys}"

if __name__ == "__main__":
    # sat_names_to_test = ["NAVSTAR76"]
    # sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B"]
    sat_names_to_test = ["GRACE-FO-A"]
    num_arcs = 1
    arc_length = 60
    prop_length = 60*60*12  # in seconds
    estimate_drag = False
    boxwing = False
    force_model_configs = [
        # {'gravity': True},
        {'gravity': True, '3BP': True},
        {'gravity': True, '3BP': True, 'drag': True},
        {'gravity': True, '3BP': True, 'drag': True, 'SRP': True},
        {'gravity': True, '3BP': True, 'drag': True, 'SRP': True, 'solid_tides': True, 'ocean_tides': True},
        {'gravity': True, '3BP': True, 'drag': True, 'SRP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True},
        {'gravity': True, '3BP': True, 'drag': True, 'SRP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True}
    ]

    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        sat_info = get_satellite_info(sat_name)
        if boxwing:
            boxwing_model = build_boxwing(sat_name)
        else:
            boxwing_model = None
        cd = sat_info['cd']
        cr = sat_info['cr']
        cross_section = sat_info['cross_section']
        mass = sat_info['mass']
        ephemeris_df = ephemeris_df.iloc[::2, :]
        time_step = (ephemeris_df['UTC'].iloc[1] - ephemeris_df['UTC'].iloc[0]).total_seconds() / 60.0  # in minutes
        time_step_seconds = time_step * 60.0
        arc_step = int(arc_length / time_step)  # Convert arc_length to number of rows

        rms_results = {}  
        cd_estimates = {}

        for arc in range(num_arcs):
            hcl_differences = {'H': {}, 'C': {}, 'L': {}}
            start_index = arc * arc_step
            end_index = start_index + arc_step
            arc_df = ephemeris_df.iloc[start_index:end_index]

            initial_values = arc_df.iloc[0][['x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
            initial_t = arc_df.iloc[0]['UTC']
            final_prop_t = initial_t + datetime.timedelta(seconds=prop_length)
            prop_observations_df = ephemeris_df[(ephemeris_df['UTC'] >= initial_t) & (ephemeris_df['UTC'] <= final_prop_t)]
            initial_vals = np.array(initial_values.tolist() + [cd, cr, cross_section, mass], dtype=float)

            a_priori_estimate = np.concatenate(([initial_t.timestamp()], initial_vals))
            observations_df = arc_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]

            for i, force_model_config in enumerate(force_model_configs):
                if not force_model_config.get('drag', False):
                    estimate_drag = False

                optimized_states, cov_mats, residuals, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag, max_patience=1, boxwing=boxwing_model)
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
                residuals_final = residuals[min_RMS_index]
                # Assume combined_residuals_plot and other necessary functions are defined here
                if estimate_drag:
                    # C_D value is the 7th element in optimized_state
                    cd_value = optimized_state[6]
                    cd = cd_value
                    config_name = generate_config_name(force_model_config, arc + 1)
                    cd_estimates[config_name] = cd_estimates.get(config_name, []) + [cd_value]

                combined_residuals_plot(observations_df, residuals_final, a_priori_estimate, optimized_state, force_model_config, RMSs[min_RMS_index], sat_name, i, arc, estimate_drag)
                # Propagation and RMS calculation logic
                optimized_state_orbit = CartesianOrbit(PVCoordinates(Vector3D(float(optimized_state[0]), float(optimized_state[1]), float(optimized_state[2])),
                                                                    Vector3D(float(optimized_state[3]), float(optimized_state[4]), float(optimized_state[5]))),
                                                       FramesFactory.getEME2000(),
                                                       datetime_to_absolutedate(initial_t),
                                                       Constants.WGS84_EARTH_MU)
                
                tolerances = NumericalPropagator.tolerances(POSITION_TOLERANCE, optimized_state_orbit, optimized_state_orbit.getType())
                integrator = DormandPrince853Integrator(INTEGRATOR_MIN_STEP, INTEGRATOR_MAX_STEP, JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
                integrator.setInitialStepSize(INTEGRATOR_INIT_STEP)
                initialState = SpacecraftState(optimized_state_orbit, mass)
                optimized_state_propagator = NumericalPropagator(integrator)
                optimized_state_propagator.setOrbitType(OrbitType.CARTESIAN)
                optimized_state_propagator.setInitialState(initialState)

                # Add all the force models
                print(f"propagating with force model config: {force_model_config}")
                print(f"using estimated drag coeff: {cd}")
                print(f"optimized state: {optimized_state}")
                optimized_state_propagator = configure_force_models(optimized_state_propagator, cr, cross_section, cd,boxwing, **force_model_config)
                ephemGen_optimized = optimized_state_propagator.getEphemerisGenerator()
                end_state_optimized = optimized_state_propagator.propagate(datetime_to_absolutedate(initial_t), datetime_to_absolutedate(final_prop_t))
                ephemeris = ephemGen_optimized.getGeneratedEphemeris()

                times, state_vectors = pos_vel_from_orekit_ephem(ephemeris, datetime_to_absolutedate(initial_t), datetime_to_absolutedate(final_prop_t), time_step_seconds)
                state_vector_data = (times, state_vectors)

                observation_state_vectors = prop_observations_df[['x', 'y', 'z', 'xv', 'yv', 'zv']].values
                observation_times = pd.to_datetime(prop_observations_df['UTC'].values)

                # Compare the propagated state vectors with the observations
                config_name = generate_config_name(force_model_config, arc + 1)
                rms_values = []
                for state_vector, observation_state_vector in zip(state_vectors, observation_state_vectors):
                    rms = np.sqrt(np.mean(np.square(state_vector[:3] - observation_state_vector[:3])))
                    rms_values.append(rms)
                rms_results[config_name] = rms_results.get(config_name, []) + [rms_values]
                h_diffs, c_diffs, l_diffs = HCL_diff(state_vectors, observation_state_vectors)
                hcl_differences['H'][config_name] = hcl_differences['H'].get(config_name, []) + [h_diffs]
                hcl_differences['C'][config_name] = hcl_differences['C'].get(config_name, []) + [c_diffs]
                hcl_differences['L'][config_name] = hcl_differences['L'].get(config_name, []) + [l_diffs]

            # Output directory for each arc
            output_dir = f"output/OD_BLS/Tapley/prop_estim_states/{sat_name}/arc{arc + 1}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for diff_type in ['H', 'C', 'L']:
                fig, ax = plt.subplots(figsize=(8, 4))  # Adjusted figure size for better layout
                vertical_offset = 0  # Starting offset for text annotations

                for config_name, diffs in hcl_differences[diff_type].items():
                    flat_diffs = np.array(diffs).flatten()
                    line, = ax.plot(observation_times, flat_diffs, label=config_name)  # Keep a reference to the line object

                    # Annotate the last point in the top left corner
                    ax.text(0.02, 0.98 - vertical_offset, f'{config_name}: {flat_diffs[-1]:.2f}', 
                            transform=ax.transAxes, color=line.get_color(), verticalalignment='top', fontsize=8)
                    vertical_offset += 0.07  # Increment offset for the next line

                ax.set_title(f'{diff_type} Differences for {sat_name}')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'{diff_type} Difference')

                # Place legend below the graph
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3, fontsize='small')

                plt.tight_layout()
                plt.savefig(f"{output_dir}/{diff_type}_diff_{arc_length}obs_{prop_length}prop.png")
                print(f"saved {diff_type} differences plot for {sat_name} to {output_dir}")
                plt.close()

    # Separate RMS plot for each arc
    for arc in range(num_arcs):
        output_dir = f"output/OD_BLS/Tapley/prop_estim_states/{sat_name}/arc{arc + 1}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(10, 6))
        sns.set_palette(sns.color_palette("bright", len(rms_results)))  # Change to a brighter color palette

        for i, (config_name, rms_values_list) in enumerate(rms_results.items()):
            if len(rms_values_list) > arc:
                rms_values = rms_values_list[arc]
                plt.scatter(observation_times, rms_values, label=config_name, s=3, alpha=0.7)

        plt.xlabel('Time')
        plt.ylabel('RMS (m)')
        plt.title(f'RMS Differences for {sat_name} - Arc {arc + 1}')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/RMS_diff_{arc_length}obs_{prop_length}prop_arc{arc + 1}.png", bbox_inches='tight')
        plt.close()