import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

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
# from org.orekit.forces import BoxAndSolarArraySpacecraft
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient, KnockeRediffusedForceModel
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants
from org.orekit.models.earth.atmosphere.data import JB2008SpaceEnvironmentData
from org.orekit.models.earth.atmosphere import JB2008, DTM2000, NRLMSISE00
from org.orekit.data import DataSource
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.time import TimeScalesFactory   

from tools.utilities import extract_acceleration, download_file_url
# from tools.ceres_data_processing import extract_hourly_ceres_data, extract_hourly_ceres_data ,combine_lw_sw_data
# from tools.CERES_ERP import CERES_ERP_ForceModel

import numpy as np
import netCDF4 as nc
from scipy.integrate import solve_ivp

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

    if config_flags.get('36x36gravity', False):
        MU = Constants.WGS84_EARTH_MU
        newattr = NewtonianAttraction(MU)
        propagator.addForceModel(newattr)
        gravityProvider = GravityFieldFactory.getNormalizedProvider(36, 36)
        gravityAttractionModel = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, False), gravityProvider)
        propagator.addForceModel(gravityAttractionModel)

    if config_flags.get('120x120gravity', False):
        MU = Constants.WGS84_EARTH_MU
        newattr = NewtonianAttraction(MU)
        propagator.addForceModel(newattr)
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

    if config_flags.get('solid_tides', False):
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

    if config_flags.get('ocean_tides', False):
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

    if config_flags.get('knocke_erp', False):
        sun = CelestialBodyFactory.getSun()
        spacecraft = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        onedeg_in_rad = np.radians(1.0)
        angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
        knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
        propagator.addForceModel(knockeModel)

    if config_flags.get('relativity', False):
        relativity = Relativity(Constants.WGS84_EARTH_MU)
        propagator.addForceModel(relativity)

    # if config_flags.get('ceres_erp', False):
    #     ceres_erp_force_model = CERES_ERP_ForceModel(ceres_times, combined_radiation_data, mass, cross_section, cr)
    #     propagator.addForceModel(ceres_erp_force_model)

    if config_flags.get('jb08drag', False):
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

    elif config_flags.get('dtm2000drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        propagator.addForceModel(dragForce)

    elif config_flags.get('nrlmsise00drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = NRLMSISE00(msafe, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
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

def propagate_STM(state_ti, t0, dt, phi_i, cr, cd, cross_section,mass, estimate_drag=False, **force_model_config):

    df_dy_size = 7 if estimate_drag else 6
    df_dy = np.zeros((df_dy_size, df_dy_size))

    state_vector_data = state_ti[:6]  # x, y, z, xv, yv, zv
    epochDate = datetime_to_absolutedate(t0)
    accelerations_t0 = np.zeros(3)
    force_models = []

    if force_model_config.get('36x36gravity', False):
        MU = Constants.WGS84_EARTH_MU
        monopolegrav = NewtonianAttraction(MU)
        force_models.append(monopolegrav)
        monopole_gravity_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, monopolegrav)
        monopole_gravity_eci_t0 = np.array([monopole_gravity_eci_t0[0].getX(), monopole_gravity_eci_t0[0].getY(), monopole_gravity_eci_t0[0].getZ()])
        accelerations_t0+=monopole_gravity_eci_t0

        gravityProvider = GravityFieldFactory.getNormalizedProvider(36,36)
        gravityfield = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, False), gravityProvider)
        force_models.append(gravityfield)
        gravityfield_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, gravityfield)
        gravityfield_eci_t0 = np.array([gravityfield_eci_t0[0].getX(), gravityfield_eci_t0[0].getY(), gravityfield_eci_t0[0].getZ()])
        accelerations_t0+=gravityfield_eci_t0

    if force_model_config.get('120x120gravity', False):
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
        # if boxwing:
        #     radiation_sensitive = boxwing
        # else:
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

    ###NOTE: Drag force model has to stay last in the if-loop (see below)
    if force_model_config.get('jb08drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False))
        jb08_data = JB2008SpaceEnvironmentData(solfsmy_data_source,
                                            dtcfile_data_source)
        utc = TimeScalesFactory.getUTC()
        atmosphere = JB2008(jb08_data, sun, wgs84Ellipsoid, utc)
        # if boxwing:
        #     drag_sensitive = boxwing
        # else:
        drag_sensitive = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, drag_sensitive)
        force_models.append(dragForce)
        atmospheric_drag_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, dragForce)
        atmospheric_drag_eci_t0 = np.array([atmospheric_drag_eci_t0[0].getX(), atmospheric_drag_eci_t0[0].getY(), atmospheric_drag_eci_t0[0].getZ()])
        accelerations_t0+=atmospheric_drag_eci_t0

    elif force_model_config.get('dtm2000drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        force_models.append(dragForce)
        atmospheric_drag_eci_t0 = extract_acceleration(state_vector_data, epochDate, mass, dragForce)
        atmospheric_drag_eci_t0 = np.array([atmospheric_drag_eci_t0[0].getX(), atmospheric_drag_eci_t0[0].getY(), atmospheric_drag_eci_t0[0].getZ()])
        accelerations_t0+=atmospheric_drag_eci_t0

    elif force_model_config.get('nrlmsise00drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = NRLMSISE00(msafe, sun, wgs84Ellipsoid)
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
            if force_model_config.get('jb08drag', False):
                wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False))
                
                jb08_data = JB2008SpaceEnvironmentData(solfsmy_data_source, dtcfile_data_source)
                utc = TimeScalesFactory.getUTC()
                atmosphere = JB2008(jb08_data, sun, wgs84Ellipsoid, utc)
            elif force_model_config.get('dtm2000drag', False):
                wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
                msafe = MarshallSolarActivityFutureEstimation(
                    MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
                    MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
                atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)
            elif force_model_config.get('nrlmsise00drag', False):
                wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
                msafe = MarshallSolarActivityFutureEstimation(
                    MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
                    MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
                atmosphere = NRLMSISE00(msafe, sun, wgs84Ellipsoid)
            # if boxwing:
            #     drag_sensitive = boxwing
            # else:
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

    print(df_dy)
    # Propagate State Transition Matrix (STM)
    dt_seconds = float(dt.total_seconds())
    print(f"Propagating STM for {dt_seconds} seconds")

    # Proceed only if dt_seconds is positive
    if dt_seconds > 61:
        t_span = [0, dt_seconds]
        t_eval = np.arange(0, dt_seconds, 60)  # Evaluate every minute
        initial_condition = phi_i.flatten()
        print(f"Initial condition: {initial_condition}")

        # Integrate and capture the STM at each minute
        result = solve_ivp(lambda t, y: (df_dy @ y.reshape(phi_i.shape)).flatten(), t_span, initial_condition, method='Radau', t_eval=t_eval)
        print(f"Integration result: {result}")

        # If integration is successful and there are results, plot the Frobenius norm
        if result.success and result.y.size > 0:
            frobenius_norms = [np.linalg.norm(stm.reshape(phi_i.shape), 'fro') for stm in result.y.T]
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(result.t, frobenius_norms, marker='o', linestyle='-', color='b')
            plt.title('Frobenius Norm of the STM over Time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frobenius Norm')
            plt.grid(True)
            plt.show()

            # Return the final STM in the same shape as the original function
            phi_t1 = result.y[:, -1].reshape(phi_i.shape)
        else:
            print("Integration failed or produced no results.")
            return None
    else:
        print("Non-positive dt_seconds, skipping propagation.")
        return phi_i  # Return the initial condition unchanged if time step is zero or negative

    return phi_t1



def rho_i(measured_state, measurement_type='state'):
    # maps a state vector to a measurement vector
    if measurement_type == 'state':
        return measured_state
    #TODO: implement other measurement types
