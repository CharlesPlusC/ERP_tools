import netCDF4 as nc
import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.hipparchus.geometry.euclidean.threed import Vector3D, FieldVector3D
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces import PythonForceModel
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit
from org.orekit.orbits import OrbitType
from org.orekit.orbits import PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from orekit import JArray_double
from java.util import Collections
from java.util.stream import Stream
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants
from org.orekit.utils import IERSConventions
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.utils import IERSConventions
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from orekit import JArray_double
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.frames import FramesFactory

import orekit
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()

import numpy as np

from tools.utilities import find_nearest_index, jd_to_utc, lla_to_ecef, julian_day_to_ceres_time
from tools.data_processing import extract_hourly_ceres_data
from tools.TLE_tools import twoLE_parse, tle_convert, sgp4_prop_TLE
from tools.data_processing import extract_hourly_ceres_data ,combine_lw_sw_data, calculate_satellite_fov, is_within_fov_vectorized, sat_normal_surface_angle_vectorized

def compute_erp_at_sc(ceres_time_index, radiation_data, sat_lat, sat_lon, sat_alt, horizon_dist):
    R = 6378.137  # Earth's radius in km

    # Latitude and longitude arrays
    lat = np.arange(-89.5, 90.5, 1)  # 1-degree step from -89.5 to 89.5
    lon = np.arange(-179.5, 180.5, 1)  # 1-degree step from -179.5 to 179.5

    # Mesh grid creation
    lon2d, lat2d = np.meshgrid(lon, lat)

    # FOV calculations
    fov_mask = is_within_fov_vectorized(sat_lat, sat_lon, horizon_dist, lat2d, lon2d)
    radiation_data_fov = np.ma.masked_where(~fov_mask, radiation_data[ceres_time_index, :, :])
    cos_thetas = sat_normal_surface_angle_vectorized(sat_alt, sat_lat, sat_lon, lat2d[fov_mask], lon2d[fov_mask])
    cosine_factors_2d = np.zeros_like(radiation_data_fov)
    cosine_factors_2d[fov_mask] = cos_thetas

    # Adjusting radiation data
    adjusted_radiation_data = radiation_data_fov * cosine_factors_2d

    # Satellite position and distance calculations
    sat_ecef = np.array(lla_to_ecef(sat_lat, sat_lon, sat_alt))
    ecef_x, ecef_y, ecef_z = lla_to_ecef(lat2d, lon2d, np.zeros_like(lat2d))
    ecef_pixels = np.stack((ecef_x, ecef_y, ecef_z), axis=-1)
    vector_diff = sat_ecef.reshape((1, 1, 3)) - ecef_pixels
    distances = np.linalg.norm(vector_diff, axis=2) * 1000  # Convert to meters

    # Radiation calculation
    delta_lat = np.abs(lat[1] - lat[0])
    delta_lon = np.abs(lon[1] - lon[0])
    area_pixel = R**2 * np.radians(delta_lat) * np.radians(delta_lon) * np.cos(np.radians(lat2d)) * (1000**2)  # Convert to m^2
    P_rad = adjusted_radiation_data * area_pixel / (np.pi * distances**2) # map of power flux in W/m^2 for each pixel
    # Calculating unit vectors and multiplying with P_rad
    unit_vectors = vector_diff / distances[..., np.newaxis]
    radiation_vectors = unit_vectors * P_rad[..., np.newaxis]

    # Summing all vectors
    total_radiation_vector = np.sum(radiation_vectors[fov_mask], axis=0)

    # Calculating the magnitude of the resultant vector
    total_radiation_magnitude = np.linalg.norm(total_radiation_vector)

    satellite_area = 10.0  # m^2 - total guess

    radiation_over_sat_surface = total_radiation_magnitude * satellite_area

    # force due to radiation pressure on specular surface
    force  = 2*radiation_over_sat_surface / 299792458
    print("force:", force)

    scalar_acc = force / 500.0 # 500 kg is a complete guesstimate
    print("acceleration:", scalar_acc)

    acceleration_vector = scalar_acc * total_radiation_vector / total_radiation_magnitude

    down_vector = sat_ecef / np.linalg.norm(sat_ecef)  # Normalize the satellite's position vector to get the down vector
    total_radiation_vector_normalized = total_radiation_vector / np.linalg.norm(total_radiation_vector)  # Normalize the total radiation vector

    cos_theta = np.dot(total_radiation_vector_normalized, down_vector)  # Cosine of the angle
    angle_radians = np.arccos(cos_theta)  # Angle in radians
    angle_degrees = np.rad2deg(angle_radians)  # Convert to degrees

    print("ERP angle:", angle_degrees)
    print("returning acceleration vector:", acceleration_vector)

    return np.array(acceleration_vector)

class CERES_ERP_ForceModel(PythonForceModel):
    def __init__(self):
        super().__init__()
        self.altitude = 0.0

    def acceleration(self, spacecraftState, doubleArray):
        # Compute the current altitude within the acceleration method
        pos = spacecraftState.getPVCoordinates().getPosition()
        alt_m = pos.getNorm() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        alt_km = alt_m / 1000.0
        horizon_dist = calculate_satellite_fov(alt_km)
        # Get the AbsoluteDate
        absolute_date = spacecraftState.getDate()
        # Convert AbsoluteDate to Python datetime
        date_time = absolutedate_to_datetime(absolute_date)
        # convert date_time to julianday
        jd_time = date_time.toordinal() + 1721425.5 + date_time.hour / 24 + date_time.minute / (24 * 60) + date_time.second / (24 * 60 * 60)
        # Get the ECI and ECEF frames
        ecef = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        # Transform the position vector to the ECEF frame
        pv_ecef = spacecraftState.getPVCoordinates(ecef).getPosition()
        # Define the Earth model
        earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
            Constants.WGS84_EARTH_FLATTENING, 
            ecef)
        # Convert ECEF coordinates to geodetic latitude, longitude, and altitude
        geo_point = earth.transform(pv_ecef, ecef, spacecraftState.getDate())
        # Extract latitude and longitude in radians
        latitude = geo_point.getLatitude()
        longitude = geo_point.getLongitude()
        # Convert radians to degrees if needed
        latitude_deg = np.rad2deg(latitude)
        longitude_deg = np.rad2deg(longitude)
        ceres_time = julian_day_to_ceres_time(jd_time)
        ceres_indices = find_nearest_index(ceres_times, ceres_time)

        erp_vec = compute_erp_at_sc(ceres_indices, sw_radiation_data, latitude_deg, longitude_deg, alt_km, horizon_dist)
        orekit_erp_vec = Vector3D(float(erp_vec[0]), float(erp_vec[1]), float(erp_vec[2]))
        print("orekit_erp_vec components:", orekit_erp_vec.getX(), orekit_erp_vec.getY(), orekit_erp_vec.getZ())
        return orekit_erp_vec
    
    def addContribution(self, spacecraftState, timeDerivativesEquations):
        # Add the conditional acceleration to the propagator
        timeDerivativesEquations.addNonKeplerianAcceleration(self.acceleration(spacecraftState, None))

    def getParametersDrivers(self):
        # This model does not have any adjustable parameters
        return Collections.emptyList()

    def init(self, spacecraftState, absoluteDate):
        pos = spacecraftState.getPVCoordinates().getPosition()
        self.altitude = pos.getNorm() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        pass

    def getEventDetectors(self):
        # No event detectors are used in this model
        return Stream.empty()

if __name__ == "__main__":
    dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc'  # Hourly data

    data = nc.Dataset(dataset_path)

    # Extract data from the CERES dataset
    ceres_times, lat, lon, lw_radiation_data, sw_radiation_data = extract_hourly_ceres_data(data)

    # Combine longwave and shortwave radiation data
    combined_radiation_data = combine_lw_sw_data(lw_radiation_data, sw_radiation_data)

    #oneweb TLE
    # TLE = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993\n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"
    #starlink TLE
    TLE = "1 58214U 23170J   23345.43674150  .00003150  00000+0  17305-3 0  9997\n2 58214  42.9996 329.1219 0001662 255.3130 104.7534 15.15957346  7032"
    jd_start = 2460069.5000000  # Force time to be within the CERES dataset that I downloaded
    jd_end = jd_start + 1/24 # 1 hr later
    dt = 60  # Seconds
    # sgp4_ephem = sgp4_prop_TLE(TLE=TLE, jd_start=jd_start, jd_end=jd_end, dt=dt)
    # tle_time = TLE_time(TLE) ##The TLE I have is not actually in the daterange of the CERES dataset I downloaded so not using this now
    utc_start = jd_to_utc(jd_start)

    # Segment the time stamp into year, month, day, hour, minute, second components
    YYYY = int(utc_start.strftime("%Y"))
    MM = int(utc_start.strftime("%m"))
    DD = int(utc_start.strftime("%d"))
    H = int(utc_start.strftime("%H"))
    M = int(utc_start.strftime("%M"))
    S = float(utc_start.strftime("%S"))

    #convert JD start epoch to UTC and pass to AbsoluteDate
    utc = TimeScalesFactory.getUTC() #instantiate UTC time scale
    #also add modified Julian date
    TLE_epochDate = AbsoluteDate(YYYY, MM, DD, H, M, S, utc)
    print("orekit AbsoluteDate:", TLE_epochDate)

    #Convert the initial position and velocity to keplerian elements
    tle_dict = twoLE_parse(TLE)
    kep_elems = tle_convert(tle_dict)

    a = float(kep_elems['a'])*1000
    e = float(kep_elems['e'])
    i = float(kep_elems['i'])
    omega = float(kep_elems['arg_p'])
    raan = float(kep_elems['RAAN'])
    lv = (float(kep_elems['true_anomaly']))

    ## Instantiate the inertial frame where the orbit is defined
    inertialFrame = FramesFactory.getEME2000()

    # ## Orbit construction as Keplerian
    initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lv,
                                PositionAngleType.TRUE,
                                inertialFrame, TLE_epochDate, Constants.WGS84_EARTH_MU)
    print("initial orekit orbit:", initialOrbit)

    # #Set parameters for numerical propagation
    minStep = 0.001
    maxstep = 1000.0
    initStep = 60.0
    positionTolerance = 1.0 
    tolerances = NumericalPropagator.tolerances(positionTolerance, 
                                                initialOrbit, 
                                                initialOrbit.getType())
    integrator = DormandPrince853Integrator(minStep, maxstep, 
        JArray_double.cast_(tolerances[0]),  # Double array of doubles needs to be casted in Python
        JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(initStep)
    satellite_mass = 500.0  # The models need a spacecraft mass, unit kg. 500kg is a complete guesstimate.

    #Initial state
    initialState = SpacecraftState(initialOrbit, satellite_mass) 
    propagator_num = NumericalPropagator(integrator)
    propagator_num.setOrbitType(OrbitType.CARTESIAN)
    propagator_num.setInitialState(initialState)

    print("initial SMA:", initialState.getA())

    # Add 10x10 gravity field
    gravityProvider = GravityFieldFactory.getNormalizedProvider(10, 10)
    propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    
    #### ADDITION OF ERP CUSTOM FORCE MODEL ###

    erp_force_model = CERES_ERP_ForceModel()
    propagator_num.addForceModel(erp_force_model)

    end_state = propagator_num.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(3600.0 * 1))
    end_state

    print("Initial state:")
    print(initialState)
    print("Final state:")
    print(end_state)
    print("final altitude:", end_state.getA())
    print("final orbit:", end_state.getOrbit())