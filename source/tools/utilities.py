from astropy.coordinates import EarthLocation
import astropy.units as u
import numpy as np
from datetime import datetime, timedelta
from pyproj import Transformer
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential
from typing import Tuple, List

import orekit
from orekit.pyhelpers import setup_orekit_curdir

orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()

from org.orekit.frames import FramesFactory
from org.orekit.utils import PVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import PVCoordinates, IERSConventions
from org.orekit.orbits import KeplerianOrbit, CartesianOrbit
from org.orekit.forces import ForceModel
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants

def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude (LLA).

    Parameters
    ----------
    x : List[float]
        x coordinates in km.
    y : List[float]
        y coordinates in km.
    z : List[float]
        z coordinates in km.

    Returns
    -------
    tuple
        Latitudes in degrees, longitudes in degrees, and altitudes in km.
    """
    # Convert input coordinates to meters
    x_m, y_m, z_m = x * 1000, y * 1000, z * 1000
    
    # Create a transformer for converting between ECEF and LLA
    transformer = Transformer.from_crs(
        "EPSG:4978", # WGS-84 (ECEF)
        "EPSG:4326", # WGS-84 (LLA)
        always_xy=True # Specify to always return (X, Y, Z) ordering
    )

    # Convert coordinates
    lon, lat, alt_m = transformer.transform(x_m, y_m, z_m)

    # Convert altitude to kilometers
    alt_km = alt_m / 1000

    return lat, lon, alt_km

def lla_to_ecef(lat2d, lon2d, alt):
    # Ensure alt is in meters for Astropy and broadcast it to match the shape of lat2d and lon2d
    alt_m = (alt * 1000) * np.ones_like(lat2d)  # if alt is in kilometers

    # Create EarthLocation objects
    locations = EarthLocation(lat=lat2d * u.deg, lon=lon2d * u.deg, height=alt_m * u.meter)

    # Extract ECEF coordinates in kilometers
    x = locations.x.to(u.kilometer).value
    y = locations.y.to(u.kilometer).value
    z = locations.z.to(u.kilometer).value

    return x, y, z

def eci2ecef_astropy(eci_pos, eci_vel, mjd):
    """
    Convert ECI (Earth-Centered Inertial) coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates using Astropy.

    Parameters
    ----------
    eci_pos : np.ndarray
        ECI position vectors.
    eci_vel : np.ndarray
        ECI velocity vectors.
    mjd : float
        Modified Julian Date.

    Returns
    -------
    tuple
        ECEF position vectors and ECEF velocity vectors.
    """
    # Convert MJD to isot format for Astropy
    time_utc = Time(mjd, format="mjd", scale='utc')

    # Convert ECI position and velocity to ECEF coordinates using Astropy
    eci_cartesian = CartesianRepresentation(eci_pos.T * u.km)
    eci_velocity = CartesianDifferential(eci_vel.T * u.km / u.s)
    gcrs_coords = GCRS(eci_cartesian.with_differentials(eci_velocity), obstime=time_utc)
    itrs_coords = gcrs_coords.transform_to(ITRS(obstime=time_utc))

    # Get ECEF position and velocity from Astropy coordinates
    ecef_pos = np.column_stack((itrs_coords.x.value, itrs_coords.y.value, itrs_coords.z.value))
    ecef_vel = np.column_stack((itrs_coords.v_x.value, itrs_coords.v_y.value, itrs_coords.v_z.value))

    return ecef_pos, ecef_vel

def convert_ceres_time_to_date(ceres_time, ceres_ref_date=datetime(2000, 3, 1)):
    #TODO: make sure to remove this default date
    #TODO: what the hell is the ceres_ref_date??

    # Separate days and fractional days
    days = int(ceres_time)
    fractional_day = ceres_time - days

    # Convert fractional day to hours and minutes
    hours = int(fractional_day * 24)
    minutes = int((fractional_day * 24 - hours) * 60)

    # Calculate the full date and time
    full_date = ceres_ref_date + timedelta(days=days, hours=hours, minutes=minutes)
    return full_date.strftime('%Y-%m-%d %H:%M')

# Function to convert MJD to datetime
def mjd_to_datetime(mjd):
    jd = mjd + 2400000.5
    jd_reference = datetime(1858, 11, 17)
    return jd_reference + timedelta(days=jd)

def calculate_distance_ecef(ecef1, ecef2):
    # Calculate Euclidean distance between two ECEF coordinates
    return np.sqrt((ecef1[0] - ecef2[0])**2 + (ecef1[1] - ecef2[1])**2 + (ecef1[2] - ecef2[2])**2)

from datetime import datetime, timedelta

def julian_day_to_ceres_time(jd, ceres_ref_date=datetime(2000, 3, 1)):
    #TODO: check the buisness wiht this ceres_ref_date
    # I think it's just the date that the CERES data starts on and then it's just used to calculate the delta time
    # between the two dates (and i think 2000, 3, 1 is the start of the CERES dataset)
    
    # The Julian Day for the reference date of the datetime.toordinal() function (January 1, year 1)
    JD_TO_ORDINAL_OFFSET = 1721424.5

    # Convert Julian Day to datetime
    # Adjusting by subtracting the offset to convert JD to the ordinal date
    satellite_datetime = datetime.fromordinal(int(jd - JD_TO_ORDINAL_OFFSET)) + timedelta(days=jd - int(jd))

    # Convert to fraction of day since the reference date
    delta = satellite_datetime - ceres_ref_date
    return delta.days + delta.seconds / (24 * 60 * 60)

# Function to find the nearest time index in CERES dataset
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def jd_to_utc(jd: float) -> datetime:
    """
    Convert Julian Date to UTC time tag (datetime object) using Astropy.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    datetime
        UTC time tag.
    """
    #convert jd to astropy time object
    time = Time(jd, format='jd', scale='utc')
    #convert astropy time object to datetime object
    utc = time.datetime
    return utc

def HCL_diff(eph1: np.ndarray, eph2: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate the Height, Cross-Track, and Along-Track differences at each time step between two ephemerides.

    Parameters
    ----------
    eph1 : np.ndarray
        List or array of state vectors for a satellite.
    eph2 : np.ndarray
        List or array of state vectors for another satellite.

    Returns
    -------
    tuple
        Three lists, each containing the height, cross-track, and along-track differences at each time step.
    """
    #check that the starting conditions are the same
    # if (eph1[0][0:3]) != (eph2[0][0:3]) or (eph1[0][3:6]) != (eph2[0][3:6]):
    #     warnings.warn('The two orbits do not have the same starting conditions. Make sure this is intentional.')

    H_diffs = []
    C_diffs = []
    L_diffs = []

    for i in range(0, len(eph1), 1):
        #calculate the HCL difference at each time step
        
        r1 = np.array(eph1[i][0:3])
        r2 = np.array(eph2[i][0:3])
        
        v1 = np.array(eph1[i][3:6])
        v2 = np.array(eph2[i][3:6])
        
        unit_radial = r1/np.linalg.norm(r1)
        unit_cross_track = np.cross(r1, v1)/np.linalg.norm(np.cross(r1, v1))
        unit_along_track = np.cross(unit_radial, unit_cross_track)

        #put the three unit vectors into a matrix
        unit_vectors = np.array([unit_radial, unit_cross_track, unit_along_track])

        #subtract the two position vectors
        r_diff = r1 - r2

        #relative position in HCL frame
        r_diff_HCL = np.matmul(unit_vectors, r_diff)

        #height, cross track and along track differences
        h_diff = r_diff_HCL[0]
        c_diff = r_diff_HCL[1]
        l_diff = r_diff_HCL[2]

        H_diffs.append(h_diff)
        C_diffs.append(c_diff)
        L_diffs.append(l_diff)

    return H_diffs, C_diffs, L_diffs

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

def extract_acceleration(state_vector_data, TLE_epochDate, SATELLITE_MASS, forceModel, rtn=False):
    """
    Extracts acceleration data using a specified force model and optionally computes RTN components.

    :param state_vector_data: Dictionary containing state vectors and corresponding times.
    :param TLE_epochDate: The initial epoch date for TLE.
    :param SATELLITE_MASS: The mass of the satellite.
    :param forceModel: The instance of the force model to be used.
    :param rtn: Boolean flag to compute RTN components.
    :return: List of acceleration vectors and optionally RTN components.
    """
    # Function to compute unit vector
    def unit_vector(vector):
        norm = vector.getNorm()
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector3D(vector.getX() / norm, vector.getY() / norm, vector.getZ() / norm)

    # Extract states and times for the ephemeris
    states_and_times = state_vector_data[next(iter(state_vector_data))]
    times, states = states_and_times

    # Extract accelerations and RTN components
    accelerations = []
    rtn_components = [] if rtn else None
    for i, pv in enumerate(states):
        duration = times[i]  # Duration in seconds
        current_date = TLE_epochDate.shiftedBy(duration)

        position = Vector3D(float(pv[0]), float(pv[1]), float(pv[2]))
        velocity = Vector3D(float(pv[3]), float(pv[4]), float(pv[5]))
        pvCoordinates = PVCoordinates(position, velocity)

        orbit = CartesianOrbit(pvCoordinates, FramesFactory.getEME2000(), current_date, Constants.WGS84_EARTH_MU)
        state = SpacecraftState(orbit, SATELLITE_MASS)

        parameters = ForceModel.cast_(forceModel).getParameters()
        acc = forceModel.acceleration(state, parameters)
        accelerations.append(acc)

        if rtn:
            # Compute RTN components
            radial_unit_vector = unit_vector(position)
            normal_unit_vector = unit_vector(Vector3D.crossProduct(position, velocity))
            transverse_unit_vector = unit_vector(Vector3D.crossProduct(normal_unit_vector, radial_unit_vector))

            radial_component = Vector3D.dotProduct(acc, radial_unit_vector)
            transverse_component = Vector3D.dotProduct(acc, transverse_unit_vector)
            normal_component = Vector3D.dotProduct(acc, normal_unit_vector)
            rtns = [radial_component, transverse_component, normal_component]
            rtn_components.append(rtns)

    return (accelerations, rtn_components) if rtn else accelerations

def keplerian_elements_from_orekit_ephem(ephemeris, initial_date, end_date, step, mu):
    times = []
    keplerian_elements = []

    current_date = initial_date
    while current_date.compareTo(end_date) <= 0:
        state = ephemeris.propagate(current_date)
        pvCoordinates = state.getPVCoordinates()
        frame = state.getFrame()

        keplerian_orbit = KeplerianOrbit(pvCoordinates, frame, current_date, mu)
        a = keplerian_orbit.getA()
        e = keplerian_orbit.getE()
        i = keplerian_orbit.getI()
        omega = keplerian_orbit.getPerigeeArgument()
        raan = keplerian_orbit.getRightAscensionOfAscendingNode()
        v = keplerian_orbit.getTrueAnomaly()

        keplerian_elements.append((a, e, np.rad2deg(i), np.rad2deg(omega), np.rad2deg(raan), np.rad2deg(v)))

        times.append(current_date.durationFrom(initial_date))
        current_date = current_date.shiftedBy(step)

    return times, keplerian_elements


def hcl_acc_from_sc_state(spacecraftState, acc_vec):
    """
    Calculate the HCL (Radial, Transverse, Normal) components of the given acc_vec.

    Parameters:
    spacecraftState (SpacecraftState): The state of the spacecraft.
    acc_vec (list or tuple): The acc vector components in ECEF frame.

    Returns:
    tuple: The Radial, Transverse, and Normal components of the acc vector in HCL frame.
    """
    # Get the ECI frame
    eci = FramesFactory.getEME2000()

    # Transform the ERP vector from ECEF to ECI frame
    ecef = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    transform = ecef.getTransformTo(eci, spacecraftState.getDate())
    erp_vec_ecef_pv = PVCoordinates(Vector3D(float(acc_vec[0]), float(acc_vec[1]), float(acc_vec[2])))
    erp_vec_eci_pv = transform.transformPVCoordinates(erp_vec_ecef_pv)
    erp_vec_eci = erp_vec_eci_pv.getPosition()

    # Calculate the ECI position and velocity vectors
    pv_eci = spacecraftState.getPVCoordinates(eci)
    position_eci = pv_eci.getPosition()
    velocity_eci = pv_eci.getVelocity()

    # Calculate the RTN (HCL) unit vectors
    radial_unit_vector = position_eci.normalize()
    normal_unit_vector = Vector3D.crossProduct(position_eci, velocity_eci).normalize()
    transverse_unit_vector = Vector3D.crossProduct(normal_unit_vector, radial_unit_vector)

    # Project the ERP vector onto the RTN axes
    radial_component = Vector3D.dotProduct(erp_vec_eci, radial_unit_vector)
    transverse_component = Vector3D.dotProduct(erp_vec_eci, transverse_unit_vector)
    normal_component = Vector3D.dotProduct(erp_vec_eci, normal_unit_vector)

    return radial_component, transverse_component, normal_component



def yyyy_mm_dd_hh_mm_ss_to_jd(year: int, month: int, day: int, hour: int, minute: int, second: int, milisecond: int) -> float:
    """
    Convert year, month, day, hour, minute, second to Julian Date.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    day : int
        Day.
    hour : int
        Hour.
    minute : int
        Minute.
    second : int
        Second.
    milisecond : int
        Millisecond.

    Returns
    -------
    float
        Julian Date.
    """
    dt_obj = datetime(year, month, day, hour, minute, second, milisecond*1000)
    jd = Time(dt_obj).jd
    return jd

def doy_to_dom_month(year: int, doy: int) -> Tuple[int, int]:
    """
    Convert day of year to day of month and month number.

    Parameters
    ----------
    year : int
        Year.
    doy : int
        Day of year.

    Returns
    -------
    tuple
        Day of month and month number.
    """
    d = datetime(year, 1, 1) + timedelta(doy - 1)
    day_of_month = d.day
    month = d.month
    return day_of_month, month

def jd_to_utc(jd: float) -> datetime:
    """
    Convert Julian Date to UTC time tag (datetime object) using Astropy.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    datetime
        UTC time tag.
    """
    #convert jd to astropy time object
    time = Time(jd, format='jd', scale='utc')
    #convert astropy time object to datetime object
    utc = time.datetime
    return utc