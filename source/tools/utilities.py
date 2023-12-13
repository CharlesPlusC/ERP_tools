from astropy.coordinates import EarthLocation
import astropy.units as u
import numpy as np
from datetime import datetime, timedelta
from pyproj import Transformer
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential

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

def julian_day_to_ceres_time(jd, ceres_ref_date=datetime(2000, 3, 1)):
    # Convert Julian Day to datetime
    jd_reference = datetime(1858, 11, 17)
    satellite_datetime = jd_reference + timedelta(days=jd - 2400000.5)

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