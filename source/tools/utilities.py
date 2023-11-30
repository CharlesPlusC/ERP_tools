from astropy.coordinates import EarthLocation
import astropy.units as u
import numpy as np
from datetime import datetime, timedelta
from pyproj import Transformer
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential

def tle_convert(tle_dict: dict) -> dict:
    """
    Convert a TLE dictionary into the corresponding Keplerian elements.

    Parameters
    ----------
    tle_dict : dict
        Dictionary of TLE data.

    Returns
    -------
    dict
        Dictionary containing Keplerian elements.
    """
    # Standard gravitational parameter for the Earth
    GM = 398600.4415 * (1e3)**3 # m^3/s^2

    # Convert RAAN from degrees to radians
    RAAN = np.radians(float(tle_dict['right ascension of the ascending node']))
    
    # Convert argument of perigee from degrees to radians
    arg_p = np.radians(float(tle_dict['argument of perigee']))
    
    # Convert mean motion from revolutions per day to radians per second
    mean_motion = float(tle_dict['mean motion']) * (2 * np.pi / 86400)
    
    # Compute the period of the orbit in seconds
    period = 2 * np.pi / mean_motion
    
    # Compute the semi-major axis
    n = mean_motion # mean motion in radians per second
    a = (GM / (n ** 2)) ** (1/3) / 1000 # in km
    
    # Convert mean anomaly from degrees to radians
    M = np.radians(float(tle_dict['mean anomaly']))
    
    # Extract eccentricity as decimal value
    e = float("0." + tle_dict['eccentricity'])
    
    # Convert inclination from degrees to radians
    inclination = np.radians(float(tle_dict['inclination']))
    
    # Initial Guess at Eccentric Anomaly
    if M < np.pi:
        E = M + (e / 2)
    else:
        E = M - (e / 2)

    # Numerical iteration for Eccentric Anomaly
    f = lambda E: E - e * np.sin(E) - M
    fp = lambda E: 1 - e * np.cos(E)
    E = np.float64(E)
    r_tol = 1e-8 # set the convergence tolerance for the iteration
    max_iter = 50 # set the maximum number of iterations allowed
    for it in range(max_iter):
        f_value = f(E)
        fp_value = fp(E)
        E_new = E - f_value / fp_value
        if np.abs(E_new - E) < r_tol:
            E = E_new
            break
        E = E_new
    else:
        raise ValueError("Eccentric anomaly did not converge")
        
    eccentric_anomaly = E

    # Compute True Anomaly
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(eccentric_anomaly / 2),
                                  np.sqrt(1 - e) * np.cos(eccentric_anomaly / 2))

    # Dictionary of Keplerian elements
    keplerian_dict = {'a': a, 'e': e, 'i': inclination, 'RAAN': RAAN, 'arg_p': arg_p, 'true_anomaly': np.degrees(true_anomaly)}
    return keplerian_dict

def TLE_time(TLE: str) -> float:
    """
    Find the time of a TLE in Julian Day format.

    Parameters
    ----------
    TLE : str
        The TLE string.

    Returns
    -------
    float
        Time in Julian Day format.
    """
    #find the epoch section of the TLE
    epoch = TLE[18:32]
    #convert the first two digits of the epoch to the year
    year = 2000+int(epoch[0:2])
    
    # the rest of the digits are the day of the year and fractional portion of the day
    day = float(epoch[2:])
    #convert the day of the year to a day, month, year format
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)
    #convert the date to a julian date
    jd = (date - datetime.datetime(1858, 11, 17)).total_seconds() / 86400.0 + 2400000.5
    return jd

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