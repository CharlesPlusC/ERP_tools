""" Data Processing Module

This module contains functions and classes for processing satellite and CERES data.

"""
import numpy as np
from sgp4.api import Satrec
import math
from geopy.distance import great_circle
from .utilities import find_nearest_index, lla_to_ecef, eci2ecef_astropy, ecef_to_lla, julian_day_to_ceres_time

def sgp4_prop_TLE(TLE: str, jd_start: float, jd_end: float, dt: float, alt_series: bool = False):
    """
    Given a TLE, a start time, end time, and time step, propagate the TLE and return the time-series of Cartesian coordinates and accompanying time-stamps (MJD).
    
    This is simply a wrapper for the SGP4 routine in the sgp4.api package (Brandon Rhodes).

    Parameters
    ----------
    TLE : str
        TLE to be propagated.
    jd_start : float
        Start time of propagation in Julian Date format.
    jd_end : float
        End time of propagation in Julian Date format.
    dt : float
        Time step of propagation in seconds.
    alt_series : bool, optional
        If True, return the altitude series as well as the position series. Defaults to False.

    Returns
    -------
    list
        List of lists containing the time-series of Cartesian coordinates, and accompanying time-stamps (MJD).
    """
    if jd_start > jd_end:
        print('jd_start must be less than jd_end')
        return

    ephemeris = []
    
    #convert dt from seconds to julian day
    dt_jd = dt/86400

    #split at the new line
    split_tle = TLE.split('\n')
    s = split_tle[0]
    r = split_tle[1]

    fr = 0.0 # precise fraction (SGP4 docs for more info)
    
    #create a satellite object
    satellite = Satrec.twoline2rv(s, r)

    time = jd_start
    # for i in range (jd_start, jd_end, dt):
    while time < jd_end:
        # propagate the satellite to the next time step
        # Position is in idiosyncratic True Equator Mean Equinox coordinate frame used by SGP4
        # Velocity is the rate at which the position is changing, expressed in kilometers per second
        error, position, velocity = satellite.sgp4(time, fr)
        if error != 0:
            print('Satellite position could not be computed for the given date')
            break
        else:
            ephemeris.append([time,position, velocity]) #jd time, pos, vel
        time += dt_jd

    return ephemeris

def sat_normal_surface_angle_vectorized(sat_lat, sat_lon, pixel_lats, pixel_lons):
    """
    Vectorized calculation of the angle between satellite normal and pixel surface normals.
    """
    sat_alt = 1200  # km
    # Convert satellite position from LLA to ECEF (including altitude)
    sat_ecef = np.array(lla_to_ecef(sat_lat, sat_lon, sat_alt))

    # Convert pixel positions from LLA to ECEF
    pixel_ecef = np.array([lla_to_ecef(lat, lon, 0) for lat, lon in zip(pixel_lats, pixel_lons)])

    # Normalize the satellite vector
    satellite_normal = sat_ecef / np.linalg.norm(sat_ecef)

    earth_surface_normals = pixel_ecef / np.linalg.norm(pixel_ecef, axis=1)[:, np.newaxis] 
    
    # Calculate angles between vectors
    dot_products = np.einsum('i,ji->j', satellite_normal, earth_surface_normals)

    cos_thetas = np.clip(dot_products, 0, 1)
    
    return cos_thetas

def is_within_fov_vectorized(sat_lat, sat_lon, horizon_dist, point_lats, point_lons):
    """
    Vectorized check if points are within the satellite's field of view.
    """
    R = 6371.0  # Radius of the Earth in kilometers

    sat_lat_rad = np.radians(sat_lat)
    sat_lon_rad = np.radians(sat_lon)
    point_lats_rad = np.radians(point_lats)
    point_lons_rad = np.radians(point_lons)

    dlon = point_lons_rad - sat_lon_rad
    dlat = point_lats_rad - sat_lat_rad

    a = np.sin(dlat / 2)**2 + np.cos(sat_lat_rad) * np.cos(point_lats_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance <= horizon_dist

# Function to calculate the satellite's FoV
def calculate_satellite_fov(altitude):
    earth_radius_km = 6371  # Average radius of the Earth in kilometers
    satellite_radius_km = earth_radius_km + altitude
    horizon_distance_km = math.sqrt(satellite_radius_km**2 - earth_radius_km**2)
    return horizon_distance_km

# Function to check if a point is within the satellite's FoV
def is_within_fov(sat_lat, sat_lon, horizon_dist, point_lat, point_lon):
    distance = great_circle((sat_lat, sat_lon), (point_lat, point_lon)).kilometers
    return distance <= horizon_dist

def extract_hourly_ceres_data(data):
    # Extract the time, latitude, and longitude variables
    ceres_times = data.variables['time'][:]  # Array of time points in the CERES dataset
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]

    #check that toa_lw_all_1h and toa_sw_all_1h exist:
    assert 'toa_lw_all_1h' in data.variables, "toa_lw_all_1h not in data- make sure you are using the hourly CERES data"
    assert 'toa_sw_all_1h' in data.variables, "toa_sw_all_1h not in data- make sure you are using the hourly CERES data"

    lw_radiation_data = data.variables['toa_lw_all_1h'][:] 
    sw_radiation_data = data.variables['toa_sw_all_1h'][:] 
    return ceres_times, lat, lon, lw_radiation_data, sw_radiation_data

def combine_lw_sw_data(lw_radiation_data, sw_radiation_data):
    # Check dimensions
    assert lw_radiation_data.shape == sw_radiation_data.shape, "LW and SW Data dimensions do not match"

    # Combine the data
    combined_radiation_data = lw_radiation_data + sw_radiation_data
    return combined_radiation_data

def process_trajectory(ephemeris, ceres_times):

    ecef_pos_list = []
    ecef_vel_list = []

    for i in range(len(ephemeris)):
        ecef_pos, ecef_vel = eci2ecef_astropy(eci_pos = np.array([ephemeris[i][1]]), eci_vel = np.array([ephemeris[i][2]]), mjd = ephemeris[i][0]-2400000.5)
        ecef_pos_list.append(ecef_pos)
        ecef_vel_list.append(ecef_vel)

    lats, lons, alts = [], [], []

    for i in range(len(ecef_pos_list)):
        lat, lon, alt = ecef_to_lla(ecef_pos_list[i][0][0], ecef_pos_list[i][0][1], ecef_pos_list[i][0][2])
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)

    # Calculate satellite positions and times
    # (Assuming you have already calculated sl_lats, sl_lons, sl_alts, and ow_lats, ow_lons, ow_alts)
    ceres_time = [julian_day_to_ceres_time(jd) for jd in (ephem[0] for ephem in ephemeris)]

    # sl_ceres_indices = [find_nearest_index(ceres_times, t) for t in sl_ceres_time]
    ceres_indices = [find_nearest_index(ceres_times, t) for t in ceres_time]

    return lats, lons, alts, ceres_indices