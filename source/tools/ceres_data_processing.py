""" Data Processing Module

This module contains functions and classes for processing satellite and CERES data.

"""
import numpy as np
import math
from geopy.distance import great_circle
from .utilities import find_nearest_index, lla_to_ecef, eci2ecef_astropy, ecef_to_lla, julian_day_to_ceres_time

def sat_normal_surface_angle_vectorized(sat_alt, sat_lat, sat_lon, pixel_lats, pixel_lons):
    """
    Compute the angle between the satellite's normal vector and the normal vectors at each pixel location on the Earth's surface.

    Parameters:
    ----------
    sat_lat : float
        Latitude of the satellite in degrees.
    sat_lon : float
        Longitude of the satellite in degrees.
    pixel_lats : array-like
        Array of latitudes for the pixels.
    pixel_lons : array-like
        Array of longitudes for the pixels.

    Returns:
    -------
    numpy.ndarray
        An array of cosine of angles between the satellite normal and each pixel surface normal.
    """
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
    Vectorized computation to check if given points are within the satellite's field of view (FoV).

    Parameters:
    ----------
    sat_lat : float
        Latitude of the satellite in degrees.
    sat_lon : float
        Longitude of the satellite in degrees.
    horizon_dist : float
        The horizon distance or the maximum distance visible from the satellite in kilometers.
    point_lats : numpy.ndarray
        Array of latitudes for the points.
    point_lons : numpy.ndarray
        Array of longitudes for the points.

    Returns:
    -------
    numpy.ndarray
        A boolean array indicating whether each point is within the satellite's FoV.
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

def calculate_satellite_fov(altitude):
    """
    Calculate the horizon distance or field of view (FoV) distance for a satellite at a given altitude.

    Parameters:
    ----------
    altitude : float
        Altitude of the satellite above the Earth's surface in kilometers.

    Returns:
    -------
    float
        The horizon distance in kilometers.
    """
    earth_radius_km = 6371  # Average radius of the Earth in kilometers
    satellite_radius_km = earth_radius_km + altitude
    horizon_distance_km = math.sqrt(satellite_radius_km**2 - earth_radius_km**2)
    return horizon_distance_km

def is_within_fov(sat_lat, sat_lon, horizon_dist, point_lat, point_lon):
    """
    Check if a given point on the Earth's surface is within the field of view (FoV) of a satellite.

    Parameters:
    ----------
    sat_lat : float
        Latitude of the satellite in degrees.
    sat_lon : float
        Longitude of the satellite in degrees.
    horizon_dist : float
        The horizon distance or the maximum distance visible from the satellite in kilometers.
    point_lat : float
        Latitude of the point in degrees.
    point_lon : float
        Longitude of the point in degrees.

    Returns:
    -------
    bool
        True if the point is within the satellite's FoV, False otherwise.
    """
    distance = great_circle((sat_lat, sat_lon), (point_lat, point_lon)).kilometers
    return distance <= horizon_dist

def extract_hourly_ceres_data(data):
    """
    Extract hourly CERES data including times, latitudes, longitudes, LW, and SW radiation data.

    Parameters:
    ----------
    data : netCDF4.Dataset
        The CERES dataset in netCDF format.

    Returns:
    -------
    tuple
        A tuple containing arrays of time, latitude, longitude, LW radiation data, and SW radiation data from the CERES dataset.
    """
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
    """
    Combine longwave (LW) and shortwave (SW) radiation data arrays.

    Parameters:
    ----------
    lw_radiation_data : numpy.ndarray
        Array of longwave radiation data.
    sw_radiation_data : numpy.ndarray
        Array of shortwave radiation data.

    Returns:
    -------
    numpy.ndarray
        The combined LW and SW radiation data.
    """
    # Check dimensions
    assert lw_radiation_data.shape == sw_radiation_data.shape, "LW and SW Data dimensions do not match"

    # Combine the data
    combined_radiation_data = lw_radiation_data + sw_radiation_data
    return combined_radiation_data

def process_trajectory(ephemeris, ceres_times):
    """
    Process satellite trajectory ephemeris data to extract latitude, longitude, altitude, and corresponding CERES time indices.

    Parameters:
    ----------
    ephemeris : list
        List of ephemeris data including position and velocity.
    ceres_times : numpy.ndarray
        Array of time points in the CERES dataset.

    Returns:
    -------
    tuple
        A tuple containing arrays of latitudes, longitudes, altitudes, and CERES time indices corresponding to the ephemeris data.
    """

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

def calculate_bearing(sat_lat, sat_lon, lat, lon):
    sat_lat_rad = np.radians(sat_lat)
    sat_lon_rad = np.radians(sat_lon)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    dlon_rad = lon_rad - sat_lon_rad

    x = np.sin(dlon_rad) * np.cos(lat_rad)
    y = np.cos(sat_lat_rad) * np.sin(lat_rad) - np.sin(sat_lat_rad) * np.cos(lat_rad) * np.cos(dlon_rad)

    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360

    return bearing

def latlon_to_fov_coordinates(lat, lon, sat_lat, sat_lon, fov_radius):
    angular_distance = []
    bearing = []

    for point_lat, point_lon in zip(lat, lon):
        # Calculate angular distance
        distance = great_circle((sat_lat, sat_lon), (point_lat, point_lon)).kilometers
        angular_distance.append(distance)

        # Calculate bearing
        brng = calculate_bearing(sat_lat, sat_lon, point_lat, point_lon)
        bearing.append(brng)

    # Convert to numpy arrays
    angular_distance = np.array(angular_distance)
    bearing = np.array(bearing)

    # Scale radial distances to match the actual FOV radius
    r = angular_distance / np.max(angular_distance) * fov_radius

    # Convert bearings to angles in radians
    theta = np.radians(bearing)

    return r, theta