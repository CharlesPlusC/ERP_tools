""" Plotting Module

This module contains functions for plotting data, including FoV radiance in different projections.

"""
import numpy as np
from scipy.interpolate import griddata
import os
import imageio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from itertools import islice
from .data_processing import latlon_to_fov_coordinates, calculate_satellite_fov, is_within_fov, is_within_fov_vectorized, sat_normal_surface_angle_vectorized
from .utilities import convert_ceres_time_to_date, lla_to_ecef

def plot_fov_radiation_mesh(variable_name, time_index, radiation_data, lat, lon, sat_lat, sat_lon, horizon_dist, output_path, ceres_times):
    # Create a mask for the FoV
    fov_mask = np.zeros((len(lat), len(lon)), dtype=bool)

    # Update the mask based on the FoV
    for i in range(len(lat)):
        for j in range(len(lon)):
            if is_within_fov(sat_lat, sat_lon, horizon_dist, lat[i], lon[j]):
                fov_mask[i, j] = True

    # Apply the mask to the radiation data
    radiation_data_fov = np.ma.masked_where(~fov_mask, radiation_data[time_index, :, :])

    # Create the meshgrid for plotting
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    vmin, vmax = (-450, 450) if "net" in variable_name else (0, 450)
    cmap = 'seismic' if "net" in variable_name else 'magma'
    mesh_plot = ax.pcolormesh(lon2d, lat2d, radiation_data_fov, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(mesh_plot, orientation='vertical', shrink=0.7)
    cbar.set_label('Radiation (W/m^2)')

    # Extract timestamp for title
    timestamp = convert_ceres_time_to_date(ceres_times[time_index])
    plt.title(f'{variable_name} - {timestamp}')

    ax.gridlines(draw_labels=True)
    plt.savefig(output_path)
    plt.close(fig)

def compute_radiance_at_sc(variable_name, time_index, radiation_data, lat, lon, sat_lat, sat_lon, sat_alt, horizon_dist, ceres_times):
    R = 6371  # Earth's radius in km

    # Mesh grid creation
    lon2d, lat2d = np.meshgrid(lon, lat)

    #FOV calculations
    fov_mask = is_within_fov_vectorized(sat_lat, sat_lon, horizon_dist, lat2d, lon2d)
    radiation_data_fov = np.ma.masked_where(~fov_mask, radiation_data[time_index, :, :])
    cos_thetas = sat_normal_surface_angle_vectorized(sat_lat, sat_lon, lat2d[fov_mask], lon2d[fov_mask])
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
    P_rad = adjusted_radiation_data * area_pixel / (np.pi * distances**2)

    # Returning the necessary data for plotting
    return lon2d, lat2d, cosine_factors_2d, distances, P_rad, ceres_times, time_index, variable_name

def plot_data_map(lon2d, lat2d, data, label, output_path, ceres_times, time_index, variable_name):
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cmap = 'nipy_spectral'
    mesh_plot = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(mesh_plot, orientation='vertical', shrink=0.7)
    cbar.set_label(label)
    timestamp = convert_ceres_time_to_date(ceres_times[time_index])
    plt.title(f'{variable_name} - {timestamp}')
    ax.gridlines(draw_labels=True)
    modified_output_path = output_path[:-4] + ".png"
    plt.savefig(modified_output_path)
    plt.close(fig)

def plot_radiance_maps(time_index, variable_name, radiation_data, lat, lon, sat_lat, sat_lon, sat_alt, horizon_dist, output_path, ceres_times, plot_cosine_factors=True, plot_distances=True, plot_radiation=True):
    # Call compute_radiance_at_sc to get the necessary data
    lon2d, lat2d, cosine_factors_2d, distances, P_rad, _, _, _ = compute_radiance_at_sc(variable_name, time_index, radiation_data, lat, lon, sat_lat, sat_lon, sat_alt, horizon_dist, ceres_times)
    print("variable_name:", variable_name)
    # Plotting the cosine factors
    if plot_cosine_factors:
        cosine_factors_output_path = output_path[:-4] + "_cosine_map.png"
        plot_data_map(lon2d, lat2d, cosine_factors_2d, 'Cosine Factors', cosine_factors_output_path, ceres_times, time_index, variable_name)

    # Plotting the distances
    if plot_distances:
        distances_output_path = output_path[:-4] + "_distance_map.png"
        plot_data_map(lon2d, lat2d, distances, 'Distance to Satellite (km)', distances_output_path, ceres_times, time_index, variable_name)

    # Plotting the radiation data
    if plot_radiation:
        total_radiation_flux = np.sum(P_rad)
        radiation_output_path = output_path[:-4] + "_radiation_map.png"
        plot_data_map(lon2d, lat2d, P_rad, f'Radiation Reaching Satellite from each pixel (W/m^2) \nrad flux:{total_radiation_flux:.2f} W/m^2', radiation_output_path, ceres_times, time_index, variable_name)

def plot_radiance_animation(alts, lats, lons, ceres_indices, lw_radiation_data, sw_radiation_data, combined_radiation_data, lat, lon, ceres_times, number_of_tsteps, lw=True, sw=True, lwsw=True, output_folder="output/FOV_sliced_data"):
    os.makedirs(output_folder, exist_ok=True)

    lw_filenames, sw_filenames, combined_filenames = [], [], []

    for idx, (alt, sat_lat, sat_lon, ceres_idx) in islice(enumerate(zip(alts, lats, lons, ceres_indices)), number_of_tsteps):
        print("idx:", idx)
        horizon_distance_km = calculate_satellite_fov(alt)

        if lw:
            lw_plot_filename = os.path.join(output_folder, f'fov_lw_rad_{idx}.png')
            plot_radiance_maps(ceres_idx, "CERES_lw_flux", lw_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_distance_km, lw_plot_filename, ceres_times)
            lw_filenames.append(lw_plot_filename)

        if sw:
            sw_plot_filename = os.path.join(output_folder, f'fov_sw_rad_{idx}.png')
            plot_radiance_maps(ceres_idx, "CERES_sw_flux", sw_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_distance_km, sw_plot_filename, ceres_times)
            sw_filenames.append(sw_plot_filename)

        if lwsw:
            combined_plot_filename = os.path.join(output_folder, f'fov_lwsw_rad_{idx}.png')
            plot_radiance_maps(ceres_idx, "CERES_combined_flux", combined_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_distance_km, combined_plot_filename, ceres_times)
            combined_filenames.append(combined_plot_filename)

    if lw:
        lw_animation_path = os.path.join(output_folder, 'lw_flux_animation.gif')
        create_animation(lw_filenames, lw_animation_path)

    if sw:
        sw_animation_path = os.path.join(output_folder, 'sw_flux_animation.gif')
        create_animation(sw_filenames, sw_animation_path)

    if lwsw:
        combined_animation_path = os.path.join(output_folder, 'combined_flux_animation_nipy.gif')
        create_animation(combined_filenames, combined_animation_path)

# Function to create animation
def create_animation(filenames, animation_path):
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(animation_path, images, duration=0.5)

def plot_fov_radiation(sat_lat, sat_lon, fov_radius, radiation_data, lat, lon, output_path, title):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=(8, 8))

    # Create meshgrid for lat/lon
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Compute the mask for the FOV
    fov_mask = is_within_fov_vectorized(sat_lat, sat_lon, fov_radius, lat2d, lon2d)
    radiation_data_fov = np.ma.masked_where(~fov_mask, radiation_data)

    # Transform each lat/lon to FOV coordinates
    r, theta = latlon_to_fov_coordinates(lat2d[fov_mask], lon2d[fov_mask], sat_lat, sat_lon, fov_radius)

    # Convert polar coordinates (r, theta) to Cartesian coordinates (X, Y)
    X = r * np.cos(theta)
    Y = r * np.sin(theta)

    # Define a grid to interpolate onto
    grid_x, grid_y = np.mgrid[-fov_radius:fov_radius:500j, -fov_radius:fov_radius:500j]

    # Interpolate the radiation data onto the grid
    grid_radiation = griddata((X, Y), radiation_data_fov[fov_mask].flatten(), (grid_x, grid_y), method='cubic', fill_value=0)

    # Apply circular mask to the grid
    D = np.sqrt(grid_x**2 + grid_y**2)
    grid_radiation_masked = np.ma.masked_where(D > fov_radius, grid_radiation)

    # Plotting
    img = ax.imshow(grid_radiation_masked, extent=(-fov_radius, fov_radius, -fov_radius, fov_radius), origin='lower', cmap='nipy_spectral')
    plt.colorbar(img, label='Radiation (J/m²)')
    ax.set_title(title)
    ax.set_xlabel('FOV X Coordinate (km)')
    ax.set_ylabel('FOV Y Coordinate (km)')

    plt.savefig(output_path)
    plt.close(fig)

import numpy as np
from scipy.interpolate import griddata

def plot_radiance_geiger(alts, lats, lons, ceres_indices, lw_radiation_data, sw_radiation_data, combined_radiation_data, lat, lon, ceres_times, number_of_tsteps, lw=True, sw=True, lwsw=True, output_folder="output/FOV_sliced_data"):
    os.makedirs(output_folder, exist_ok=True)

    # Initialize cumulative radiation data arrays for local coordinates
    #should be a roughly circular set of arrays
    local_fov_radiation = np.zeros((number_of_tsteps, 180, 180)) #this is like the geiger counter

    for idx, (alt, sat_lat, sat_lon, ceres_idx) in enumerate(zip(alts, lats, lons, ceres_indices)):
        if idx >= number_of_tsteps:
            break

        horizon_dist = calculate_satellite_fov(alt)
        time_step_duration = (ceres_times[ceres_idx + 1] - ceres_times[ceres_idx]) * 24 * 60

        if sw:
            _, _, _, _, sw_P_rad, _, _, _ = compute_radiance_at_sc("SW", ceres_idx, sw_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_dist, ceres_times)
            local_sw = convert_to_xy(sw_P_rad, sat_lat, sat_lon, horizon_dist, lat, lon) * time_step_duration
            #stretch the local_sw to be (180,180)- normally it is smaller
            local_sw = np.pad(local_sw, ((0, 180-local_sw.shape[0]), (0, 180-local_sw.shape[1])), 'constant', constant_values=0)
            local_fov_radiation[idx, :, :] += local_sw

        # Plot at each step
        if sw:
            sw_output_path = os.path.join(output_folder, f"cumulative_sw_{idx}.png")
            #now sum all the local_fov_radiation up to this point and plot it
            sum_cum_local_sw = np.sum(local_fov_radiation[:idx, :, :], axis=0)
            
            plt.figure(figsize=(10, 7))
            plt.imshow(sum_cum_local_sw, cmap='nipy_spectral')
            plt.colorbar()
            plt.title(f'SW - {idx}')
            plt.savefig(sw_output_path)
            # plt.show()
            plt.close()

def convert_to_xy(radiation_data, sat_lat, sat_lon, fov_radius, lat, lon):
    # Create meshgrid for lat/lon
    lon2d, lat2d = np.meshgrid(lon, lat)
    # Compute the mask for the FOV
    fov_mask = is_within_fov_vectorized(sat_lat, sat_lon, fov_radius, lat2d, lon2d)
    radiation_data_fov = np.ma.masked_where(~fov_mask, radiation_data)
    # Transform each lat/lon to FOV coordinates
    r, theta = latlon_to_fov_coordinates(lat2d[fov_mask], lon2d[fov_mask], sat_lat, sat_lon, fov_radius)
    # Convert polar coordinates (r, theta) to Cartesian coordinates (X, Y)
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    # Define a grid to interpolate onto
    grid_x, grid_y = np.mgrid[-fov_radius:fov_radius:180, -fov_radius:fov_radius:360]
    # Interpolate the radiation data onto the grid
    grid_radiation = griddata((X, Y), radiation_data_fov[fov_mask].flatten(), (grid_x, grid_y), method='linear', fill_value=0)
    return grid_radiation
