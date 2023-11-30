""" Plotting Module

This module contains functions for plotting data, including FoV radiance in different projections.
Add detailed module description here.

"""
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import imageio
from .data_processing import is_within_fov, is_within_fov_vectorized, sat_normal_surface_angle_vectorized
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

def plot_fov_radiance(variable_name, time_index, radiation_data, lat, lon, sat_lat, sat_lon, sat_alt, horizon_dist, output_path, ceres_times):
    R = 6371  # Earth's radius in km

    # Ensure the mesh grid creation is correct
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Call the lla_to_ecef function
    ecef_x, ecef_y, ecef_z = lla_to_ecef(lat2d, lon2d, np.zeros_like(lat2d))
    fov_mask = is_within_fov_vectorized(sat_lat, sat_lon, horizon_dist, lat2d, lon2d)
    radiation_data_fov = np.ma.masked_where(~fov_mask, radiation_data[time_index, :, :])
    cos_thetas = sat_normal_surface_angle_vectorized(sat_lat, sat_lon, lat2d[fov_mask], lon2d[fov_mask])
    cosine_factors_2d = np.zeros_like(radiation_data_fov)
    cosine_factors_2d[fov_mask] = cos_thetas
    #I think the cosine factors might be wrong. Looks like they are stronger at one edge of the fov
    
    #plot cosine factors on map
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    vmin, vmax = (-1, 1)
    cmap = 'nipy_spectral'
    mesh_plot = ax.pcolormesh(lon2d, lat2d, cosine_factors_2d, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(mesh_plot, orientation='vertical', shrink=0.7)
    cbar.set_label('Cosine Factors')
    # Extract timestamp for title
    timestamp = convert_ceres_time_to_date(ceres_times[time_index])
    plt.title(f'{variable_name} - {timestamp}')
    ax.gridlines(draw_labels=True)
    #modiufy output path to include cosine factors
    cosine_map_output_path = output_path[:-4] + "_cosine_map.png"
    plt.savefig(cosine_map_output_path)

    adjusted_radiation_data = radiation_data_fov * cosine_factors_2d

    # Call the lla_to_ecef function
    ecef_x, ecef_y, ecef_z = lla_to_ecef(lat2d, lon2d, np.zeros_like(lat2d))
    # Stack the results
    ecef_pixels = np.stack((ecef_x, ecef_y, ecef_z), axis=-1)

    # Satellite ECEF Position 
    sat_ecef = np.array(lla_to_ecef(sat_lat, sat_lon, sat_alt))

    # Reshape for broadcasting
    sat_ecef_reshaped = sat_ecef.reshape((1, 1, 3))

    # Calculate vector difference
    vector_diff = sat_ecef_reshaped - ecef_pixels

    # Calculate the Euclidean distance
    distances = np.linalg.norm(vector_diff, axis=2)
    # Debugging: Print distances for a few pixels

    #plot a map of the distances
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cmap = 'nipy_spectral'
    mesh_plot = ax.pcolormesh(lon2d, lat2d, distances, cmap=cmap,  transform=ccrs.PlateCarree())
    cbar = plt.colorbar(mesh_plot, orientation='vertical', shrink=0.7)
    cbar.set_label('Distance to Satellite (km)')
    # Extract timestamp for title
    timestamp = convert_ceres_time_to_date(ceres_times[time_index])
    plt.title(f'{variable_name} - {timestamp}')
    ax.gridlines(draw_labels=True)
    #modiufy output path to include dist map
    distance_map_output_path = output_path[:-4] + "_distance_map.png"
    plt.savefig(distance_map_output_path)

    # Radiation calculation
    delta_lat = np.abs(lat[1] - lat[0])
    delta_lon = np.abs(lon[1] - lon[0])
    lat_radians = np.radians(lat2d)
    area_pixel = R**2 * np.radians(delta_lat) * np.radians(delta_lon) * np.cos(lat_radians)
    #convert area pixel to m^2
    area_pixel = area_pixel * (1000**2)
    #convert distances to m
    distances = distances * 1000
    P_rad = adjusted_radiation_data * area_pixel / (np.pi * distances**2) #this aligns with equation 11 in the paper- although does not include the ADM anisotropic factor

    # print("radiation flux reaching satellite:", np.sum(P_rad))
    # print("force due to radiation:", np.sum(P_rad) / 299792458)
    # print("acceleration into a 1000kg satellite:", (np.sum(P_rad) / 299792458) / 1000)

    # Plotting P_rad
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    vmin = 0
    vmax = 1
    cmap = 'nipy_spectral'

    mesh_plot = ax.pcolormesh(lon2d, lat2d, P_rad, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(mesh_plot, orientation='vertical', shrink=0.7)
    cbar.set_label('Radiation Reaching Satellite from each pixel (W/m^2)')
    #add the total radiation flux to the title

    # Extract timestamp for title
    timestamp = convert_ceres_time_to_date(ceres_times[time_index])
    plt.title(f'rad source:{variable_name} \n {timestamp} \nrad flux:{np.sum(P_rad):.2f} W/m^2')

    ax.gridlines(draw_labels=True)
    plt.savefig(output_path)
    plt.close(fig)

# Function to create animation
def create_animation(filenames, animation_path):
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(animation_path, images, duration=0.5)