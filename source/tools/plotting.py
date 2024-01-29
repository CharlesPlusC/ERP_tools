""" Plotting Module

"""
import numpy as np
from scipy.interpolate import griddata
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import datetime
import cartopy.crs as ccrs
from itertools import islice
from .ceres_data_processing import latlon_to_fov_coordinates, calculate_satellite_fov, is_within_fov, is_within_fov_vectorized, sat_normal_surface_angle_vectorized
from .utilities import convert_ceres_time_to_date, lla_to_ecef
import seaborn as sns
from matplotlib.gridspec import GridSpec

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

def plot_radiance_maps(time_index, variable_name, radiation_data, lat, lon, sat_lat, sat_lon, sat_alt, horizon_dist, output_path, ceres_times, plot_cosine_factors=False, plot_distances=False, plot_radiation=True):
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

def plot_radiance_geiger(alts, lats, lons, ceres_indices, lw_radiation_data, sw_radiation_data, combined_radiation_data, lat, lon, ceres_times, number_of_tsteps, lw=True, sw=True, lwsw=True, output_folder="output/FOV_sliced_data"):
    os.makedirs(output_folder, exist_ok=True)

    # Initialize cumulative radiation data arrays for local coordinates
    local_fov_radiation_lw = np.zeros((number_of_tsteps, 180, 180))
    local_fov_radiation_sw = np.zeros((number_of_tsteps, 180, 180))
    local_fov_radiation_lwsw = np.zeros((number_of_tsteps, 180, 180))

    plot_paths = []

    for idx, (alt, sat_lat, sat_lon, ceres_idx) in enumerate(zip(alts, lats, lons, ceres_indices)):
        if idx >= number_of_tsteps:
            break
        print("radiance geiger step:", idx)

        horizon_dist = calculate_satellite_fov(alt)
        time_step_duration = (ceres_times[ceres_idx + 1] - ceres_times[ceres_idx]) * 24 * 60

        # Process LW, SW, and LWSW radiation data
        if lw:
            _, _, _, _, lw_P_rad, _, _, _ = compute_radiance_at_sc("LW", ceres_idx, lw_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_dist, ceres_times)
            local_lw = convert_to_xy(lw_P_rad, sat_lat, sat_lon, horizon_dist, lat, lon) * time_step_duration
            local_lw = np.pad(local_lw, ((0, 180-local_lw.shape[0]), (0, 180-local_lw.shape[1])), 'constant', constant_values=0)
            local_fov_radiation_lw[idx, :, :] += local_lw

        if sw:
            _, _, _, _, sw_P_rad, _, _, _ = compute_radiance_at_sc("SW", ceres_idx, sw_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_dist, ceres_times)
            local_sw = convert_to_xy(sw_P_rad, sat_lat, sat_lon, horizon_dist, lat, lon) * time_step_duration
            local_sw = np.pad(local_sw, ((0, 180-local_sw.shape[0]), (0, 180-local_sw.shape[1])), 'constant', constant_values=0)
            local_fov_radiation_sw[idx, :, :] += local_sw

        if lwsw:
            _, _, _, _, lwsw_P_rad, _, _, _ = compute_radiance_at_sc("LWSW", ceres_idx, combined_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_dist, ceres_times)
            local_lwsw = convert_to_xy(lwsw_P_rad, sat_lat, sat_lon, horizon_dist, lat, lon) * time_step_duration
            local_lwsw = np.pad(local_lwsw, ((0, 180-local_lwsw.shape[0]), (0, 180-local_lwsw.shape[1])), 'constant', constant_values=0)
            local_fov_radiation_lwsw[idx, :, :] += local_lwsw

        # Plot at every 5th time step
        if idx % 5 == 0 and idx > 0:
            if lw or sw or lwsw:
                output_path = os.path.join(output_folder, f"cumulative_radiation_{idx}.png")
                plot_paths.append(output_path)
                fig, axes = plt.subplots(1, (lw + sw + lwsw), figsize=(15, 5))

                # Determine the maximum radiation value across all types for consistent colorbar scale
                max_radiation_value = max(
                    np.sum(local_fov_radiation_lw[:idx, :, :], axis=0).max(),
                    np.sum(local_fov_radiation_sw[:idx, :, :], axis=0).max(),
                    np.sum(local_fov_radiation_lwsw[:idx, :, :], axis=0).max()
                )

                formatted_time = convert_ceres_time_to_date(ceres_times[ceres_idx])

                for ax_idx, radiation_type in enumerate(['LW', 'SW', 'LWSW']):
                    if radiation_type == 'LW' and lw:
                        sum_cum_local = np.sum(local_fov_radiation_lw[:idx, :, :], axis=0)
                    elif radiation_type == 'SW' and sw:
                        sum_cum_local = np.sum(local_fov_radiation_sw[:idx, :, :], axis=0)
                    elif radiation_type == 'LWSW' and lwsw:
                        sum_cum_local = np.sum(local_fov_radiation_lwsw[:idx, :, :], axis=0)
                    else:
                        continue

                    ax = axes[ax_idx] if (lw + sw + lwsw) > 1 else axes
                    im = ax.imshow(sum_cum_local, cmap='nipy_spectral', vmin=0, vmax=max_radiation_value)
                    ax.set_title(f'Cumulative {radiation_type} Radiation\n CERES Time: {formatted_time}')
                    fig.colorbar(im, ax=ax, label='Joules/m²')

                    # Set aspect ratio and add grid lines
                    ax.set_aspect('equal')
                    for radius in range(25, 180, 25):
                        circle = plt.Circle((90, 90), radius, color='white', fill=False)
                        ax.add_patch(circle)

                    # Add thin center crosshair
                    ax.axhline(y=90, color='white', linestyle='-', linewidth=1)
                    ax.axvline(x=90, color='white', linestyle='-', linewidth=1)

                    # Hide the square frame
                    ax.set_xlim(0, 180)
                    ax.set_ylim(0, 180)

                    # Label axes
                    ax.set_xlabel('FOV x (\N{DEGREE SIGN})')
                    ax.set_ylabel('FOV y (\N{DEGREE SIGN})')

                plt.tight_layout()
                plt.savefig(output_path)
                plt.close(fig)

    combined_animation_path = os.path.join(output_folder, 'cumulative_flux_anim.gif')
    create_animation(plot_paths, combined_animation_path)

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
    # Define a 180x180 grid to interpolate onto
    grid_x, grid_y = np.mgrid[-fov_radius:fov_radius:180j, -fov_radius:fov_radius:180j] # Adjust to create a 180x180 grid
    # Interpolate the radiation data onto the grid
    grid_radiation = griddata((X, Y), radiation_data_fov[fov_mask].flatten(), (grid_x, grid_y), method='linear', fill_value=0)
    return grid_radiation

def plot_hcl_differences(hcl_diffs, time_data, titles, colors):
    _, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
    
    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.set_xlabel('Time (seconds from start)')
        ax.set_ylabel('Difference (meters)')
        ax.grid(True)

        for name, diffs in hcl_diffs.items():
            ax.plot(time_data, diffs[i], label=f'{name} - No ERP', color=colors[name], linestyle='--')
        ax.legend()

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'output/ERP_prop/{timenow}_HCL_differences.png')
    # plt.show()

def plot_kepels_evolution(keplerian_element_data, sat_name):
    # Define a list of colors for different ephemeris generators
    colors = ['xkcd:sky blue', 'xkcd:light red', 'xkcd:light green']

    # Titles for each subplot
    titles = ['Semi-Major Axis', 'Eccentricity', 'Inclination', 
              'Argument of Perigee', 'Right Ascension of Ascending Node', 'True Anomaly']

    # Plot Keplerian Elements (subplot 3x2) for each propagator
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

    # Custom legend handles
    legend_handles = []

    # Iterate over each subplot
    for ax_index, ax in enumerate(axes.flatten()):
        ax.set_title(titles[ax_index])
        ax.set_xlabel('Time (seconds from start)')
        ax.set_ylabel('Value')
        ax.grid(True)

        # Format y-axis to avoid scientific notation
        ax.ticklabel_format(useOffset=False, style='plain')

        # Plot data for each ephemeris generator
        for name_index, (name, keplerian_data) in enumerate(keplerian_element_data.items()):
            times = keplerian_data[0]
            keplerian_elements = keplerian_data[1]

            # Extract the i-th Keplerian element for each time point
            element_values = [element[ax_index] for element in keplerian_elements]

            # Use a different color for each name
            color = colors[name_index % len(colors)]

            # Plot the i-th Keplerian element for the current generator
            line, = ax.plot(times, element_values, color=color, linestyle='--')

            # Add to custom legend handles
            if ax_index == 0:  # Only add once
                legend_handles.append(mlines.Line2D([], [], color=color, linestyle='--', label=name))

    # Add figure-level legend
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(keplerian_element_data))

    plt.subplots_adjust(hspace=0.5, wspace=0.4, top=0.85)
    plt.tight_layout()

    # Save and show the plot
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'output/ERP_prop/{sat_name}_{timenow}_ERP_Kepels.png')
    # plt.show()

def format_array(array, precision=6):
    return np.array2string(array, formatter={'float_kind':lambda x: f"{x:.{precision}f}"})

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def combined_residuals_plot(observations_df, residuals_final, a_priori_estimate, optimized_state, force_model_config, final_RMS, sat_name, i, arc_num, estimate_drag):
    fig = plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    gs = GridSpec(5, 4, figure=fig)  # Increased number of rows in GridSpec

    # Scatter plots for position and velocity residuals
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])

    # Position residuals plot
    sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,0], ax=ax1, color="xkcd:blue", s=10, label='x')
    sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,1], ax=ax1, color="xkcd:green", s=10, label='y')
    sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,2], ax=ax1, color="xkcd:red", s=10, label='z')
    ax1.set_ylabel("Position Residual (m)")
    ax1.set_xlabel("Observation time (UTC)")
    ax1.legend()

    # Velocity residuals plot
    sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,3], ax=ax2, color="xkcd:purple", s=10, label='u')
    sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,4], ax=ax2, color="xkcd:orange", s=10, label='v')
    sns.scatterplot(data=observations_df, x='UTC', y=residuals_final[:,5], ax=ax2, color="xkcd:yellow", s=10, label='w')
    ax2.set_ylabel("Velocity Residual (m/s)")
    ax2.set_xlabel("Observation time (UTC)")
    ax2.legend()

    # Histograms for position and velocity residuals
    ax3 = fig.add_subplot(gs[2, :2])
    ax4 = fig.add_subplot(gs[2, 2:])

    sns.histplot(residuals_final[:,0:3], bins=20, ax=ax3, palette=["xkcd:blue", "xkcd:green", "xkcd:red"], legend=False)
    ax3.set_xlabel("Position Residual (m)")
    ax3.set_ylabel("Frequency")
    ax3.legend(['x', 'y', 'z'])

    sns.histplot(residuals_final[:,3:6], bins=20, ax=ax4, palette=["xkcd:purple", "xkcd:orange", "xkcd:yellow"], legend=False)
    ax4.set_xlabel("Velocity Residual (m/s)")
    ax4.set_ylabel("Frequency")
    ax4.legend(['u', 'v', 'w'])

    # Table for force model configuration, initial state, and final estimated state
    ax5 = fig.add_subplot(gs[3:5, :])  # Increase the height of the table
    formatted_initial_state = format_array(a_priori_estimate, precision=6)
    formatted_optimized_state = format_array(optimized_state, precision=6)
    force_model_data = [
        ['Force Model Config', str(force_model_config)],
        ['Initial State', formatted_initial_state],
        ['Final Estimated State', formatted_optimized_state],
        ['Estimated Parameters', 'Position, Velocity' + (', C_D' if estimate_drag else '')]
    ]
    table = plt.table(cellText=force_model_data, colWidths=[0.2, 0.8], loc='center', cellLoc='left')  # Adjust column widths
    ax5.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.4)  # Adjust the scale for table height

    plt.suptitle(f"{sat_name} - Residuals (O-C) for best BLS iteration, RMS: {final_RMS:.3f}", fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.4)

    # Save the combined plot
    sat_name_folder = f"output/OD_BLS/Tapley/combined_plots/{sat_name}"
    if not os.path.exists(sat_name_folder):
        os.makedirs(sat_name_folder)
    obs_length_folder = f"{sat_name_folder}/{len(observations_df)}"
    if not os.path.exists(obs_length_folder):
        os.makedirs(obs_length_folder)
    save_path = f"{obs_length_folder}/arcnum{arc_num}_fmodel_{i}_estdrag_{estimate_drag}.png"
    plt.savefig(save_path)
    print(f"saving to {save_path}")
    plt.close()
