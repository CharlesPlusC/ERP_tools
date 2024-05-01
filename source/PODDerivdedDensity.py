import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

import os
from tools.utilities import pv_to_kep, project_acc_into_HCL, calculate_acceleration, interpolate_positions, get_satellite_info
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from org.orekit.utils import PVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from orekit.pyhelpers import setup_orekit_curdir, datetime_to_absolutedate
from tools.GFODataReadTools import get_gfo_inertial_accelerations

def density_compare_scatter(density_df, moving_avg_window, sat_name):
    
    save_path = f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert moving average minutes to the number of points based on data frequency
    if not isinstance(density_df.index, pd.DatetimeIndex):
        density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True)
        density_df.set_index('Epoch', inplace=True)
    
    # Calculate moving average for the Computed Density
    freq_in_seconds = pd.to_timedelta(pd.infer_freq(density_df.index)).seconds
    window_size = (moving_avg_window * 60) // freq_in_seconds
    
    # Compute the moving average for Computed Density
    density_df['Computed Density MA'] = density_df['Computed Density'].rolling(window=window_size, center=False).mean()

    # Calculate the number of points to shift equivalent to half of the moving average window
    shift_periods = int((moving_avg_window / 2 * 60) // freq_in_seconds)

    # Shift the moving average back by the calculated periods
    density_df['Computed Density MA'] = density_df['Computed Density MA'].shift(-shift_periods)

    # Model names to compare
    model_names = ['JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']

    for model in model_names:
        plot_data = density_df.dropna(subset=['Computed Density MA', model])
        plot_data = plot_data[plot_data['Computed Density MA'] > 0]  # Ensure positive values for log scale
        
        f, ax = plt.subplots(figsize=(6, 6))

        # Draw a combo histogram and scatterplot with density contours
        sns.scatterplot(x=plot_data[model], y=plot_data['Computed Density MA'], s=5, color=".15", ax=ax)
        sns.histplot(x=plot_data[model], y=plot_data['Computed Density MA'], bins=50, pthresh=.1, cmap="rocket", cbar=True, ax=ax)
        sns.kdeplot(x=plot_data[model], y=plot_data['Computed Density MA'], levels=4, color="xkcd:white", linewidths=1, ax=ax)
        #log the x and y 
        ax.set_xscale('log')
        ax.set_yscale('log')
        #add a line of y=x
        ax.plot([1e-13, 1e-11], [1e-13, 1e-11], color='black', linestyle='--')
        #constrain the axes to be between 1e-13 and 1e-11 and of same length
        ax.set_xlim(1e-13, 3e-12)
        ax.set_ylim(1e-13, 3e-12)
        ax.set_title(f'Comparison of {model} vs. Computed Density')
        ax.set_xlabel('Model Density')
        ax.set_ylabel('Computed Density')
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        plot_filename = f'comparison_{model.replace(" ", "_")}.png'
        plt.savefig(os.path.join(save_path, plot_filename))
        plt.close()

        # Line plot of density over time for both the model and the computed density
        plt.figure(figsize=(11, 7))
        plt.plot(plot_data.index, plot_data['Computed Density MA'], label='Computed Density')
        plt.plot(plot_data.index, plot_data[model], label=model)
        plt.title(f'{model} vs. Computed Density Over Time')
        plt.xlabel('Epoch (UTC)')
        plt.ylabel('Density')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plot_filename = f'comparison_{model.replace(" ", "_")}_time.png'
        plt.savefig(os.path.join(save_path, plot_filename))
        plt.close()

def plot_density_data(data_frames, moving_avg_minutes, sat_name):
    sns.set_style(style="whitegrid")
    
    # Define color palette for all densities including model densities
    custom_palette = sns.color_palette("Set2", len(data_frames) + 3)  # Adding 3 for model densities

    # First plot for the computed densities MAs
    plt.figure(figsize=(10, 6))
    for i, density_df in enumerate(data_frames):
        print(f'Processing density_df {i+1}')
        print(f'columns: {density_df.columns}')
        if density_df['Epoch'].dtype != 'datetime64[ns]':
            density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True)
        if not isinstance(density_df.index, pd.DatetimeIndex):
            density_df.set_index('Epoch', inplace=True)
        if pd.infer_freq(density_df.index) is None:
            density_df = density_df.asfreq('infer')
        seconds_per_point = pd.to_timedelta(pd.infer_freq(density_df.index)).seconds
        window_size = (moving_avg_minutes * 60) // seconds_per_point
        density_df['Computed Density MA'] = density_df['Computed Density'].rolling(window=window_size, center=False).mean()
        shift_periods = int((moving_avg_minutes / 2 * 60) // seconds_per_point)
        density_df['Computed Density MA'] = density_df['Computed Density MA'].shift(-shift_periods)
        
        sns.lineplot(data=density_df, x=density_df.index, y='Computed Density MA', label=f'Computed Density {i+1}', linestyle='--', palette=[custom_palette[i]])

    plt.title(f'Computed and Modelled Atmospheric Density for {sat_name}', fontsize=14)
    plt.xlabel('Epoch (UTC)', fontsize=14)
    plt.ylabel('Density (log scale)', fontsize=14)
    plt.legend(loc='upper right', frameon=True)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    datenow = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/computed_density_moving_averages_{datenow}.png')

    # Second plot for the first data frame with model densities along with computed densities
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='JB08 Density', label='JB08 Density', color=custom_palette[-3])
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='DTM2000 Density', label='DTM2000 Density', color=custom_palette[-2])
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='NRLMSISE00 Density', label='NRLMSISE00 Density', color=custom_palette[-1])

    # Include computed densities from all data frames again on the same plot
    for i, density_df in enumerate(data_frames):
        sns.lineplot(data=density_df, x=density_df.index, y='Computed Density MA', label=f'Computed Density {i+1} (overlay)', linestyle='--', palette=[custom_palette[i]])

    plt.title('Model Densities vs. Computed Density Moving Averages', fontsize=14)
    plt.xlabel('Epoch (UTC)', fontsize=14)
    plt.ylabel('Density (log scale)', fontsize=14)
    plt.legend(loc='upper right', frameon=True)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/model_density_vs_computed_density_{datenow}.png')

from matplotlib.colors import LogNorm
# def plot_density_arglat(data_frames, moving_avg_minutes, sat_name):
#     sns.set_style("darkgrid", {
#         'axes.facecolor': '#2d2d2d', 'axes.edgecolor': 'white', 
#         'axes.labelcolor': 'white', 'xtick.color': 'white', 
#         'ytick.color': 'white', 'figure.facecolor': '#2d2d2d', 'text.color': 'white'
#     })
    
#     density_types = ['Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']
#     titles = ['Computed Density', 'JB08 Model Density', 'DTM2000 Model Density', 'NRLMSISE00 Model Density']

#     # Check if 'Accelerometer Density' is present and add it to lists
#     if 'Accelerometer Density' in data_frames[0].columns:
#         density_types.append('Accelerometer Density')
#         titles.append('Accelerometer Density')
#         # Multiply all the accelerometer densities by -1
#         for density_df in data_frames:
#             density_df['Accelerometer Density'] *= -1
    
#     # Create subplot with a dynamic number of rows based on the number of density types
#     nrows = len(density_types)
#     fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 7 * nrows // 2), dpi=100)
#     if nrows == 1:
#         axes = [axes]  # Ensure axes is always iterable

#     vmin, vmax = 3e-13, 2e-12

#     for i, density_df in enumerate(data_frames):
#         density_df = get_arglat_from_df(density_df)
#         density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True)
#         density_df.set_index('Epoch', inplace=True)

#         if moving_avg_minutes > 0:
#             window_size = (moving_avg_minutes * 60) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds
#             shift_periods = (moving_avg_minutes * 30) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds

#             for density_type in density_types:
#                 if density_type in density_df.columns:
#                     density_df[f'{density_type} MA'] = density_df[density_type].rolling(window=window_size, min_periods=1, center=True).mean().shift(-shift_periods)
#                     density_df = density_df.iloc[5:-5]  # Adjust indices as necessary
#         else:
#             for density_type in density_types:
#                 if density_type in density_df.columns:
#                     density_df[f'{density_type} MA'] = density_df[density_type]

#         for j, density_type in enumerate(density_types):
#             if f'{density_type} MA' in density_df.columns:
#                 sc = axes[j].scatter(density_df.index, density_df['arglat'], c=density_df[f'{density_type} MA'], cmap='cubehelix', alpha=0.6, edgecolor='none', norm=LogNorm(vmin=vmin, vmax=vmax))
#                 axes[j].set_title(titles[j], fontsize=12)
#                 axes[j].set_xlabel('Time (UTC)')
#                 axes[j].set_ylabel('Argument of Latitude')
#                 for label in axes[j].get_xticklabels():
#                     label.set_rotation(45)
#                     label.set_horizontalalignment('right')
#                 cbar = fig.colorbar(sc, ax=axes[j])
#                 cbar.set_label('Density (kg/m³)', rotation=270, labelpad=15)

#     plt.suptitle(f'Atmospheric Density as Function of Argument of Latitude for {sat_name}')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/density_arglat_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', dpi=600)
#     plt.show()

def plot_density_arglat_diff(data_frames, moving_avg_minutes, sat_name):
    sns.set_style("darkgrid", {
        'axes.facecolor': '#2d2d2d', 'axes.edgecolor': 'white', 
        'axes.labelcolor': 'white', 'xtick.color': 'white', 
        'ytick.color': 'white', 'figure.facecolor': '#2d2d2d', 'text.color': 'white'
    })
    
    density_types = ['Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']
    titles = ['Computed Density', 'JB08 Model Density', 'DTM2000 Model Density', 'NRLMSISE00 Model Density']
    density_diff_titles = ['|Computed - JB08|', '|Computed - DTM2000|', '|Computed - NRLMSISE00|']

    nrows = len(density_types)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 4 * nrows), dpi=100)

    vmin, vmax = 3e-13, 2e-12
    diff_vmin, diff_vmax = 1e-15, 1e-11

    for i, density_df in enumerate(data_frames):
        density_df = get_arglat_from_df(density_df)
        density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True)
        density_df.set_index('Epoch', inplace=True)

        window_size = (moving_avg_minutes * 60) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds if moving_avg_minutes > 0 else 1
        shift_periods = (moving_avg_minutes * 30) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds if moving_avg_minutes > 0 else 0

        for density_type in density_types:
            if density_type in density_df.columns:
                density_df[f'{density_type} MA'] = density_df[density_type].rolling(window=window_size, min_periods=1, center=True).mean().shift(-shift_periods)
                if density_type != 'Computed Density':
                    density_df[f'{density_type} Difference'] = abs(density_df['Computed Density MA'] - density_df[f'{density_type} MA'])

        for j, density_type in enumerate(density_types):
            if f'{density_type} MA' in density_df.columns:
                sc = axes[j, 0].scatter(density_df.index, density_df['arglat'], c=density_df[f'{density_type} MA'], cmap='cubehelix', alpha=0.6, edgecolor='none', norm=LogNorm(vmin=vmin, vmax=vmax))
                axes[j, 0].set_title(titles[j], fontsize=12)
                axes[j, 0].set_xlabel('Time (UTC)')
                axes[j, 0].set_ylabel('Argument of Latitude')
                cbar = fig.colorbar(sc, ax=axes[j, 0])
                cbar.set_label('Density (kg/m³)', rotation=270, labelpad=15)
            if density_type != 'Computed Density' and f'{density_type} Difference' in density_df.columns:
                sc_diff = axes[j, 1].scatter(density_df.index, density_df['arglat'], c=density_df[f'{density_type} Difference'], cmap='coolwarm', alpha=0.6, edgecolor='none', norm=LogNorm(vmin=diff_vmin, vmax=diff_vmax))
                axes[j, 1].set_title(density_diff_titles[j - 1], fontsize=12)
                axes[j, 1].set_xlabel('Time (UTC)')
                axes[j, 1].set_ylabel('Argument of Latitude')
                cbar_diff = fig.colorbar(sc_diff, ax=axes[j, 1])
                cbar_diff.set_label('Density Difference (kg/m³)', rotation=270, labelpad=15)

    plt.suptitle(f'Atmospheric Density as Function of Argument of Latitude for {sat_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/densitydiff_arglat{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', dpi=600)

def get_arglat_from_df(densitydf_df):
    frame = FramesFactory.getEME2000()
    use_column = 'Epoch' in densitydf_df.columns

    for index, row in densitydf_df.iterrows():
        x = row['x']
        y = row['y']
        z = row['z']
        xv = row['xv']
        yv = row['yv']
        zv = row['zv']
        
        if use_column:
            UTC = row['Epoch']
            # Check if UTC needs to be converted from string to datetime
            if isinstance(UTC, str):
                UTC = datetime.datetime.strptime(UTC, '%Y-%m-%d %H:%M:%S')
        else:
            UTC = index  # Use the index directly, which should already be in datetime format

        position = Vector3D(float(x), float(y), float(z))
        velocity = Vector3D(float(xv), float(yv), float(zv))
        pvCoordinates = PVCoordinates(position, velocity)
        time = datetime_to_absolutedate(UTC)
        kep_els = pv_to_kep(pvCoordinates, frame, time)
        arglat = kep_els[3] + kep_els[5]
        densitydf_df.at[index, 'arglat'] = arglat

    return densitydf_df

def density_inversion(sat_name, interp_ephemeris_df, force_model_config, accelerometer_data=None):
    sat_info = get_satellite_info(sat_name)
    settings = {
        'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
        'density_freq': '15S'
    }

    # Convert UTC column to datetime and set it as index
    interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
    interp_ephemeris_df.set_index('UTC', inplace=True)
    interp_ephemeris_df = interp_ephemeris_df.asfreq(settings['density_freq'])

    # Handle accelerometer data
    if accelerometer_data is not None:
        accelerometer_data['UTC'] = pd.to_datetime(accelerometer_data['UTC'])
        accelerometer_data.set_index('UTC', inplace=True)
        accelerometer_data = accelerometer_data.asfreq(settings['density_freq'])
        interp_ephemeris_df = pd.merge(interp_ephemeris_df, accelerometer_data, how='left', left_index=True, right_index=True, suffixes=('', '_drop'))

        # Drop the duplicate columns from the accelerometer data
        columns_to_drop = [col for col in interp_ephemeris_df.columns if '_drop' in col]
        interp_ephemeris_df.drop(columns=columns_to_drop, inplace=True)

        columns = [
            'Epoch', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz',
            *(['JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density'])
        ]
        if accelerometer_data is not None:
            columns.append('Accelerometer Density')

        density_inversion_df = pd.DataFrame(columns=columns)

        for i in tqdm(range(1, len(interp_ephemeris_df)), desc='Processing Density Inversion'):
            epoch = interp_ephemeris_df.index[i]
            vel = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
            state_vector = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], vel[0], vel[1], vel[2]])
            if accelerometer_data is not None:
                force_model_config = {'3BP': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},

            conservative_accelerations = state2acceleration(state_vector, epoch, settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], **force_model_config)

            computed_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
            if accelerometer_data is not None:
                observed_acc = np.array([interp_ephemeris_df['inertial_x_acc'][i], interp_ephemeris_df['inertial_y_acc'][i], interp_ephemeris_df['inertial_z_acc'][i]])
            else:
                observed_acc = np.array(interp_ephemeris_df['accx'][i], interp_ephemeris_df['accy'][i], interp_ephemeris_df['accz'][i])

            diff = computed_accelerations_sum - observed_acc
            diff_x, diff_y, diff_z = diff[0], diff[1], diff[2]
            _, _, diff_l = project_acc_into_HCL(diff_x, diff_y, diff_z, interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i])

            r = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i]])
            v = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
            atm_rot = np.array([0, 0, 72.9211e-6])
            v_rel = v - np.cross(atm_rot, r)
            rho = -2 * (diff_l / (settings['cd'] * settings['cross_section'])) * (settings['mass'] / np.abs(np.linalg.norm(v_rel))**2)
            time = epoch

            if accelerometer_data is None:
                row_data = {
                    'Epoch': time, 'x': r[0], 'y': r[1], 'z': r[2], 'xv': v[0], 'yv': v[1], 'zv': v[2],
                    'accx': diff_x, 'accy': diff_y, 'accz': diff_z, 'Computed Density': rho,
                    **({key: value for key, value in zip(['JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density'], 
                                                         [query_jb08(r, time), query_dtm2000(r, time), query_nrlmsise00(r, time)])})
                            }
                
            if accelerometer_data is not None:
                row_data = {
                    'Epoch': time, 'x': r[0], 'y': r[1], 'z': r[2], 'xv': v[0], 'yv': v[1], 'zv': v[2],
                    'accx': diff_x, 'accy': diff_y, 'accz': diff_z, 'Accelerometer Density': rho, 'Computed Density': interp_ephemeris_df['Computed Density'][i],
                    #TODO: somehow the accelerometer density is inverted... 
                    #take the exisitng jb08, dtm2000, and nrlmsise00 densities from the interp_ephemeris_df 
                    **({key: interp_ephemeris_df[key][i] for key in ['JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']})
                }

            new_row = pd.DataFrame(row_data, index=[0])
            density_inversion_df = pd.concat([density_inversion_df, new_row], ignore_index=True)

    return density_inversion_df

def save_density_inversion_data(sat_name, density_inversion_dfs):
    save_folder = f"output/DensityInversion/PODBasedAccelerometry/Data/{sat_name}/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for i, df in enumerate(density_inversion_dfs):
        filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{sat_name}_fm{i}_density_inversion.csv"
        if isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(save_folder, filename), index=False)

def plot_relative_density_change(data_frames, moving_avg_minutes, sat_name):
    sns.set_style("darkgrid", {
        'axes.facecolor': '#2d2d2d', 'axes.edgecolor': 'white',
        'axes.labelcolor': 'white', 'xtick.color': 'white',
        'ytick.color': 'white', 'figure.facecolor': '#2d2d2d', 'text.color': 'white'
    })

    density_types = ['Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']
    titles = ['Delta Density: Computed vs JB08', 'Delta Density: Computed vs DTM2000', 'Delta Density: Computed vs NRLMSISE00']

    fig, axes = plt.subplots(nrows=len(titles), ncols=1, figsize=(7, 3 * len(titles)), dpi=200)

    for density_df in data_frames:
        density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True) if 'Epoch' in density_df.columns else density_df.index
        density_df = get_arglat_from_df(density_df)
        if 'Epoch' in density_df.columns:
            density_df.set_index('Epoch', inplace=True)

        
        # Calculate moving average
        window_size = (moving_avg_minutes * 60) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds if moving_avg_minutes > 0 else 1
        for density_type in density_types:
            if density_type in density_df.columns:
                density_df[f'{density_type} MA'] = density_df[density_type].rolling(window=window_size, min_periods=1, center=True).mean()
                #drop the first and last 450 points
                density_df = density_df.iloc[450:-450]

        # Calculate delta density relative to initial value
        for density_type in density_types:
            initial_value = density_df[f'{density_type} MA'].iloc[0]
            density_df[f'{density_type} Delta'] = density_df[f'{density_type} MA'] - initial_value

        # Calculate relative change in delta densities
        for j, title in enumerate(titles):
            model_density = density_types[j + 1]  # skip 'Computed Density' for title indexing
            if f'{model_density} Delta' in density_df.columns:
                density_df[f'Relative Change {model_density}'] = density_df['Computed Density Delta'] - density_df[f'{model_density} Delta']
                sc = axes[j].scatter(density_df.index, density_df['arglat'], c=density_df[f'Relative Change {model_density}'], cmap='nipy_spectral', alpha=0.6, edgecolor='none')
                axes[j].set_title(title, fontsize=12)
                axes[j].set_xlabel('Time (UTC)')
                for label in axes[j].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                axes[j].set_ylabel('Argument of Latitude')
                cbar = fig.colorbar(sc, ax=axes[j])
                cbar.set_label('Delta Density Difference (kg/m³)', rotation=270, labelpad=15)

    plt.suptitle(f'Relative Change in Atmospheric Density for {sat_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/rel_densitydiff_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', dpi=600)

def main():
    sat_names_to_test = ["GRACE-FO-A"]
    force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        # ephemeris_df = ephemeris_df.head(180*35)
        # interp_ephemeris_df = interpolate_positions(ephemeris_df, '0.01S')
        # interp_ephemeris_df = calculate_acceleration(interp_ephemeris_df, '0.01S', 21, 7)
        interp_ephemeris_df = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-26_01-22-32_GRACE-FO-A_fm12597_density_inversion.csv")

        acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-05_C_04.txt"
        quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-05_C_04.txt"
        inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)
        ephemeris_df_copy = interp_ephemeris_df.copy()

        # Ensure utc_time in both DataFrames is converted to datetime
        ephemeris_df_copy['Epoch'] = pd.to_datetime(ephemeris_df_copy['Epoch'])
        #rename Epoch to UTC
        ephemeris_df_copy.rename(columns={'Epoch': 'UTC'}, inplace=True)
        inertial_gfo_data.rename(columns={'utc_time': 'UTC'}, inplace=True)
        inertial_gfo_data['UTC'] = pd.to_datetime(inertial_gfo_data['UTC'])

        merged_df = pd.merge(inertial_gfo_data, ephemeris_df_copy, on='UTC')
        density_inversion_dfs_acc = density_inversion(sat_name, merged_df, force_model_config, accelerometer_data=inertial_gfo_data)
        pd.DataFrame(density_inversion_dfs_acc).to_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-26_01-22-32_GRACE-FO-A_fm12597_density_inversion_with_acc.csv")

if __name__ == "__main__":
    # main()

    #TODO:# # Do a more systematic analysis of the effect of the interpolation window length and polynomial order on the RMS error
    densitydf_gfoa = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-26_01-22-32_GRACE-FO-A_fm12597_density_inversion.csv")
    densitydf_tsx = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/TerraSAR-X/2024-04-26_06-24-57_TerraSAR-X_fm12597_density_inversion.csv")
    # # #read in the x,y,z,xv,yv,zv, and UTC from the densitydf_df
    sat_names = ["GRACE-FO-A", "TerraSAR-X"]
    for df_num, density_df in enumerate([densitydf_gfoa, densitydf_tsx]):
        density_dfs = [density_df]
        #SELECT THE SAT NAME IN USING THE NU
        sat_name = sat_names[df_num]
        print(f"sat_name: {sat_name}")
        plot_density_arglat_diff(density_dfs, 45, sat_name)
        # plot_density_data(density_dfs, 45, sat_name)
        plot_relative_density_change(density_dfs, 45, sat_name)

# def density_inversion(sat_name, interp_ephemeris_df, force_model_configs, accelerometer_data=None):
#     density_inversion_dfs = []
#     sat_info = get_satellite_info(sat_name)
#     settings = {
#         'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
#         'density_freq': '15S'
#     }

#     interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
#     interp_ephemeris_df.set_index('UTC', inplace=True)
#     interp_ephemeris_df = interp_ephemeris_df.asfreq(settings['density_freq'])

#     for force_model_config_number, force_model_config in enumerate(force_model_configs):
#         if force_model_config_number == 0:
#             density_inversion_df = pd.DataFrame(columns=[
#                 'Epoch', 'Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density',
#                 'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz'
#             ])
#         else:
#             density_inversion_df = pd.DataFrame(columns=[
#                 'Epoch', 'Computed Density', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz'
#             ])

#         for i in tqdm(range(1, len(interp_ephemeris_df)), desc='Processing Density Inversion'):
#             epoch = interp_ephemeris_df.index[i]
#             vel = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
#             state_vector = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], vel[0], vel[1], vel[2]])
#             conservative_accelerations = state2acceleration(state_vector, epoch, settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], **force_model_config)

#             computed_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
#             observed_acc = np.array([interp_ephemeris_df['accx'][i], interp_ephemeris_df['accy'][i], interp_ephemeris_df['accz'][i]])
#             diff = computed_accelerations_sum - observed_acc
#             diff_x, diff_y, diff_z = diff[0], diff[1], diff[2]
#             _, _, diff_l = project_acc_into_HCL(diff_x, diff_y, diff_z, interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i])

#             r = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i]])
#             v = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
#             atm_rot = np.array([0, 0, 72.9211e-6])
#             v_rel = v - np.cross(atm_rot, r)
#             rho = -2 * (diff_l / (settings['cd'] * settings['cross_section'])) * (settings['mass'] / np.abs(np.linalg.norm(v_rel))**2)
#             time = interp_ephemeris_df.index[i]
#             if force_model_config_number == 0:
#                 jb_08_rho = query_jb08(r, time)
#                 dtm2000_rho = query_dtm2000(r, time)
#                 nrlmsise00_rho = query_nrlmsise00(r, time)
#                 new_row = pd.DataFrame({
#                     'Epoch': [time], 'Computed Density': [rho], 'JB08 Density': [jb_08_rho], 'DTM2000 Density': [dtm2000_rho], 'NRLMSISE00 Density': [nrlmsise00_rho],
#                     'x': [r[0]], 'y': [r[1]], 'z': [r[2]], 'xv': [v[0]], 'yv': [v[1]], 'zv': [v[2]],
#                     'accx': [diff_x], 'accy': [diff_y], 'accz': [diff_z]
#                 })
#             else:
#                 new_row = pd.DataFrame({
#                     'Epoch': [time], 'Computed Density': [rho],
#                     'x': [r[0]], 'y': [r[1]], 'z': [r[2]], 'xv': [v[0]], 'yv': [v[1]], 'zv': [v[2]],
#                     'accx': [diff_x], 'accy': [diff_y], 'accz': [diff_z]
#                 })
#             density_inversion_df = pd.concat([density_inversion_df, new_row], ignore_index=True)

#             density_inversion_dfs.append(density_inversion_df)

#     return density_inversion_dfs