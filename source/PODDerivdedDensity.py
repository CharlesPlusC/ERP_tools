import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

import os
from tools.utilities import project_acc_into_HCL, calculate_acceleration, interpolate_positions, get_satellite_info
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def main():
    sat_names_to_test = ["CHAMP"]
    density_inversion_dfs = []
    for sat_name in sat_names_to_test:
        sat_info = get_satellite_info(sat_name)
        ephemeris_df = sp3_ephem_to_df(sat_name)
        force_model_configs = [
            # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'relativity': True},
            # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True},
            {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
            ]
        
        for force_model_config_number, force_model_config in enumerate(force_model_configs):

            settings = {
                'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
                'no_points_to_process': 180*2, 'filter_window_length': 21, 'filter_polyorder': 7,
                'ephemeris_interp_freq': '0.01S', 'density_freq': '15S'
            }
            
            ephemeris_df = ephemeris_df.head(settings['no_points_to_process'])
            interp_ephemeris_df = interpolate_positions(ephemeris_df, settings['ephemeris_interp_freq'])
            interp_ephemeris_df = calculate_acceleration(interp_ephemeris_df, 
                                                         settings['ephemeris_interp_freq'],
                                                         settings['filter_window_length'],
                                                         settings['filter_polyorder'])

            interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
            interp_ephemeris_df.set_index('UTC', inplace=True)
            interp_ephemeris_df = interp_ephemeris_df.asfreq(settings['density_freq'])
            
            if force_model_config_number == 0:
                density_inversion_df = pd.DataFrame(columns=[
                    'Epoch', 'Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density',
                    'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz'
                ])
            else:
                density_inversion_df = pd.DataFrame(columns=[
                    'Epoch', 'Computed Density', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz'
                ])

            for i in tqdm(range(1, len(interp_ephemeris_df)), desc='Processing Density Inversion'):
                epoch = interp_ephemeris_df.index[i]
                vel = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
                state_vector = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], vel[0], vel[1], vel[2]])
                conservative_accelerations = state2acceleration(state_vector, epoch, settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], **force_model_config)

                computed_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
                observed_acc = np.array([interp_ephemeris_df['accx'][i], interp_ephemeris_df['accy'][i], interp_ephemeris_df['accz'][i]])
                diff = computed_accelerations_sum - observed_acc
                diff_x, diff_y, diff_z = diff[0], diff[1], diff[2]
                _, _, diff_l = project_acc_into_HCL(diff_x, diff_y, diff_z, interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i])

                r = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i]])
                v = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
                atm_rot = np.array([0, 0, 72.9211e-6])
                v_rel = v - np.cross(atm_rot, r)
                rho = -2 * (diff_l / (settings['cd'] * settings['cross_section'])) * (settings['mass'] / np.abs(np.linalg.norm(v_rel))**2)
                time = interp_ephemeris_df.index[i]
                #only query for the first force_model_config
                if force_model_config_number == 0:
                    jb_08_rho = query_jb08(r, time)
                    dtm2000_rho = query_dtm2000(r, time)
                    nrlmsise00_rho = query_nrlmsise00(r, time)
                    new_row = pd.DataFrame({
                        'Epoch': [time], 'Computed Density': [rho], 'JB08 Density': [jb_08_rho], 'DTM2000 Density': [dtm2000_rho], 'NRLMSISE00 Density': [nrlmsise00_rho],
                        'x': [r[0]], 'y': [r[1]], 'z': [r[2]], 'xv': [v[0]], 'yv': [v[1]], 'zv': [v[2]],
                        'accx': [diff_x], 'accy': [diff_y], 'accz': [diff_z]
                    })
                else:
                    new_row = pd.DataFrame({
                        'Epoch': [time], 'Computed Density': [rho],
                        'x': [r[0]], 'y': [r[1]], 'z': [r[2]], 'xv': [v[0]], 'yv': [v[1]], 'zv': [v[2]],
                        'accx': [diff_x], 'accy': [diff_y], 'accz': [diff_z]
                    })
                density_inversion_df = pd.concat([density_inversion_df, new_row], ignore_index=True)
                density_inversion_dfs.append(density_inversion_df)

                save_folder = f"output/DensityInversion/PODBasedAccelerometry/Data/{sat_name}/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                filename = f"{datetime.datetime.now().strftime('%Y-%m-%d')}_{sat_name}_fm{force_model_config_number}_density_inversion.csv"
                density_inversion_df.to_csv(os.path.join(save_folder, filename), index=False)

    return density_inversion_dfs

def plot_density_data(data_frames, moving_avg_minutes, sat_name):
    sns.set_style(style="whitegrid")
    
    # Define color palette for all densities including model densities
    custom_palette = sns.color_palette("Set2", len(data_frames) + 3)  # Adding 3 for model densities

    # First plot for the computed densities MAs
    plt.figure(figsize=(10, 6))
    for i, density_df in enumerate(data_frames):
        print(f'Processing density_df {i+1}')
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
        density_df['Computed Density MA'] = density_df['Computed Density MA'].where(density_df['Computed Density MA'] >= 3e-13).ffill()
        
        sns.lineplot(data=density_df, x=density_df.index, y='Computed Density MA', label=f'Computed Density {i+1}', linestyle='--', palette=[custom_palette[i]])

    plt.title('Computed Density Moving Averages', fontsize=14)
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

if __name__ == "__main__":
    density_dfs = main()
    # densitydf_gfo_A = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-18_GRACE-FO-A_density_inversion.csv")
    # densitydf_gfo_B_fm0 = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-B/2024-04-22_GRACE-FO-B_fm0_density_inversion.csv")
    # densitydf_gfo_B_fm1 = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-B/2024-04-22_GRACE-FO-B_fm1_density_inversion.csv")
    # densitydf_gfo_B_fm2 = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-B/2024-04-22_GRACE-FO-B_fm2_density_inversion.csv")
    champ_density_df = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/CHAMP/2024-04-24_CHAMP_fm0_density_inversion.csv")
    density_dfs = [champ_density_df]
    # densitydf_tsx = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/TerraSAR-X/2024-04-19_TerraSAR-X_density_inversion.csv")
    sat_name = 'CHAMP'
    # density_compare_scatter(champ_density_df, 45)
    plot_density_data(density_dfs, 45, sat_name)
 

#TODO:
# Do a more systematic analysis of the effect of the interpolation window length and polynomial order on the RMS error
