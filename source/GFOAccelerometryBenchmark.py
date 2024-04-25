import pandas as pd
import matplotlib.pyplot as plt
import orekit
from orekit.pyhelpers import setup_orekit_curdir
# download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants
from tools.utilities import interpolate_positions,calculate_acceleration, get_satellite_info, project_acc_into_HCL
from tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
from tools.sp3_2_ephemeris import sp3_ephem_to_df
import numpy as np
import datetime
from tqdm import tqdm
from tools.GFODataReadTools import get_gfo_inertial_accelerations

def compute_acc_from_vel(sat_name = "GRACE-FO-A", 
                         start_date = datetime.datetime(2023, 5, 5, 0, 0, 0), 
                         end_date = datetime.datetime(2023, 5, 5, 2, 0, 0),
                         window_length=21, polyorder=7,
                         ephemeris_interp_freq='0.01S', downsample_freq='1S'):
    
    sat_info = get_satellite_info(sat_name)
    ephemeris_df = sp3_ephem_to_df(sat_name)
    ephemeris_df = ephemeris_df[(ephemeris_df['UTC'] >= start_date) & (ephemeris_df['UTC'] <= end_date)]

    inverted_accelerations_df = pd.DataFrame(columns=['utc_time', 'inertial_x_acc', 'inertial_y_acc', 'inertial_z_acc'])
   
    settings = {
        'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
        'filter_window_length': window_length, 'filter_polyorder': polyorder,
        'ephemeris_interp_freq': ephemeris_interp_freq, 'downsample_freq': downsample_freq
    }
    
    interp_ephemeris_df = interpolate_positions(ephemeris_df, settings['ephemeris_interp_freq'])
    print(f"columns in interp_ephemeris_df: {interp_ephemeris_df.columns}")
    interp_ephemeris_df = calculate_acceleration(interp_ephemeris_df, 
                                                    settings['ephemeris_interp_freq'],
                                                    settings['filter_window_length'],
                                                    settings['filter_polyorder'])
    print(f"columns in interp_ephemeris_df after calculating acceleration: {interp_ephemeris_df.columns}")
    interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
    interp_ephemeris_df.set_index('UTC', inplace=True)
    interp_ephemeris_df = interp_ephemeris_df.asfreq(settings['downsample_freq'])

    for i in tqdm(range(1, len(interp_ephemeris_df)), desc='Computing Accelerations'):
        epoch = interp_ephemeris_df.index[i]
        vel = np.array([
            interp_ephemeris_df['xv'].iloc[i],
            interp_ephemeris_df['yv'].iloc[i],
            interp_ephemeris_df['zv'].iloc[i]
        ])
        state_vector = np.array([
            interp_ephemeris_df['x'].iloc[i],
            interp_ephemeris_df['y'].iloc[i],
            interp_ephemeris_df['z'].iloc[i],
            vel[0], vel[1], vel[2]
        ])
        force_model_config = {'120x120gravity': True, '3BP': True}
        conservative_accelerations = state2acceleration(state_vector, epoch, settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], **force_model_config)
        computed_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
        observed_acc = np.array([
            interp_ephemeris_df['accx'].iloc[i],
            interp_ephemeris_df['accy'].iloc[i],
            interp_ephemeris_df['accz'].iloc[i]
        ])
        diff = computed_accelerations_sum - observed_acc
        diff_x, diff_y, diff_z = diff[0], diff[1], diff[2]
        new_row = {
            'utc_time': epoch,
            'inverted_x_acc': diff_x,
            'inverted_y_acc': diff_y,
            'inverted_z_acc': diff_z,
            'observed_x_acc': interp_ephemeris_df['accx'].iloc[i],
            'observed_y_acc': interp_ephemeris_df['accy'].iloc[i],
            'observed_z_acc': interp_ephemeris_df['accz'].iloc[i],
            'computed_x_acc': computed_accelerations_sum[0],
            'computed_y_acc': computed_accelerations_sum[1],
            'computed_z_acc': computed_accelerations_sum[2],
            'x': interp_ephemeris_df['x'].iloc[i],
            'y': interp_ephemeris_df['y'].iloc[i],
            'z': interp_ephemeris_df['z'].iloc[i],
            'xv': interp_ephemeris_df['xv'].iloc[i],
            'yv': interp_ephemeris_df['yv'].iloc[i],
            'zv': interp_ephemeris_df['zv'].iloc[i],}
        inverted_accelerations_df = pd.concat([inverted_accelerations_df, pd.DataFrame([new_row])])

    datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    inverted_accelerations_df.to_csv(f"output/DensityInversion/PODBasedAccelerometry/Data/{sat_name}/{datenow}_win{window_length}_poly{polyorder}_inv_accs.csv", index=False)
    return inverted_accelerations_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, coherence

def compute_and_plot_psd_coherence(df, components, window_lengths, fs=1, nperseg=256):
    for window_length in window_lengths:
        fig, axes = plt.subplots(nrows=len(components), ncols=2, figsize=(15, 8), sharex=True)
        min_psd, max_psd, max_coh = float('inf'), float('-inf'), float('-inf')

        # First pass to determine the global min and max for PSD, max for Coherence
        for component in components:
            signal_inverted = df[f'inverted_{component}_acc'].rolling(window=window_length, center=True).mean().dropna()
            signal_inertial = df[f'inertial_{component}_acc']#.rolling(window=window_length, center=True).mean().dropna()

            # PSD and Coherence calculation
            f_inv, Pxx_inv = welch(signal_inverted, fs=fs, nperseg=nperseg)
            f_iner, Pxx_iner = welch(signal_inertial, fs=fs, nperseg=nperseg)
            _, Cxy = coherence(signal_inverted, signal_inertial, fs=fs, nperseg=nperseg)

            min_psd = min(min_psd, np.min(Pxx_inv), np.min(Pxx_iner))
            max_psd = max(max_psd, np.max(Pxx_inv), np.max(Pxx_iner))
            max_coh = max(max_coh, np.max(Cxy))

        # Second pass to plot using consistent axes
        for i, component in enumerate(components):
            signal_inverted = df[f'inverted_{component}_acc'].rolling(window=window_length, center=True).mean().dropna()
            signal_inertial = df[f'inertial_{component}_acc']#.rolling(window=window_length, center=True).mean().dropna()

            # PSD and Coherence calculation again
            f_inv, Pxx_inv = welch(signal_inverted, fs=fs, nperseg=nperseg)
            f_iner, Pxx_iner = welch(signal_inertial, fs=fs, nperseg=nperseg)
            f_coh, Cxy = coherence(signal_inverted, signal_inertial, fs=fs, nperseg=nperseg)

            # Plot PSD
            axes[i, 0].plot(f_inv, Pxx_inv, label=f'Inverted {component}')
            axes[i, 0].plot(f_iner, Pxx_iner, label=f'Inertial {component}')
            axes[i, 0].set_title(f'PSD for {component} Component')
            axes[i, 0].set_xlabel('Frequency (Hz)')
            axes[i, 0].set_ylabel('PSD (m/s^-2 Hz^-0.5)')
            axes[i, 0].legend()
            axes[i, 0].set_yscale('log')
            axes[i, 0].set_xscale('log')
            axes[i, 0].set_ylim([min_psd, max_psd])
            axes[i, 0].grid(which='both', linestyle='--')

            # Plot Coherence
            axes[i, 1].plot(f_coh, Cxy, label=f'Coherence {component}')
            axes[i, 1].set_title(f'Coherence for {component} Component')
            axes[i, 1].set_xlabel('Frequency (Hz)')
            axes[i, 1].set_ylabel('Coherence')
            axes[i, 1].set_xscale('log')
            axes[i, 1].legend()
            axes[i, 1].set_ylim([0, max_coh])
            axes[i, 1].grid(which='both', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/AccelerometerBenchmarking/PSD/PSD_Coherence_Comparison_{window_length}.png")
        plt.close()
def avg_win_length_rms(merged_df):
    #plot the RMS difference between the inverted and measured accelerations as a function of averaging window size
    #used to select the averaging window size that gives the best agreement between the inverted and measured accelerations

    for component in ["h", "c", "l"]:

        #print the first five h values
        print(merged_df[f'inverted_{component}_acc'].head())
        print(merged_df[f'inertial_{component}_acc'].head())
        #make both the values positive
        merged_df[f'inverted_{component}_acc'] = merged_df[f'inverted_{component}_acc'].abs()
        merged_df[f'inertial_{component}_acc'] = merged_df[f'inertial_{component}_acc'].abs()

        rmss = []
        for window in range(5, 256, 5):
            
            #plot the 'window' minute rolling average of the inverted against the measured accelerations
            rolling_inverted = merged_df[f'inverted_{component}_acc'].rolling(window=window, center=True).mean()
            rms_diff = np.sqrt(np.mean((rolling_inverted - merged_df[f'inertial_{component}_acc'])**2))
            rmss.append(rms_diff)
            print(f"RMS: {rms_diff}, window size: {window}")
            plt.plot(merged_df['utc_time'], rolling_inverted, label=f'inverted_{component}_acc')
            plt.plot(merged_df['utc_time'], merged_df[f'inertial_{component}_acc'], label=f'inertial_{component}_acc')
            plt.xlabel('Time')
            plt.ylabel(f'{component} Acceleration (m/s^2)')
            plt.legend()
            #set ylim from 1e-8 to 1e-6
            plt.ylim(1e-8, 1e-5)
            #log scale
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/AccelerometerBenchmarking/HCLAccDiffs/{component}_diff_comparison{window}.png")
            plt.close()

        #plot the RMS difference as a function of window size
        plt.plot(range(5, 256, 5), rmss)
        plt.xlabel('Window Size')
        plt.ylabel('RMS Difference')
        plt.tight_layout()
        plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/AccelerometerBenchmarking/HCLAccDiffs/{component}_diff_RMS_Difference.png")
        plt.close()

if __name__ == '__main__':
    
    acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-05_C_04.txt" # accelerometer data
    quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-05_C_04.txt" # quaternions

    inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)

    # velocity_based_accelerations = compute_acc_from_vel(window_length=21, polyorder=7)
    # if already computed, you can load them from the csv file instead
    velocity_based_accelerations = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-22_15-03-16_fm0_win21_poly7_inv_accs.csv")

    inertial_gfo_data['utc_time'] = pd.to_datetime(inertial_gfo_data['utc_time'])
    velocity_based_accelerations['utc_time'] = pd.to_datetime(velocity_based_accelerations['utc_time'])

    merged_df = pd.merge(inertial_gfo_data, velocity_based_accelerations, on='utc_time', how='inner')
    print(f"columns in merged_df: {merged_df.columns}")
    inverted_x_acc = merged_df['inverted_x_acc']
    inverted_y_acc = merged_df['inverted_y_acc']
    inverted_z_acc = merged_df['inverted_z_acc']
    inertial_x_acc = merged_df['inertial_x_acc_x']
    inertial_y_acc = merged_df['inertial_y_acc_x']
    inertial_z_acc = merged_df['inertial_z_acc_x']
    density_inversion_df = pd.DataFrame(columns=[
        'Epoch', 'Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density',
        'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz'
    ])
    #now convert to HCL components
    for i in range(1, len(merged_df)):
        h_acc_inv, c_diff_inv, l_diff_inv = project_acc_into_HCL(inverted_x_acc[i], inverted_y_acc[i], inverted_z_acc[i], merged_df['x'][i], merged_df['y'][i], merged_df['z'][i], merged_df['xv'][i], merged_df['yv'][i], merged_df['zv'][i])
        h_acc_meas, c_diff_meas, l_diff_meas = project_acc_into_HCL(inertial_x_acc[i], inertial_y_acc[i], inertial_z_acc[i], merged_df['x'][i], merged_df['y'][i], merged_df['z'][i], merged_df['xv'][i], merged_df['yv'][i], merged_df['zv'][i])
        
        r = np.array([merged_df.loc[i, 'x'], merged_df.loc[i, 'y'], merged_df.loc[i, 'z']])
        v = np.array([merged_df.loc[i, 'xv'], merged_df.loc[i, 'yv'], merged_df.loc[i, 'zv']])
        atm_rot = np.array([0, 0, 72.9211e-6])
        v_rel = v - np.cross(atm_rot, r)
        rho = -2 * (l_diff_meas / (2.2 * 1.004)) * (600.2 / np.abs(np.linalg.norm(v_rel))**2)
        time = merged_df.loc[i, 'utc_time']
        jb_08_rho = query_jb08(r, time)
        dtm2000_rho = query_dtm2000(r, time)
        nrlmsise00_rho = query_nrlmsise00(r, time)
        new_row = pd.DataFrame({
            'Epoch': time,
            'Computed Density': rho,
            'JB08 Density': jb_08_rho,
            'DTM2000 Density': dtm2000_rho,
            'NRLMSISE00 Density': nrlmsise00_rho
        })
        merged_df.loc[i, 'inverted_h_acc'] = h_acc_inv
        merged_df.loc[i, 'inverted_c_acc'] = c_diff_inv
        merged_df.loc[i, 'inverted_l_acc'] = l_diff_inv

        merged_df.loc[i, 'inertial_h_acc'] = h_acc_meas
        merged_df.loc[i, 'inertial_c_acc'] = c_diff_meas
        merged_df.loc[i, 'inertial_l_acc'] = l_diff_meas





    # avg_win_length_rms(merged_df)

    # components = ['h', 'c', 'l']
    # window_lengths = range(5, 256, 5)
    # psd_data = compute_and_plot_psd_coherence(merged_df, components, window_lengths)


    #TODO: # Most useful plot to get here is going to tell us how much of the singal we recover. 
            # What kind of bias and what we lose from the detail in the along track.
                # first we plot just the two along track accelerations -> use RMS as a metric for average length?
                # second we plot the PSD of the two along track accelerations
           # Make it so that we can come back with other accelerations and compare those as well.