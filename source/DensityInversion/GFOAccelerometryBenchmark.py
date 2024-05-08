import pandas as pd
import matplotlib.pyplot as plt
import orekit
from orekit.pyhelpers import setup_orekit_curdir
# download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants
from ..tools.utilities import interpolate_positions,calculate_acceleration, get_satellite_info, project_acc_into_HCL
from ..tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from .KinematicDensity import ephemeris_to_density
import numpy as np
import datetime
from tqdm import tqdm
from ..tools.GFODataReadTools import get_gfo_inertial_accelerations

# podaac-data-downloader -c GRACEFO_L1B_ASCII_GRAV_JPL_RL04 -d ./GRACE-FO_A_DATA -sd 2023-05-05T00:00:00Z -ed 2023-05-05T23:59:59Z -e ".*" --verbose

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
    acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-05_C_04.txt"
    quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-05_C_04.txt"

    inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)
    inertial_gfo_data['utc_time'] = pd.to_datetime(inertial_gfo_data['utc_time'])
    intertial_t0 = inertial_gfo_data['utc_time'].iloc[0]
    inertial_gfo_data = inertial_gfo_data[(inertial_gfo_data['utc_time'] >= intertial_t0) & (inertial_gfo_data['utc_time'] <= intertial_t0 + pd.Timedelta(hours=2))]

    ephemeris_df = sp3_ephem_to_df(satellite="GRACE-FO-A", date="2023-05-05")
    print(f"start date: {ephemeris_df['UTC'].iloc[0]}, end date: {ephemeris_df['UTC'].iloc[-1]}")
    
    ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    ephemeris_df = ephemeris_df[(ephemeris_df['UTC'] >= intertial_t0) & (ephemeris_df['UTC'] <= intertial_t0 + pd.Timedelta(hours=2))]
    print(f"new start date: {ephemeris_df['UTC'].iloc[0]}, new end date: {ephemeris_df['UTC'].iloc[-1]}")
    
    velocity_based_accelerations = ephemeris_to_density(sat_name="GRACE-FO-A", ephemeris_df = ephemeris_df, 
                                                        force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},
                                                        savgol_poly=7, savgol_window=21)
    
    print(f"columns in velocity_based_accelerations: {velocity_based_accelerations.columns}")
    
    velocity_based_accelerations['utc_time'] = pd.to_datetime(velocity_based_accelerations['UTC'])

    inertial_gfo_data['utc_time'] = pd.to_datetime(inertial_gfo_data['utc_time'])

    merged_df = pd.merge(inertial_gfo_data, velocity_based_accelerations, on='utc_time', how='inner')
    print(f"columns in merged_df: {merged_df.columns}")
    inverted_x_acc = merged_df['accx']
    inverted_y_acc = merged_df['accy']
    inverted_z_acc = merged_df['accz']
    inertial_x_acc = merged_df['inertial_x_acc']
    inertial_y_acc = merged_df['inertial_y_acc']
    inertial_z_acc = merged_df['inertial_z_acc']

    merged_df['inverted_h_acc'] = 0
    merged_df['inverted_c_acc'] = 0
    merged_df['inverted_l_acc'] = 0
    merged_df['inertial_h_acc'] = 0
    merged_df['inertial_c_acc'] = 0
    merged_df['inertial_l_acc'] = 0
    merged_df["ACT_rho"] = 0
    for i in range(1, len(merged_df)):
        h_acc_inv, c_acc_inv, l_acc_inv = project_acc_into_HCL(inverted_x_acc[i], inverted_y_acc[i], inverted_z_acc[i], merged_df['x'][i], merged_df['y'][i], merged_df['z'][i], merged_df['xv'][i], merged_df['yv'][i], merged_df['zv'][i])
        h_acc_meas, c_acc_meas, l_acc_meas = project_acc_into_HCL(inertial_x_acc[i], inertial_y_acc[i], inertial_z_acc[i], merged_df['x'][i], merged_df['y'][i], merged_df['z'][i], merged_df['xv'][i], merged_df['yv'][i], merged_df['zv'][i])
        merged_df.at[i, 'inverted_l_acc'] = l_acc_inv
        merged_df.at[i, 'inertial_l_acc'] = l_acc_meas
        atm_rot = np.array([0, 0, 72.9211e-6])
        r = np.array([merged_df['x'][i], merged_df['y'][i], merged_df['z'][i]])
        v = np.array([merged_df['xv'][i], merged_df['yv'][i], merged_df['zv'][i]])
        v_rel = v - np.cross(atm_rot, r)        
        merged_df.at[i, 'ACT_rho'] = -2 * (l_acc_meas / (2.2 * 1.004)) * (600.2 / np.abs(np.linalg.norm(v_rel))**2)

    #calculte the 45-point rolling average of the inverted and measured accelerations and add them to a new column called 'rolling_inverted_acc' and 'rolling_savgol_acc'
    merged_df[f'rolling_inverted_l_acc'] = merged_df[f'inverted_l_acc'].rolling(window=45, center=True).mean()

    #drop the frist and last 90 points
    merged_df = merged_df[90:-90] 
    plt.figure(figsize=(15, 6))
    plt.plot(merged_df['utc_time'], merged_df['inverted_l_acc'], label='Inverted L Acceleration')
    plt.plot(merged_df['utc_time'], merged_df['rolling_inverted_l_acc'], label='Rolling Inverted-for L Acceleration')
    plt.plot(merged_df['utc_time'], merged_df['inertial_l_acc'], label='Measured L Acceleration')
    plt.ylabel('along track Acceleration (m/s^2)')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/compare_alongtrackacc{timenow}.png")
    # plt.show()
    plt.close()
    #calculate the rolling_Computed_Density
    merged_df['rolling_Computed_Density'] = merged_df['Computed Density'].rolling(window=45, center=True).mean()
    #now plot the ACT_rho and the Computed Density and rolling_Computed Density, and the JB08 Density

    plt.figure(figsize=(15, 6))
    plt.plot(merged_df['utc_time'], merged_df['ACT_rho'], label='ACT_rho')
    plt.plot(merged_df['utc_time'], merged_df['Computed Density'], label='Computed Density')
    plt.plot(merged_df['utc_time'], merged_df['rolling_Computed_Density'], label='Rolling Computed Density')
    plt.plot(merged_df['utc_time'], merged_df['JB08 Density'], label='JB08 Density')
    plt.ylabel('Density (kg/m^3)')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/compare_density{timenow}.png")
    # plt.show()
    plt.close()

    # components = ['h', 'c', 'l']
    # window_lengths = range(5, 256, 5)
    # psd_data = compute_and_plot_psd_coherence(merged_df, components, window_lengths)


    #TODO: # Most useful plot to get here is going to tell us how much of the singal we recover. 
            # What kind of bias and what we lose from the detail in the along track.
                # first we plot just the two along track accelerations -> use RMS as a metric for average length?
                # second we plot the PSD of the two along track accelerations
           # Make it so that we can come back with other accelerations and compare those as well.