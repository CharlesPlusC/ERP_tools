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
from .KinematicDensity import density_inversion
import numpy as np
import datetime
from tqdm import tqdm
from ..tools.GFODataReadTools import get_gfo_inertial_accelerations

# podaac-data-downloader -c GRACEFO_L1B_ASCII_GRAV_JPL_RL04 -d ./GRACE-FO_A_DATA -sd 2023-05-05T00:00:00Z -ed 2023-05-05T23:59:59Z -e ".*" --verbose


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
    sat_name = "GRACE-FO-A"
    force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
    max_time = 12
    acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-05_C_04.txt"
    quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-05_C_04.txt"
    inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)
    inertial_gfo_data['UTC'] = pd.to_datetime(inertial_gfo_data['utc_time'])
    inertial_gfo_data.drop(columns=['utc_time'], inplace=True)

    intertial_t0 = inertial_gfo_data['UTC'].iloc[0]
    inertial_act_gfo_data = inertial_gfo_data[(inertial_gfo_data['UTC'] >= intertial_t0) & (inertial_gfo_data['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))]
    sp3_ephemeris_df = sp3_ephem_to_df(satellite="GRACE-FO-A", date="2023-05-05")
    sp3_ephemeris_df['UTC'] = pd.to_datetime(sp3_ephemeris_df['UTC'])
    sp3_ephemeris_df = sp3_ephemeris_df[(sp3_ephemeris_df['UTC'] >= intertial_t0) & (sp3_ephemeris_df['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))]

    inertial_act_gfo_ephem = pd.merge(inertial_act_gfo_data, sp3_ephemeris_df, on='UTC', how='inner')
    print(f"head of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.head()}")
    print(f"columns of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.columns}")

    act_x_acc_col, act_y_acc_col, act_z_acc_col = 'inertial_x_acc', 'inertial_y_acc', 'inertial_z_acc'
    rho_from_ACT = density_inversion(sat_name, inertial_act_gfo_ephem, 
                                     act_x_acc_col, act_y_acc_col, act_z_acc_col, 
                                     force_model_config, nc_accs=True, 
                                     models_to_query=['JB08'], density_freq='15S')
    rho_from_ACT.rename(columns={'Computed Density': 'ACT_Computed Density'}, inplace=True)

    print(f"head of rho_from_ACT: {rho_from_ACT.head()}")

    interp_ephemeris_df = interpolate_positions(sp3_ephemeris_df, '0.01S')
    sp3_velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
    sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z = 'vel_acc_x', 'vel_acc_y', 'vel_acc_z'
    rho_from_vel = density_inversion(sat_name, sp3_velacc_ephem, 
                                    sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z, 
                                    force_model_config=force_model_config, nc_accs=False, 
                                    models_to_query=[None], density_freq='15S')
    rho_from_vel.rename(columns={'Computed Density': 'Velocity_Computed Density'}, inplace=True)

    print(f"head of rho_from_vel: {rho_from_vel.head()}")

    merged_df = pd.merge(rho_from_ACT[['UTC', 'ACT_Computed Density', 'JB08']], rho_from_vel[['UTC', 'Velocity_Computed Density']], on='UTC', how='inner')

    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    merged_df.dropna(inplace=True)

    print(f"head of merged_df: {merged_df.head()}")
    print(f"columns of merged_df: {merged_df.columns}")
    merged_df.to_csv(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/{timenow}_bench.csv", index=False)

    #load the data from path 
    # path = "output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/2024-05-09_10-06-48_bench.csv"
    # merged_df = pd.read_csv(path)
    merged_df = merged_df.iloc[10:-10]
    median_ACT = merged_df['ACT_Computed Density'].median()
    flipped_ACT = 2 * median_ACT - merged_df['ACT_Computed Density'] #TODO: figure out why the accelerometer density is inverted? Is the X/along track sign different in the ACT file?
    rolling_av45 = merged_df['Velocity_Computed Density'].rolling(window=90, center=True).mean()
    plt.plot(merged_df['UTC'], flipped_ACT, label='ACT Computed Density')
    # plt.plot(merged_df['UTC'], merged_df['Velocity_Computed Density'], label='Velocity Computed Density')
    plt.plot(merged_df['UTC'], rolling_av45, label='Velocity Computed  (MA)', linewidth=2)
    plt.plot(merged_df['UTC'], merged_df['JB08'], label='JB08 Density', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Density (kg/m^3)')
    #log scale
    #only display every hour on the x-axis (the points come every 30s)
    plt.xticks(merged_df['UTC'][::120], rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/{timenow}_bench.png")
    plt.show()
    plt.close()

    # components = ['h', 'c', 'l']
    # window_lengths = range(5, 256, 5)
    # psd_data = compute_and_plot_psd_coherence(merged_df, components, window_lengths)


    #TODO: # Most useful plot to get here is going to tell us how much of the singal we recover. 
            # What kind of bias and what we lose from the detail in the along track.
                # first we plot just the two along track accelerations -> use RMS as a metric for average length?
                # second we plot the PSD of the two along track accelerations
           # Make it so that we can come back with other accelerations and compare those as well.