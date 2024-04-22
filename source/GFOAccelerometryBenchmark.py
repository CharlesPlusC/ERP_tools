import pandas as pd
import matplotlib.pyplot as plt
import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir
# download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants
from tools.utilities import improved_interpolation_and_acceleration, extract_acceleration, utc_to_mjd, HCL_diff, get_satellite_info, pos_vel_from_orekit_ephem, EME2000_to_ITRF, ecef_to_lla, posvel_to_sma, project_acc_into_HCL
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import state2acceleration
import numpy as np
import datetime
import seaborn as sns
from scipy.special import lpmn
from scipy.integrate import trapz
import matplotlib.dates as mdates
from tqdm import tqdm
from tools.GFODataReadTools import get_gfo_inertial_accelerations

def compute_acc_from_vel(window_length=21, polyorder=7):
    inverted_accelerations_df = pd.DataFrame(columns=['utc_time', 'inertial_x_acc', 'inertial_y_acc', 'inertial_z_acc'])
    sat_name = "GRACE-FO-A"
    sat_info = get_satellite_info(sat_name)
    start_date = datetime.datetime(2023, 5, 5, 0, 0, 0)
    end_date = datetime.datetime(2023, 5, 5, 6, 0, 0)
    ephemeris_df = sp3_ephem_to_df(sat_name)

    force_model_configs = [
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'relativity': True},
        # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True},
        # {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
        ]
    
    for force_model_config_number, force_model_config in enumerate(force_model_configs):

        settings = {
            'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
            'filter_window_length': window_length, 'filter_polyorder': polyorder,
            'ephemeris_interp_freq': '0.01S', 'downsample_freq': '15S'
        }
        
        #select all points in the ephemeris_df that are within the start and end date
        ephemeris_df = ephemeris_df[(ephemeris_df['UTC'] >= start_date) & (ephemeris_df['UTC'] <= end_date)]
        
        interp_ephemeris_df = improved_interpolation_and_acceleration(
            ephemeris_df, settings['ephemeris_interp_freq'],
            filter_window_length=settings['filter_window_length'],
            filter_polyorder=settings['filter_polyorder']
        )
        interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
        interp_ephemeris_df.set_index('UTC', inplace=True)
        interp_ephemeris_df = interp_ephemeris_df.asfreq(settings['downsample_freq'])
        for i in tqdm(range(1, len(interp_ephemeris_df)), desc='Computing Accelerations'):
            epoch = interp_ephemeris_df.index[i]
            vel = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
            state_vector = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], vel[0], vel[1], vel[2]])
            conservative_accelerations = state2acceleration(state_vector, epoch, settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], **force_model_config)
            computed_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
            observed_acc = np.array([interp_ephemeris_df['accx'][i], interp_ephemeris_df['accy'][i], interp_ephemeris_df['accz'][i]])
            diff = computed_accelerations_sum - observed_acc
            diff_x, diff_y, diff_z = diff[0], diff[1], diff[2]
            new_row = {
                'utc_time': epoch,
                'inverted_x_acc': diff_x,
                'inverted_y_acc': diff_y,
                'inverted_z_acc': diff_z,
                'observed_x_acc': interp_ephemeris_df['accx'][i],
                'observed_y_acc': interp_ephemeris_df['accy'][i],
                'observed_z_acc': interp_ephemeris_df['accz'][i],
                'computed_x_acc': computed_accelerations_sum[0],
                'computed_y_acc': computed_accelerations_sum[1],
                'computed_z_acc': computed_accelerations_sum[2],
                'x': interp_ephemeris_df['x'][i],
                'y': interp_ephemeris_df['y'][i],
                'z': interp_ephemeris_df['z'][i],
                'xv': interp_ephemeris_df['xv'][i],
                'yv': interp_ephemeris_df['yv'][i],
                'zv': interp_ephemeris_df['zv'][i],}
            inverted_accelerations_df = pd.concat([inverted_accelerations_df, pd.DataFrame([new_row])])

    datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    inverted_accelerations_df.to_csv(f"output/DensityInversion/PODBasedAccelerometry/Data/{sat_name}/{datenow}_fm{force_model_config_number}_win{window_length}_poly{polyorder}_inv_accs.csv", index=False)
    return inverted_accelerations_df

if __name__ == '__main__':
    acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-05_C_04.txt"
    quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-05_C_04.txt"

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

    #now convert to HCL components
    for i in range(1, len(merged_df)):
        h_acc_inv, c_diff_inv, l_diff_inv = project_acc_into_HCL(inverted_x_acc[i], inverted_y_acc[i], inverted_z_acc[i], merged_df['x'][i], merged_df['y'][i], merged_df['z'][i], merged_df['xv'][i], merged_df['yv'][i], merged_df['zv'][i])
        h_acc_meas, c_diff_meas, l_diff_meas = project_acc_into_HCL(inertial_x_acc[i], inertial_y_acc[i], inertial_z_acc[i], merged_df['x'][i], merged_df['y'][i], merged_df['z'][i], merged_df['xv'][i], merged_df['yv'][i], merged_df['zv'][i])

        merged_df.loc[i, 'inverted_h_acc'] = h_acc_inv
        merged_df.loc[i, 'inverted_c_acc'] = c_diff_inv
        merged_df.loc[i, 'inverted_l_acc'] = l_diff_inv

        merged_df.loc[i, 'inertial_h_acc'] = h_acc_meas
        merged_df.loc[i, 'inertial_c_acc'] = c_diff_meas
        merged_df.loc[i, 'inertial_l_acc'] = l_diff_meas


    #print the first five h values
    print(merged_df['inverted_l_acc'].head())
    print(merged_df['inertial_l_acc'].head())
    #make both the values positive
    merged_df['inverted_l_acc'] = merged_df['inverted_l_acc'].abs()
    merged_df['inertial_l_acc'] = merged_df['inertial_l_acc'].abs()

    rmss = []
    for window in range(5, 256, 5):
        
        #plot the 45 minute rolling average of the inverted against the measured accelerations
        rolling_inverted = merged_df['inverted_l_acc'].rolling(window=window, center=True).mean()
        #calculat the RMS difference between the two
        rms_diff = np.sqrt(np.mean((rolling_inverted - merged_df['inertial_l_acc'])**2))
        rmss.append(rms_diff)
        #drop the first and last 5 values of both
        merged_df = merged_df.iloc[5:-5]
        rolling_inverted = rolling_inverted.iloc[5:-5]
        
        print(f"RMS: {rms_diff}, window size: {window}")
        plt.plot(merged_df['utc_time'], rolling_inverted, label='inverted_l_acc')
        plt.plot(merged_df['utc_time'], merged_df['inertial_l_acc'], label='inertial_l_acc')
        plt.xlabel('Time')
        plt.ylabel('Acceleration (m/s^2)')
        plt.legend()
        #set ylim from 1e-8 to 1e-6
        plt.ylim(1e-8, 1e-6)
        #log scale
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/AccelerometerBenchmarking/l_diff_comparison{window}.png")
        plt.close()

    #plot the RMS difference as a function of window size
    plt.plot(range(5, 256, 5), rmss)
    plt.xlabel('Window Size')
    plt.ylabel('RMS Difference')
    plt.tight_layout()
    plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/AccelerometerBenchmarking/RMS_Difference.png")
    plt.close()


    #TODO: # Most useful plot to get here is going to tell us how much of the singal we recover. 
           # What kind of bias and what we lose from the detail in the along track.
                # first we plot just the two along track accelerations -> use RMS as a metric for average length?
                # second we plot the PSD of the two along track accelerations
           # Make it so that we can come back with other accelerations and compare those as well.


