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
from tools.utilities import improved_interpolation_and_acceleration, extract_acceleration, utc_to_mjd, HCL_diff, get_satellite_info, pos_vel_from_orekit_ephem, EME2000_to_ITRF, ecef_to_lla, posvel_to_sma
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

def get_velocity_based_accelerations():
    inverted_accelerations_df = pd.DataFrame(columns=['utc_time', 'inertial_x_acc', 'inertial_y_acc', 'inertial_z_acc'])
    sat_name = "GRACE-FO-A"
    sat_info = get_satellite_info(sat_name)
    start_date = datetime.datetime(2023, 5, 5, 0, 0, 0)
    end_date = datetime.datetime(2023, 5, 5, 0, 1, 0)
    ephemeris_df = sp3_ephem_to_df(sat_name)
    force_model_configs = [
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'relativity': True},
        # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True},
        # {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
        ]
    
    for force_model_config_number, force_model_config in enumerate(force_model_configs):

        settings = {
            'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
            'filter_window_length': 21, 'filter_polyorder': 7,
            'ephemeris_interp_freq': '0.01S', 'downsample_freq': '1S'
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
            
            computed_acc = np.array([computed_accelerations_sum[0], computed_accelerations_sum[1], computed_accelerations_sum[2]])
            observed_acc = np.array([interp_ephemeris_df['accx'][i], interp_ephemeris_df['accy'][i], interp_ephemeris_df['accz'][i]])
            diff = computed_accelerations_sum - observed_acc
            diff_x, diff_y, diff_z = diff[0], diff[1], diff[2]
            new_row = {
                'utc_time': epoch,
                'inverted_x_acc': diff_x,
                'inverted_y_acc': diff_y,
                'inverted_z_acc': diff_z
            }
            inverted_accelerations_df = pd.concat([inverted_accelerations_df, pd.DataFrame([new_row])])
            #save csv file in output/DensityInversion/PODBasedAccelerometry/Data/{spacecraft_name}/date{date}_fm{force_model_config_number}_inverted_accelerations.csv
            datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            inverted_accelerations_df.to_csv(f"output/DensityInversion/PODBasedAccelerometry/Data/{sat_name}/date{datenow}_fm{force_model_config_number}_inverted_accelerations.csv", index=False)
    return inverted_accelerations_df

def compare_measured_and_inverted_accelerations():
    acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-05_C_04.txt"
    quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-05_C_04.txt"

    inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)
    velocity_based_accelerations = get_velocity_based_accelerations()

    inertial_gfo_data['utc_time'] = pd.to_datetime(inertial_gfo_data['utc_time'])
    velocity_based_accelerations['utc_time'] = pd.to_datetime(velocity_based_accelerations['utc_time'])

    merged_df = pd.merge(inertial_gfo_data, velocity_based_accelerations, on='utc_time', how='inner')

    # Drop the columns that are all NaN
    merged_df = merged_df.dropna(axis=1, how='all')

    print(merged_df.head())

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    ax[0].plot(merged_df['utc_time'], merged_df['inertial_x_acc_x'], label='Measured Acceleration')
    ax[0].plot(merged_df['utc_time'], merged_df['inverted_x_acc'], label='Inverted Acceleration')
    ax[0].set_ylabel('Acceleration (m/s^2)')
    ax[0].set_title('Acceleration in X-axis')
    ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].plot(merged_df['utc_time'], merged_df['inertial_y_acc_x'], label='Measured Acceleration')
    ax[1].plot(merged_df['utc_time'], merged_df['inverted_y_acc'], label='Inverted Acceleration')
    ax[1].set_ylabel('Acceleration (m/s^2)')
    ax[1].set_title('Acceleration in Y-axis')
    ax[1].set_yscale('log')
    ax[1].legend()

    ax[2].plot(merged_df['utc_time'], merged_df['inertial_z_acc_x'], label='Measured Acceleration')
    ax[2].plot(merged_df['utc_time'], merged_df['inverted_z_acc'], label='Inverted Acceleration')
    ax[2].set_ylabel('Acceleration (m/s^2)')
    ax[2].set_title('Acceleration in Z-axis')
    ax[2].set_yscale('log')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    compare_measured_and_inverted_accelerations()
