import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants

import os
from tools.utilities import project_acc_into_HCL, improved_interpolation_and_acceleration, extract_acceleration, utc_to_mjd, HCL_diff, get_satellite_info, pos_vel_from_orekit_ephem, EME2000_to_ITRF, ecef_to_lla, posvel_to_sma
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy.special import lpmn
from scipy.integrate import trapz

import numpy as np
import pandas as pd
import datetime
import os
from tqdm import tqdm

def main():
    sat_names_to_test = ["GRACE-FO-A"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        force_model_config = {
            '120x120gravity': True, '3BP': True, 'solid_tides': True,
            'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True
        }
        settings = {
            'cr': 1.5, 'cd': 3.2, 'cross_section': 1.004, 'mass': 600.0,
            'no_points_to_process': 180*5, 'filter_window_length': 21, 'filter_polyorder': 7,
            'ephemeris_interp_freq': '0.01S', 'density_freq': '15S'
        }
        ephemeris_df = ephemeris_df.head(settings['no_points_to_process'])
        interp_ephemeris_df = improved_interpolation_and_acceleration(
            ephemeris_df, settings['ephemeris_interp_freq'],
            filter_window_length=settings['filter_window_length'],
            filter_polyorder=settings['filter_polyorder']
        )

        interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
        interp_ephemeris_df.set_index('UTC', inplace=True)
        interp_ephemeris_df = interp_ephemeris_df.asfreq(settings['density_freq'])

        density_inversion_df = pd.DataFrame(columns=[
            'Epoch', 'Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density',
            'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz'
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
            diff_h, diff_c, diff_l = project_acc_into_HCL(diff_x, diff_y, diff_z, interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i])
            
            r = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i]])
            v = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
            atm_rot = np.array([0, 0, 72.9211e-6])
            v_rel = v - np.cross(atm_rot, r)
            rho = -2 * (diff_l / (settings['cd'] * settings['cross_section'])) * (settings['mass'] / np.abs(np.linalg.norm(v_rel))**2)
            time = interp_ephemeris_df.index[i]
            jb_08_rho = query_jb08(r, time)
            dtm2000_rho = query_dtm2000(r, time)
            nrlmsise00_rho = query_nrlmsise00(r, time)
            
            new_row = pd.DataFrame({
                'Epoch': [time], 'Computed Density': [rho], 'JB08 Density': [jb_08_rho], 'DTM2000 Density': [dtm2000_rho], 'NRLMSISE00 Density': [nrlmsise00_rho],
                'x': [r[0]], 'y': [r[1]], 'z': [r[2]], 'xv': [v[0]], 'yv': [v[1]], 'zv': [v[2]],
                'accx': [diff_x], 'accy': [diff_y], 'accz': [diff_z]
            })
            density_inversion_df = pd.concat([density_inversion_df, new_row], ignore_index=True)

        save_folder = f"output/DensityInversion/PODBasedAccelerometry/Data/{sat_name}/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        filename = f"{datetime.datetime.now().strftime('%Y-%m-%d')}_{sat_name}_density_inversion.csv"
        density_inversion_df.to_csv(os.path.join(save_folder, filename), index=False)
        return density_inversion_df

def plot_density_data(density_df, moving_avg_minutes):
    # Convert moving average minutes to the number of points based on data frequency
    freq_in_seconds = pd.infer_freq(density_df['Epoch'])
    seconds_per_point = pd.to_timedelta(freq_in_seconds).seconds
    window_size = (moving_avg_minutes * 60) // seconds_per_point

    # Set Epoch as the index for plotting
    density_df.set_index('Epoch', inplace=True)

    # Compute the moving average for Computed Density
    density_df['Computed Density MA'] = density_df['Computed Density'].rolling(window=window_size).mean()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(density_df.index, density_df['Computed Density'], label='Computed Density')
    plt.plot(density_df.index, density_df['JB08 Density'], label='JB08 Density')
    plt.plot(density_df.index, density_df['DTM2000 Density'], label='DTM2000 Density')
    plt.plot(density_df.index, density_df['NRLMSISE00 Density'], label='NRLMSISE00 Density')
    plt.plot(density_df.index, density_df['Computed Density MA'], label='Computed Density Moving Average', linestyle='--')

    plt.title('Density Data Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Save the plot
    save_folder = "output/DensityInversion/PODBasedAccelerometry/Plots"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    filename = f"{datetime.datetime.now().strftime('%Y-%m-%d')}_density_plot.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()

if __name__ == "__main__":
    densitydf = main()
    # densitydf = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-18_GRACE-FO-A_density_inversion.csv")
    # plot_density_data(densitydf, 35)


#TODO:
# Compare accelerations that result from differentiation of SP3 velocity data to GNV_1B PODAAC accelerometer readings
# Compute accelerations from GNV_1B 5s velocities to see if POD solution resolution affects the accelerations -> cant find 5s velocities anywhere??
# Do a more systematic analysis of the effect of the interpolation window length and polynomial order on the RMS error