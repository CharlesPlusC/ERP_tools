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
import matplotlib

def take_derivative(velocities, times, decimals=6):
    times = np.array(times, dtype='datetime64[ns]')
    delta_t = np.diff(times) / np.timedelta64(1, 's')
    
    velocities = np.array(velocities)
    delta_v = np.diff(velocities, axis=0)
    
    # Calculate accelerations and round to a specified number of decimal places
    accs = np.around(delta_v / delta_t[:, np.newaxis], decimals=decimals)
    return accs


# def main():
#     sat_names_to_test = ["GRACE-FO-A"]
#     rms_results = []

#     for sat_name in sat_names_to_test:
#         ephemeris_df = sp3_ephem_to_df(sat_name)  # Ensure this function returns a properly formatted DataFrame
#         noncons_force_model_config = {
#             '120x120gravity': True,
#             '3BP': True,
#             'solid_tides': True,
#             'ocean_tides': True,
#             'knocke_erp': True,
#             'relativity': True,
#             'SRP': True,
#             'jb08drag': True,
#         }
#         cr = 1.5
#         cd = 3.2
#         cross_section = 1.004
#         mass = 600.0
#         no_points_to_process = 15
#         ephemeris_df = ephemeris_df.head(no_points_to_process)

#         for filter_window_length in range(4, 51):
#             for filter_polyorder in range(1, 20):
#                 print(f"Processing window length {filter_window_length} and polyorder {filter_polyorder}")
#                 if filter_polyorder < filter_window_length:
#                     interp_ephemeris_df = improved_interpolation_and_acceleration(ephemeris_df, '0.01S', filter_window_length, filter_polyorder)
#                     interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
#                     interp_ephemeris_df.set_index('UTC', inplace=True)
#                     interp_ephemeris_df = interp_ephemeris_df.asfreq('30S')

#                     for i in range(1, len(interp_ephemeris_df)):
#                         epoch = interp_ephemeris_df.index[i]
#                         state_vector = np.array([
#                             interp_ephemeris_df['x'].iloc[i], interp_ephemeris_df['y'].iloc[i], interp_ephemeris_df['z'].iloc[i],
#                             interp_ephemeris_df['xv'].iloc[i], interp_ephemeris_df['yv'].iloc[i], interp_ephemeris_df['zv'].iloc[i]
#                         ])
#                         observed_acc = np.array([
#                             interp_ephemeris_df['accx'].iloc[i], interp_ephemeris_df['accy'].iloc[i], interp_ephemeris_df['accz'].iloc[i]
#                         ])
#                         noncons_accelerations = state2acceleration(state_vector, epoch, cr, cd, cross_section, mass, **noncons_force_model_config)

#                         computed_acc_sum = np.sum(list(noncons_accelerations.values()), axis=0)
#                         rms = np.sqrt(np.mean((computed_acc_sum - observed_acc) ** 2))
#                         rms_results.append({
#                             'Window Length': filter_window_length,
#                             'Polyorder': filter_polyorder,
#                             'RMS': rms
#                         })

#     df_rms = pd.DataFrame(rms_results)
#     # Ensure there are no duplicates or use aggregation if needed
#     pivot_table = df_rms.pivot_table(index="Polyorder", columns="Window Length", values="RMS", aggfunc='mean')
#     # Plotting the heatmap with scientific notation for RMS values and a logarithmic color scale
#     plt.figure(figsize=(8, 8))
#     ax = sns.heatmap(pivot_table, annot=True, fmt=".1e", cmap="nipy_spectral", norm=matplotlib.colors.LogNorm(), cbar_kws={'label': 'RMS Error'})
#     plt.title('RMS Error for Different Window Lengths and Polynomial Orders')
#     plt.xlabel('Window Length')
#     plt.ylabel('Polynomial Order')
#     plt.savefig("output/DensityInversion/InterpolationExperiments/RMS_Heatmap.png")

# if __name__ == "__main__":
#     main()

def main():
    sat_names_to_test = ["GRACE-FO-A"]
    # sat_names_to_test = ["TerraSAR-X"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        force_model_config = {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
        # cons_force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'relativity': True}
        cr = 1.5
        cd = 3.2
        cross_section = 1.004
        mass = 600.0
        no_points_to_process = 80
        ephemeris_df = ephemeris_df.head(no_points_to_process)

        interp_ephemeris_df = improved_interpolation_and_acceleration(ephemeris_df,'0.01S', filter_window_length=21, filter_polyorder=7) 
        x_psuedo_acc, y_psuedo_acc, z_psuedo_acc = interp_ephemeris_df['accx'], interp_ephemeris_df['accy'], interp_ephemeris_df['accz']
        interp_ephemeris_df = interp_ephemeris_df.head(len(x_psuedo_acc))

        #add the x_acc, y_acc, z_acc to the dataframe
        interp_ephemeris_df['accx'] = x_psuedo_acc
        interp_ephemeris_df['accy'] = y_psuedo_acc
        interp_ephemeris_df['accz'] = z_psuedo_acc

        interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
        interp_ephemeris_df.set_index('UTC', inplace=True)
        #now resample the ephmeris to 30 seconds to make the computation cheaper
        interp_ephemeris_df = interp_ephemeris_df.asfreq('30S')

        computed_rhos = []
        jb08_rhos = []
        msis_rhos = []
        dtm_rhos = []
        hcl_acc_diffs = []
        a_cs = []
        atm_rot = np.array([0, 0, 72.9211e-6])
        sampling_arc_length = 25 #this is how long we will average the density for
        for i in range(1, len(interp_ephemeris_df)):

            print(f"pts done: {i/len(interp_ephemeris_df)}")
            epoch = interp_ephemeris_df.index[i]
            vel = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
            state_vector = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], vel[0], vel[1], vel[2]])
            conservative_accelerations = state2acceleration(state_vector, epoch, cr, cd, cross_section, mass, **force_model_config)

            computed_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
            print(f"Conservative Accelerations Sum: {computed_accelerations_sum}")
            print(f"Observed Accelerations: {np.array([interp_ephemeris_df['accx'][i], interp_ephemeris_df['accy'][i], interp_ephemeris_df['accz'][i]])}")
            diff = computed_accelerations_sum - np.array([interp_ephemeris_df['accx'][i], interp_ephemeris_df['accy'][i], interp_ephemeris_df['accz'][i]])
            print(f"acc diff: {diff}")
            a_cs.append(computed_accelerations_sum)
            diff_x, diff_y, diff_z = diff[0], diff[1], diff[2]
            diff_h, diff_c, diff_l = project_acc_into_HCL (diff_x, diff_y, diff_z, interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i])
            hcl_acc_diffs.append(np.array([diff_h, diff_c, diff_l]))
            print(f"Diff Acc in HCL: {diff_h, diff_c, diff_l}")
            
            r = np.array((interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i]))
            v = np.array((interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]))
            v_rel = v - np.cross(atm_rot, r)
            unit_v_rel = v_rel / np.linalg.norm(v_rel)

            # diff_dotv = np.dot(diff, unit_v_rel) #component of diff in the direction of v_rel
            print(f"diff in acceleration dot velocity: {diff_l}")
            rho = -2 * (diff_l / (cd * cross_section)) * (mass / np.abs(np.linalg.norm(v_rel))**2)
            time = interp_ephemeris_df.index[i]
            jb_08_rho = query_jb08(r, time)
            dtm2000_rho = query_dtm2000(r, time)
            nrlmsise00_rho = query_nrlmsise00(r, time)
            computed_rhos.append(rho)
            jb08_rhos.append(jb_08_rho)
            msis_rhos.append(nrlmsise00_rho)
            dtm_rhos.append(dtm2000_rho)
            print(f"Computed Density: {rho}")
            print(f"JB08 Density: {jb_08_rho}")
            print(f"DTM2000 Density: {dtm2000_rho}")
            print(f"NRLMSISE00 Density: {nrlmsise00_rho}")

        #percentage of computed densities that are negative
        print(f"Percentage of computed densities that are negative: {np.sum(np.array(computed_rhos) < 0) / len(computed_rhos)}")
        

        #plot the a_aeros (x,y,z) components
        plt.figure()
        hcl_acc_diffs = np.array(hcl_acc_diffs)
        #drop the first and last 10 points
        hcl_acc_diffs = hcl_acc_diffs[10:-10]
        #force all valyes to be positive
        hcl_acc_diffs = np.abs(hcl_acc_diffs)
        plt.plot(hcl_acc_diffs[:, 0], label='H drag acceleration')
        plt.plot(hcl_acc_diffs[:, 1], label='C drag acceleration')
        plt.plot(hcl_acc_diffs[:, 2], label='L drag acceleration')
        plt.legend()
        #log y axis
        plt.yscale('log')
        plt.xlabel('Ephemeris Point')
        plt.ylabel('Observed Non-Conservative Acceleration (m/s^2)')
        plt.grid(True)
        plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/{sat_name}_hcl_drag.png")

        #make computed rhos positive
        abs_computed_rhos = np.abs(computed_rhos)
        rolling_avg_rhos = np.convolve(abs_computed_rhos, np.ones(20), 'valid') / 20
        #make a 20-point moving average
        ### plot the comptued and JB08 densities
        fig, ax = plt.subplots()
        ax.plot(computed_rhos, label='Computed Density')
        ax.plot(rolling_avg_rhos, label='20-point Moving Average')
        ax.plot(jb08_rhos, label='JB08 Density')
        ax.plot(dtm_rhos, label='DTM2000 Density')
        ax.plot( msis_rhos, label='NRLMSISE00 Density')
        ax.set_xlabel('UTC')
        ax.set_ylabel('Density (kg/m^3)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/{sat_name}_density_compare.png")

        #separate graph with only rolling average
        plt.figure()
        rolling_avg_rhos = np.convolve(computed_rhos, np.ones(20), 'valid') / 20
        plt.plot(rolling_avg_rhos, label='20-point Moving Average')
        plt.xlabel('Ephemeris Point')
        plt.ylabel('Density (kg/m^3)')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/{sat_name}_density_rolling_avg.png")


        #subtract the computed acceleration from the observed acceleration to get the aero acceleration
        # v_components_of_o_minus_cs = []
        # computed_rhos = []
        # jb08_rhos = []
        # dtm2000_rhos = []
        # nrlmsise00_rhos = []
        # average_accelerations = []
        # arc_duration = 25 * 60  # 25 minutes in seconds
        # arc_accelerations = []
        # arc_start_index = 0

        #         state_vector = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], v2[0], v2[1], v2[2]])
        #         computed_accelerations_dict = state2acceleration(state_vector, t2, cr, cd, cross_section, mass, **force_model_config)
        #         computed_accelerations_sum = np.sum(list(computed_accelerations_dict.values()), axis=0)
        #         a_aero = computed_accelerations_sum - average_acceleration
        #         r = np.array((interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i]))
        #         v = np.array((interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]))
        #         atm_rot = np.array([0, 0, 72.9211e-6])
        #         v_rel = v - np.cross(atm_rot, r)
        #         unit_v_rel = v_rel / np.linalg.norm(v_rel)
        #         a_aero_dotv = np.dot(a_aero, unit_v_rel)
        #         v_components_of_o_minus_cs.append(a_aero_dotv)
        #         rho = -2 * (a_aero_dotv / (cd * cross_section)) * (mass / np.abs(np.linalg.norm(v_rel))**2)
        #         computed_rhos.append(rho)
        #         jb_08_rho = query_jb08(r, t2)
        #         # dtm2000_rho = query_dtm2000(r, t2)
        #         # nrlmsise00_rho = query_nrlmsise00(r, t2)
        #         jb08_rhos.append(jb_08_rho)
        #         # dtm2000_rhos.append(dtm2000_rho)
        #         # nrlmsise00_rhos.append(nrlmsise00_rho)

        # #plot 
        # utc_for_plotting = interp_ephemeris_df['UTC'][1:len(v_components_of_o_minus_cs) + 1]
        # plt.plot(utc_for_plotting, v_components_of_o_minus_cs)
        # plt.xlabel('Modified Julian Date')
        # plt.ylabel('v_components_of_o_minus_cs (m/s^2)')
        # plt.title(f"{sat_name}: v_components_of_o_minus_cs")
        # plt.grid(True)
        # plt.show()

        # #plot rho and jb08_rho 
        # plt.plot(utc_for_plotting, computed_rhos, label='Computed Density')
        # plt.plot(utc_for_plotting, jb08_rhos, label='JB08 Density')
        # plt.plot(utc_for_plotting, dtm2000_rhos, label='DTM2000 Density')
        # plt.plot(utc_for_plotting, nrlmsise00_rhos, label='NRLMSISE00 Density')
        # plt.xlabel('Modified Julian Date')
        # plt.ylabel('Density (kg/m^3)')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

if __name__ == '__main__':
    main()

#TODO:
# Compare accelerations that result from differentiation of SP3 velocity data to GNV_1B PODAAC accelerometer readings
# Compute accelerations from GNV_1B 5s velocities to see if POD solution resolution affects the accelerations
#Issue could be from the fact the data is every 30sec and I am taking gradient at every 0.01s