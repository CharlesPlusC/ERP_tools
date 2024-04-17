import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants

import os
from tools.utilities import improved_interpolation_and_acceleration, extract_acceleration, utc_to_mjd, HCL_diff, get_satellite_info, pos_vel_from_orekit_ephem, EME2000_to_ITRF, ecef_to_lla, posvel_to_sma
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

def take_derivative(velocities, times, decimals=6):
    times = np.array(times, dtype='datetime64[ns]')
    delta_t = np.diff(times) / np.timedelta64(1, 's')
    
    velocities = np.array(velocities)
    delta_v = np.diff(velocities, axis=0)
    
    # Calculate accelerations and round to a specified number of decimal places
    accs = np.around(delta_v / delta_t[:, np.newaxis], decimals=decimals)
    return accs

def main():
    sat_names_to_test = ["GRACE-FO-A"]
    # sat_names_to_test = ["TerraSAR-X"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        # force_model_config = {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
        force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'relativity': True}
        cr = 1.5
        cd = 3.2
        cross_section = 1.004
        mass = 600.0
        no_points_to_process = 180
        ephemeris_df = ephemeris_df.head(no_points_to_process)
        #skip every 2nd point to reduce the number of points to process
        # ephemeris_df = ephemeris_df.iloc[::2]
        #truncate all digits in the ephemeris to have at most 6 digits after the decimal
        # ephemeris_df = ephemeris_df.round(6)
        # Example usage
        interp_ephemeris_df = improved_interpolation_and_acceleration(ephemeris_df,'0.01S')
        print(f"Interpolated Ephemeris: {interp_ephemeris_df}")
        x_psuedo_acc, y_psuedo_acc, z_psuedo_acc = interp_ephemeris_df['accx'], interp_ephemeris_df['accy'], interp_ephemeris_df['accz']

        # #plot the smoothed and unsmoothed pseudo accelerations
        # plt.plot(x_psuedo_acc, label='x pseudo acceleration')
        # plt.plot(y_psuedo_acc, label='y pseudo acceleration')
        # plt.plot(z_psuedo_acc, label='z pseudo acceleration')
        # plt.xlabel('UTC')
        # plt.ylabel('Acceleration (m/s^2)')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
        #slice interp_ephemeris_df to be the same length as the smoothed pseudo accelerations
        interp_ephemeris_df = interp_ephemeris_df.head(len(x_psuedo_acc))

        #add the x_acc, y_acc, z_acc to the dataframe
        interp_ephemeris_df['x_acc'] = x_psuedo_acc
        interp_ephemeris_df['y_acc'] = y_psuedo_acc
        interp_ephemeris_df['z_acc'] = z_psuedo_acc

        interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
        interp_ephemeris_df.set_index('UTC', inplace=True)
        #now resample the ephmeris to 30 seconds to make the computation easier
        interp_ephemeris_df = interp_ephemeris_df.asfreq('30S')

        computed_rhos = []
        jb08_rhos = []
        msis_rhos = []
        dtm_rhos = []
        a_ncs = []
        alongtrack_o_minus_cs = []
        atm_rot = np.array([0, 0, 72.9211e-6])

        for i in range(1, len(interp_ephemeris_df)):
            print(f"pts done: {i/len(interp_ephemeris_df)}")
            epoch = interp_ephemeris_df.index[i]
            vel = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
            state_vector = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], vel[0], vel[1], vel[2]])
            conservative_accelerations = state2acceleration(state_vector, epoch, cr, cd, cross_section, mass, **force_model_config)
            # print(f"Computed Accelerations: {computed_accelerations_dict}")
            computed_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
            print(f"Conservative Accelerations Sum: {computed_accelerations_sum}")
            print(f"Observed Accelerations: {np.array([interp_ephemeris_df['x_acc'][i], interp_ephemeris_df['y_acc'][i], interp_ephemeris_df['z_acc'][i]])}")
            #subtract the computed acceleration from the observed acceleration to get the "aero" acceleration
            a_nc = computed_accelerations_sum - np.array([interp_ephemeris_df['x_acc'][i], interp_ephemeris_df['y_acc'][i], interp_ephemeris_df['z_acc'][i]])
            a_ncs.append(a_nc)
            print(f"Non-Conservative Acc: {a_nc}")
            r = np.array((interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i]))
            v = np.array((interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]))
            v_rel = v - np.cross(atm_rot, r)
            unit_v_rel = v_rel / np.linalg.norm(v_rel)
            a_aero_dotv = np.dot(a_nc, unit_v_rel)
            alongtrack_o_minus_cs.append(np.linalg.norm(a_aero_dotv))
            print(f"areo acceleration dot velocity: {a_aero_dotv}")
            # v_components_of_o_minus_cs.append(a_aero_dotv)
            # rho = -2 * (a_aero_dotv / (cd * cross_section)) * (mass / np.abs(np.linalg.norm(v_rel))**2)
            # time = interp_ephemeris_df.index[i]
            # jb_08_rho = query_jb08(r, time)
            # # dtm2000_rho = query_dtm2000(r, time)
            # # nrlmsise00_rho = query_nrlmsise00(r, time)
            # computed_rhos.append(rho)
            # jb08_rhos.append(jb_08_rho)
            # msis_rhos.append(nrlmsise00_rho)
            # dtm_rhos.append(dtm2000_rho)
            # print(f"Computed Density: {rho}")
            # print(f"JB08 Density: {jb_08_rho}")
            # print(f"DTM2000 Density: {dtm2000_rho}")
            # print(f"NRLMSISE00 Density: {nrlmsise00_rho}")

        #plot the a_aeros (x,y,z) components
        a_ncs = np.array(a_ncs)
        plt.plot(a_ncs[:, 0], label='x aero acceleration')
        plt.plot(a_ncs[:, 1], label='y aero acceleration')
        plt.plot(a_ncs[:, 2], label='z aero acceleration')
        plt.xlabel('UTC')
        plt.ylabel('Acceleration (m/s^2)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # ### plot the comptued and JB08 densities
        # fig, ax = plt.subplots()
        # ax.plot(interp_ephemeris_df.index[1:], computed_rhos, label='Computed Density')
        # ax.plot(interp_ephemeris_df.index[1:], jb08_rhos, label='JB08 Density')
        # # ax.plot(interp_ephemeris_df.index[1:], dtm_rhos, label='DTM2000 Density')
        # # ax.plot(interp_ephemeris_df.index[1:], msis_rhos, label='NRLMSISE00 Density')
        # ax.set_xlabel('UTC')
        # ax.set_ylabel('Density (kg/m^3)')
        # ax.legend()
        # ax.grid(True)
        # plt.show()

        # #plot the differences in the alongtrack component of the acceleration 
        # fig, ax = plt.subplots()
        # ax.plot(interp_ephemeris_df.index[1:], alongtrack_o_minus_cs)


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