import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

# download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants

import os
from tools.utilities import extract_acceleration, utc_to_mjd, HCL_diff, get_satellite_info, pos_vel_from_orekit_ephem, EME2000_to_ITRF, ecef_to_lla, posvel_to_sma, interpolate_ephemeris
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

def estimate_acceleration_from_vels(velocities, times):
    # using arc-to-chord estimation outlined in Calabia et al. 2015
    # acc_dt = r_t1 - r_t-1/ delta_t
    accs = []
    for i in range(1, len(velocities)):
        delta_t = (times[i] - times[i - 1]).total_seconds()
        acc = (velocities[i] - velocities[i - 1]) / delta_t
        accs.append(acc)
    return accs


def main():
    sat_names_to_test = ["GRACE-FO-A"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        force_model_config = {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
        cr = 1.5
        cd = 3.2
        cross_section = 1.004
        mass = 600.0
        no_points_to_process = 180*2

        ephemeris_df = ephemeris_df.head(no_points_to_process)
        interp_ephemeris_df = interpolate_ephemeris(ephemeris_df, ephemeris_df['UTC'].iloc[0], ephemeris_df['UTC'].iloc[-1], freq='0.01S')
        print(f"head of ephemeris_df: {ephemeris_df.head(10)}")
        print(f"head of interp_ephemeris_df: {interp_ephemeris_df.head()}")
        #where they are the same time, print the difference between the two
        for i in range(len(ephemeris_df)):
            if ephemeris_df['UTC'].iloc[i] == interp_ephemeris_df['UTC'].iloc[i]:
                print(f"diff in x: {ephemeris_df['x'].iloc[i] - interp_ephemeris_df['x'].iloc[i]}")
                print(f"diff in y: {ephemeris_df['y'].iloc[i] - interp_ephemeris_df['y'].iloc[i]}")
                print(f"diff in z: {ephemeris_df['z'].iloc[i] - interp_ephemeris_df['z'].iloc[i]}")
                print(f"diff in xv: {ephemeris_df['xv'].iloc[i] - interp_ephemeris_df['xv'].iloc[i]}")
                print(f"diff in yv: {ephemeris_df['yv'].iloc[i] - interp_ephemeris_df['yv'].iloc[i]}")
                print(f"diff in zv: {ephemeris_df['zv'].iloc[i] - interp_ephemeris_df['zv'].iloc[i]}")

        #calculate the pseudo accelerations
        pseudo_accelerations =  estimate_acceleration_from_vels(np.array([interp_ephemeris_df['xv'], interp_ephemeris_df['yv'], interp_ephemeris_df['zv']]).T, interp_ephemeris_df['UTC'])
        x_psuedo_acc = [acc[0] for acc in pseudo_accelerations]
        y_psuedo_acc = [acc[1] for acc in pseudo_accelerations]
        z_psuedo_acc = [acc[2] for acc in pseudo_accelerations]

        # #plot the pseudo accelerations each in their own subplot
        # fig, axs = plt.subplots(3)
        # fig.suptitle(f"{sat_name}: Pseudo Accelerations")
        # axs[0].plot(interp_ephemeris_df['UTC'][:-1], x_psuedo_acc)

        # axs[1].plot(interp_ephemeris_df['UTC'][:-1], y_psuedo_acc)

        # axs[2].plot(interp_ephemeris_df['UTC'][:-1], z_psuedo_acc)

        # plt.show()


        #drop the first el of interp_ephemeris_df to match the length of pseudo_accelerations
        interp_ephemeris_df = interp_ephemeris_df.drop(interp_ephemeris_df.index[0])

        #add the x_acc, y_acc, z_acc to the dataframe
        interp_ephemeris_df['x_acc'] = x_psuedo_acc
        interp_ephemeris_df['y_acc'] = y_psuedo_acc
        interp_ephemeris_df['z_acc'] = z_psuedo_acc

        interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
        interp_ephemeris_df.set_index('UTC', inplace=True)
        #now resample the ephmeris to 30 seconds to make the computation easier
        interp_ephemeris_df = interp_ephemeris_df.resample('30S').mean() #not sure about using the mean here...

        computed_rhos = []
        jb08_rhos = []
        msis_rhos = []
        dtm_rhos = []

        for i in range(1, len(interp_ephemeris_df)):
            print(f"pts done: {i/len(interp_ephemeris_df)}")
            epoch = interp_ephemeris_df.index[i]
            vel = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
            state_vector = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], vel[0], vel[1], vel[2]])
            computed_accelerations_dict = state2acceleration(state_vector, epoch, cr, cd, cross_section, mass, **force_model_config)
            print(f"Computed Accelerations: {computed_accelerations_dict}")
            computed_accelerations_sum = np.sum(list(computed_accelerations_dict.values()), axis=0)
            print(f"Computed Accelerations Sum: {computed_accelerations_sum}")
            print(f"Measured Accelerations: {np.array([interp_ephemeris_df['x_acc'][i], interp_ephemeris_df['y_acc'][i], interp_ephemeris_df['z_acc'][i]])}")
            #subtract the computed acceleration from the observed acceleration to get the "aero" acceleration
            a_aero = computed_accelerations_sum - np.array([interp_ephemeris_df['x_acc'][i], interp_ephemeris_df['y_acc'][i], interp_ephemeris_df['z_acc'][i]])
            print(f"Aero Acceleration: {a_aero}")
            r = np.array((interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i]))
            v = np.array((interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]))
            atm_rot = np.array([0, 0, 72.9211e-6])
            v_rel = v - np.cross(atm_rot, r)
            unit_v_rel = v_rel / np.linalg.norm(v_rel)
            a_aero_dotv = np.dot(a_aero, unit_v_rel)
            # v_components_of_o_minus_cs.append(a_aero_dotv)
            rho = -2 * (a_aero_dotv / (cd * cross_section)) * (mass / np.abs(np.linalg.norm(v_rel))**2)
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

        ### plot the comptued and JB08 densities
        utc_for_plotting = interp_ephemeris_df.index[1:len(computed_rhos) + 1]
        plt.plot(utc_for_plotting, computed_rhos, label='Computed Density')
        plt.plot(utc_for_plotting, jb08_rhos, label='JB08 Density')
        plt.plot(utc_for_plotting, msis_rhos, label='MSIS Density')
        plt.plot(utc_for_plotting, dtm_rhos, label='DTM Density')
        plt.xlabel('UTC')
        #log y axis
        plt.yscale('log')
        plt.ylabel('Density (kg/m^3)')
        plt.legend()
        plt.grid(True)
        plt.show()

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