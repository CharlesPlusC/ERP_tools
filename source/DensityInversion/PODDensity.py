import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

import os
from ..tools.utilities import project_acc_into_HCL, get_satellite_info, interpolate_positions, calculate_acceleration
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from ..tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
import numpy as np
import datetime
from tqdm import tqdm
import pandas as pd
from orekit.pyhelpers import setup_orekit_curdir, datetime_to_absolutedate
from ..tools.GFODataReadTools import get_gfo_inertial_accelerations
from ..tools.SWIndices import get_kp_ap_dst_f107
from .Plotting.PODDerivedDensityPlotting import plot_relative_density_vs_dst_symh,plot_densities_and_indices, plot_densities_and_residuals, model_reldens_sat_megaplot, reldens_sat_megaplot, get_arglat_from_df, plot_density_arglat_diff, plot_relative_density_change, density_compare_scatter

def density_inversion(sat_name, ephemeris_df, x_acc_col, y_acc_col, z_acc_col, force_model_config, nc_accs=False, models_to_query=['JB08'], density_freq='15S'):
    sat_info = get_satellite_info(sat_name)
    settings = {
        'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
        'density_freq': density_freq
    }

    available_models = ['JB08', 'DTM2000', 'NRLMSISE00', None]
    for model in models_to_query:
        assert model in available_models

    assert 'UTC' in ephemeris_df.columns

    ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    ephemeris_df.set_index('UTC', inplace=True)
    ephemeris_df = ephemeris_df.asfreq(settings['density_freq'])

    #drop NA rows
    ephemeris_df.dropna(inplace=True)

    columns = [
        'UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz',
        'nc_accx', 'nc_accy', 'nc_accz', 'Computed Density', *(models_to_query)
    ]

    # Initialize the DataFrame with the specified columns and correct dtypes
    density_inversion_df = pd.DataFrame(columns=columns)
    rows_list = []

    for i in tqdm(range(1, len(ephemeris_df)), desc='Processing Density Inversion'):
        time = ephemeris_df.index[i]
        vel = np.array([ephemeris_df['xv'].iloc[i], ephemeris_df['yv'].iloc[i], ephemeris_df['zv'].iloc[i]])
        state_vector = np.array([ephemeris_df['x'].iloc[i], ephemeris_df['y'].iloc[i], ephemeris_df['z'].iloc[i], vel[0], vel[1], vel[2]])
        if not nc_accs: 
            #except drag acceleration
            all_accelerations = state2acceleration(state_vector, time, 
                                                            settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], 
                                                            **force_model_config)
            all_accelerations_sum = np.sum(list(all_accelerations.values()), axis=0)
            observed_acc = np.array([ephemeris_df['accx'].iloc[i], ephemeris_df['accy'].iloc[i], ephemeris_df['accz'].iloc[i]])

            nc_accelerations = all_accelerations_sum - observed_acc

            nc_accx, nc_accy, nc_accz = nc_accelerations[0], nc_accelerations[1], nc_accelerations[2]
    
        else:
            #now just compute radiation pressure since our observed only contain non conservative (SRP + ERP)
            #assuming a_nonconservative = a_rp + a_drag, where a_rp = a_srp + a_erp
            rp_fm_config = {
            'knocke_erp': True,
            'SRP': True}

            rp_accelerations = state2acceleration(state_vector, time, 
                                                            settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], 
                                                            **rp_fm_config)
            print(f"rp_accelerations: {rp_accelerations}")
            rp_accelerations_sum = np.sum(list(rp_accelerations.values()), axis=0)
            observed_acc = np.array([ephemeris_df[x_acc_col].iloc[i], ephemeris_df[y_acc_col].iloc[i], ephemeris_df[z_acc_col].iloc[i]])

            nc_accelerations = rp_accelerations_sum - observed_acc

            nc_accx, nc_accy, nc_accz = nc_accelerations[0], nc_accelerations[1], nc_accelerations[2]

        _, _, nc_acc_l = project_acc_into_HCL(nc_accx, nc_accy, nc_accz, 
                                            ephemeris_df['x'].iloc[i], ephemeris_df['y'].iloc[i], ephemeris_df['z'].iloc[i],
                                             ephemeris_df['xv'].iloc[i], ephemeris_df['yv'].iloc[i], ephemeris_df['zv'].iloc[i])

        r = np.array([ephemeris_df['x'].iloc[i], ephemeris_df['y'].iloc[i], ephemeris_df['z'].iloc[i]])
        v = np.array([ephemeris_df['xv'].iloc[i], ephemeris_df['yv'].iloc[i], ephemeris_df['zv'].iloc[i]])
        atm_rot = np.array([0, 0, 72.9211e-6])
        v_rel = v - np.cross(atm_rot, r)

        rho = -2 * (nc_acc_l / (settings['cd'] * settings['cross_section'])) * (settings['mass'] /  np.abs(np.linalg.norm(v_rel))**2)

        row_data = {
            'UTC': time, 'x': r[0], 'y': r[1], 'z': r[2], 'xv': v[0], 'yv': v[1], 'zv': v[2],
            'nc_accx': nc_accx, 'nc_accy': nc_accy, 'nc_accz': nc_accz, 'Computed Density': rho
        }

        for model in models_to_query:
            if model is not None and globals().get(f"query_{model.lower()}"):
                model_func = globals()[f"query_{model.lower()}"]
                row_data[model] = model_func(position=r, datetime=time)

        rows_list.append(row_data)

    if rows_list:
        new_rows_df = pd.DataFrame(rows_list)
        density_inversion_df = pd.concat([density_inversion_df, new_rows_df], ignore_index=True)

    return density_inversion_df

if __name__ == "__main__":
    # main()
    # daily_indices, kp3hrly, dst_hrly = get_kp_ap_dst_f107()

    # force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
    # ephemeris_df = sp3_ephem_to_df("GRACE-FO-A", date = '2024-05-11')
    # ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    # ephemeris_df = ephemeris_df[ephemeris_df['UTC'] >= '2024-05-10 15:00:00']
    # #make the ephemeris stop 24hrs after the start time
    # print(f"ephemeris_df: {ephemeris_df.head()}")
    # ephemeris_df = ephemeris_df[ephemeris_df['UTC'] <= '2024-05-12 07:00:00']
    # print(f"ephemeris_df: {ephemeris_df.head()}")
    # #plot the norm of the position vector
    # pos_norm = np.linalg.norm(ephemeris_df[['x', 'y', 'z']].values, axis=1)
    # ephemeris_df['pos_norm'] = pos_norm
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(ephemeris_df['UTC'], ephemeris_df['pos_norm'])
    # ax.set_ylabel('Position Norm (km)')
    # ax.set_xlabel('UTC')
    # ax.set_title('GRACE-FO-A Position Norm')
    # plt.show()


    # interp_ephemeris_df = interpolate_positions(ephemeris_df, '0.01S')
    # velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
    # print(f"velacc_ephem: {velacc_ephem.head()}")
    # vel_acc_col_x, vel_acc_col_y, vel_acc_col_z = 'vel_acc_x', 'vel_acc_y', 'vel_acc_z'
    # density_df = density_inversion("GRACE-FO-A", velacc_ephem, 'vel_acc_x', 'vel_acc_y', 'vel_acc_z', force_model_config, nc_accs=False, 
    #                                 models_to_query=['JB08', 'DTM2000', "NRLMSISE00"], density_freq='15S')
    # TODO:# # Do a more systematic analysis of the effect of the interpolation window length and polynomial order on the RMS error
    # densitydf_gfoa = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-26_01-22-32_GRACE-FO-A_fm12597_density_inversion.csv")
    # densitydf_tsx = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/TerraSAR-X/2024-04-26_06-24-57_TerraSAR-X_fm12597_density_inversion.csv")
    # densitydf_champ = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/CHAMP/2024-04-24_CHAMP_fm0_density_inversion.csv")

    # instead of continuing to manually list the paths just iterate over the list of satellite names in "output/DensityInversion/PODBasedAccelerometry/Data/StormAnalysis/"
    # Base directory for storm analysis
    base_dir = "output/DensityInversion/PODBasedAccelerometry/Data/StormAnalysis/"
    # List of satellite names
    sat_names = ["CHAMP"] #"GRACE-FO-A", "TerraSAR-X"

    for sat_name in sat_names:
        storm_analysis_dir = os.path.join(base_dir, sat_name)
        
        # Check if the directory exists before listing files
        if os.path.exists(storm_analysis_dir):
            for storm_file in os.listdir(storm_analysis_dir):
                # Form the full path to the storm file
                storm_file_path = os.path.join(storm_analysis_dir, storm_file)
                
                # Check if it's actually a file
                if os.path.isfile(storm_file_path):
                    storm_df = pd.read_csv(storm_file_path) 
                    density_compare_scatter(storm_df, 45, sat_name)
                    # plot_relative_density_change([storm_df], 45, sat_name)
                    # plot_density_arglat_diff([storm_df], 45, sat_name)
                    # plot_densities_and_residuals([storm_df], 90, sat_name)
                    # plot_densities_and_indices([storm_df], 90, sat_name)
                    # density_compare_scatter([storm_df], 45, sat_name) 

    # Example Usage
    # base_dir = "output/DensityInversion/PODBasedAccelerometry/Data/StormAnalysis/"
    # sat_names = ["CHAMP"] #"GRACE-FO-A", "TerraSAR-X"
    # for sat_name in sat_names:
        # reldens_sat_megaplot(base_dir, sat_name, moving_avg_minutes=45)
        # plot_relative_density_vs_dst_symh(base_dir, sat_name, moving_avg_minutes=45)
        # model_reldens_sat_megaplot(base_dir, sat_name, moving_avg_minutes=90)

    # sat_names = ["CHAMP","GRACE-FO-A", "TerraSAR-X"] #"GRACE-FO-A", "TerraSAR-X"
    # for sat_name in sat_names:
        # reldens_sat_megaplot(base_dir, sat_name, moving_avg_minutes=45)
        # plot_relative_density_vs_dst_symh(base_dir, sat_name, moving_avg_minutes=90)
        # model_reldens_sat_megaplot(base_dir, sat_name, moving_avg_minutes=90)


    #### Checking storm denstiy
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # import numpy as np

    # storm_g5_df = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/StormAnalysis/GRACE-FO-A/GRACE-FO-A_storm_density_0_1_20240524153130.csv")

    # storm_g5_df['UTC'] = pd.to_datetime(storm_g5_df['UTC'])
    # storm_g5_df.set_index('UTC', inplace=True)

    # start_time = '2024-05-10 23:00:00'
    # end_time = '2024-05-11 11:00:00'

    # mask = (storm_g5_df.index < start_time) | (storm_g5_df.index > end_time)
    # mean_computed_density = storm_g5_df.loc[mask, 'Computed Density'].mean()

    # storm_g5_df.loc[start_time:end_time, 'Computed Density'] = mean_computed_density

    # storm_g5_df[['Computed Density']].plot()
    # plt.xlabel('UTC')
    # plt.ylabel('Density')
    # plt.title('Computed Density over UTC')
    # plt.show()

    # storm_g5_df['norm'] = np.linalg.norm(storm_g5_df[['x', 'y', 'z']].values, axis=1)
    # storm_g5_df['vel_norm'] = np.linalg.norm(storm_g5_df[['xv', 'yv', 'zv']].values, axis=1)
    # storm_g5_df['norm'] = (storm_g5_df['norm']/1000) - 6378.137
    # storm_g5_df['vel_norm'] = storm_g5_df['vel_norm']/1000
    # storm_g5_df['UTC'] = pd.to_datetime(storm_g5_df['UTC'])
    # storm_g5_df.set_index('UTC', inplace=True)

    # fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # # Plot altitude
    # axs[0].plot(storm_g5_df.index, storm_g5_df['norm'], label='Altitude')
    # axs[0].set_ylabel('Altitude (km)')
    # axs[0].axhline(y=500, color='r', linestyle='--')
    # axs[0].axhline(y=1000, color='g', linestyle='--')
    # axs[0].text(storm_g5_df.index[0], 500, '500km', color='r')
    # axs[0].text(storm_g5_df.index[0], 1000, '1000km', color='g')
    # axs[0].grid()
    # axs[0].set_title('GRACE-FO-A Storm Density Altitude')
    # axs[0].set_yscale('log')

    # # Plot velocity
    # axs[1].plot(storm_g5_df.index, storm_g5_df['vel_norm'], label='Velocity', color='b')
    # axs[1].set_xlabel('UTC')
    # axs[1].set_ylabel('Velocity (km/s)')
    # axs[1].grid()
    # axs[1].set_title('GRACE-FO-A Storm Density Velocity')

    # plt.tight_layout()
    # plt.show()
    
    # plot_densities_and_indices([storm_g5_df], 90, "GRACE-FO-A")