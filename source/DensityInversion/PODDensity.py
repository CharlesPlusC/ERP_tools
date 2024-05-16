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
from .Plotting.PODDerivedDensityPlotting import model_reldens_sat_megaplot, reldens_sat_megaplot, get_arglat_from_df, plot_density_arglat_diff, plot_density_data, plot_relative_density_change, density_compare_scatter

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
            conservative_accelerations = state2acceleration(state_vector, time, 
                                                            settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], 
                                                            **force_model_config)
            conservative_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
            observed_acc = np.array([ephemeris_df['accx'].iloc[i], ephemeris_df['accy'].iloc[i], ephemeris_df['accz'].iloc[i]])

            nc_accelerations = conservative_accelerations_sum - observed_acc

            nc_accx, nc_accy, nc_accz = nc_accelerations[0], nc_accelerations[1], nc_accelerations[2]
        else:
            nc_accx, nc_accy, nc_accz = ephemeris_df[x_acc_col].iloc[i], ephemeris_df[y_acc_col].iloc[i], ephemeris_df[z_acc_col].iloc[i]

        nc_acc_h, nc_acc_c, nc_acc_l = project_acc_into_HCL(nc_accx, nc_accy, nc_accz, 
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
    # for df_num, density_df in enumerate([densitydf_gfoa, densitydf_tsx, densitydf_champ]):
    # force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
    # ephemeris_df = sp3_ephem_to_df("GRACE-FO-A", date = '2023-05-04')
    # ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    # ephemeris_df = ephemeris_df[ephemeris_df['UTC'] >= '2023-05-05 18:00:00']
    # #make the ephemeris stop 24hrs after the start time
    # print(f"ephemeris_df: {ephemeris_df.head()}")
    # ephemeris_df = ephemeris_df[ephemeris_df['UTC'] <= '2023-05-06 18:00:00']
    # print(f"ephemeris_df: {ephemeris_df.head()}")
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
    sat_names = ["GRACE-FO-A", "TerraSAR-X", "CHAMP"]

    for sat_name in sat_names:
        # Correctly set the path for the current satellite
        storm_analysis_dir = os.path.join(base_dir, sat_name)
        
        # Check if the directory exists before listing files
        if os.path.exists(storm_analysis_dir):
            for storm_file in os.listdir(storm_analysis_dir):
                # Form the full path to the storm file
                storm_file_path = os.path.join(storm_analysis_dir, storm_file)
                
                # Check if it's actually a file
                if os.path.isfile(storm_file_path):
                    storm_df = pd.read_csv(storm_file_path) 
                    # plot_relative_density_change([storm_df], 45, sat_name)
                    # plot_density_arglat_diff([storm_df], 45, sat_name)
                    plot_density_data([storm_df], 90, sat_name)
                    # density_compare_scatter([storm_df], 45, sat_name)

    # Example Usage
    # base_dir = "output/DensityInversion/PODBasedAccelerometry/Data/StormAnalysis/"
    # sat_names = ["CHAMP", "GRACE-FO-A", "TerraSAR-X"]
    # for sat_name in sat_names:
    #     # reldens_sat_megaplot(base_dir, sat_name)
    #     model_reldens_sat_megaplot(base_dir, sat_name)