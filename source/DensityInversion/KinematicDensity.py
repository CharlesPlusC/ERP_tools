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
from ..tools.SWIndices import get_sw_indices
from .Plotting.PODDerivedDensityPlotting import plot_density_arglat_diff, plot_density_data, plot_relative_density_change

def density_inversion(sat_name, interp_ephemeris_df, force_model_config, accelerometer_data=None):
    sat_info = get_satellite_info(sat_name)
    settings = {
        'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
        'density_freq': '15S'
    }

    # Convert UTC column to datetime and set it as index
    interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
    interp_ephemeris_df.set_index('UTC', inplace=True)
    interp_ephemeris_df = interp_ephemeris_df.asfreq(settings['density_freq'])

    # Handle accelerometer data
    if accelerometer_data is not None:
        accelerometer_data['UTC'] = pd.to_datetime(accelerometer_data['UTC'])
        accelerometer_data.set_index('UTC', inplace=True)
        accelerometer_data = accelerometer_data.asfreq(settings['density_freq'])
        interp_ephemeris_df = pd.merge(interp_ephemeris_df, accelerometer_data, how='left', left_index=True, right_index=True, suffixes=('', '_drop'))

        # Drop the duplicate columns from the accelerometer data
        columns_to_drop = [col for col in interp_ephemeris_df.columns if '_drop' in col]
        interp_ephemeris_df.drop(columns=columns_to_drop, inplace=True)

    columns = [
        'Epoch', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz',
        *(['JB08 Density'])
        # *(['JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density'])
    ]
    if accelerometer_data is not None:
        columns.append('Accelerometer Density')

    density_inversion_df = pd.DataFrame(columns=columns)

    for i in tqdm(range(1, len(interp_ephemeris_df)), desc='Processing Density Inversion'):
        epoch = interp_ephemeris_df.index[i]
        vel = np.array([interp_ephemeris_df['xv'].iloc[i], interp_ephemeris_df['yv'].iloc[i], interp_ephemeris_df['zv'].iloc[i]])
        state_vector = np.array([interp_ephemeris_df['x'].iloc[i], interp_ephemeris_df['y'].iloc[i], interp_ephemeris_df['z'].iloc[i], vel[0], vel[1], vel[2]])
        if accelerometer_data is not None:
            force_model_config = {'3BP': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},
        conservative_accelerations = state2acceleration(state_vector, epoch, settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], **force_model_config)

        computed_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
        if accelerometer_data is not None:
            observed_acc = np.array([interp_ephemeris_df['inertial_x_acc'].iloc[i], interp_ephemeris_df['inertial_y_acc'].iloc[i], interp_ephemeris_df['inertial_z_acc'].iloc[i]])
        else:
            observed_acc = np.array([interp_ephemeris_df['accx'].iloc[i], interp_ephemeris_df['accy'].iloc[i], interp_ephemeris_df['accz'].iloc[i]])

        diff = computed_accelerations_sum - observed_acc
        diff_x, diff_y, diff_z = diff[0], diff[1], diff[2]
        _, _, diff_l = project_acc_into_HCL(diff_x, diff_y, diff_z, interp_ephemeris_df['x'].iloc[i], interp_ephemeris_df['y'].iloc[i], interp_ephemeris_df['z'].iloc[i], interp_ephemeris_df['xv'].iloc[i], interp_ephemeris_df['yv'].iloc[i], interp_ephemeris_df['zv'].iloc[i])

        r = np.array([interp_ephemeris_df['x'].iloc[i], interp_ephemeris_df['y'].iloc[i], interp_ephemeris_df['z'].iloc[i]])
        v = np.array([interp_ephemeris_df['xv'].iloc[i], interp_ephemeris_df['yv'].iloc[i], interp_ephemeris_df['zv'].iloc[i]])
        atm_rot = np.array([0, 0, 72.9211e-6])
        v_rel = v - np.cross(atm_rot, r)
        rho = -2 * (diff_l / (settings['cd'] * settings['cross_section'])) * (settings['mass'] / np.abs(np.linalg.norm(v_rel))**2)
        time = epoch

        if accelerometer_data is None:
            row_data = {
                'Epoch': time, 'x': r[0], 'y': r[1], 'z': r[2], 'xv': v[0], 'yv': v[1], 'zv': v[2],
                'accx': diff_x, 'accy': diff_y, 'accz': diff_z, 'Computed Density': rho,
                **({key: value for key, value in zip(['JB08 Density'], 
                                                     [query_jb08(r, time)])})
                # **({key: value for key, value in zip(['JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density'], 
                                                        # [query_jb08(r, time), query_dtm2000(r, time), query_nrlmsise00(r, time)])})

                        }
            
        if accelerometer_data is not None:
            row_data = {
                'Epoch': time, 'x': r[0], 'y': r[1], 'z': r[2], 'xv': v[0], 'yv': v[1], 'zv': v[2],
                'accx': diff_x, 'accy': diff_y, 'accz': diff_z, 'Accelerometer Density': rho, 'Computed Density': interp_ephemeris_df['Computed Density'][i],
                #TODO: somehow the accelerometer density is inverted... 
                #take the exisitng jb08, dtm2000, and nrlmsise00 densities from the interp_ephemeris_df 
                **({key: interp_ephemeris_df[key][i] for key in ['JB08 Density']})
                # **({key: interp_ephemeris_df[key][i] for key in ['JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']})
            }

        new_row = pd.DataFrame(row_data, index=[0])
        density_inversion_df = pd.concat([density_inversion_df, new_row], ignore_index=True)

    return density_inversion_df

def main():
    sat_names_to_test = ["GRACE-FO-A"]
    force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        ephemeris_df = ephemeris_df.head(180*35)
        ephemeris_to_density(sat_name, ephemeris_df, force_model_config)

def ephemeris_to_density(sat_name, ephemeris_df, force_model_config, path_output_folder="output/DensityInversion/PODBasedAccelerometry/Data/"):
    interp_ephemeris_df = interpolate_positions(ephemeris_df, '0.01S')
    interp_ephemeris_df = calculate_acceleration(interp_ephemeris_df, '0.01S', 21, 7)
    density_inversion_dfs = density_inversion(sat_name, interp_ephemeris_df, force_model_config, accelerometer_data=None)
    # interp_ephemeris_df = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-26_01-22-32_GRACE-FO-A_fm12597_density_inversion.csv")

    #TODO: integrate the accelerometer data for GFO-A
    # acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-05_C_04.txt"
    # quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-05_C_04.txt"
    # inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)
    # ephemeris_df_copy = interp_ephemeris_df.copy()

    # # Ensure utc_time in both DataFrames is converted to datetime
    # ephemeris_df_copy['Epoch'] = pd.to_datetime(ephemeris_df_copy['Epoch'])
    # #rename Epoch to UTC
    # ephemeris_df_copy.rename(columns={'Epoch': 'UTC'}, inplace=True)
    # inertial_gfo_data.rename(columns={'utc_time': 'UTC'}, inplace=True)
    # inertial_gfo_data['UTC'] = pd.to_datetime(inertial_gfo_data['UTC'])
    # merged_df = pd.merge(inertial_gfo_data, ephemeris_df_copy, on='UTC')


    if path_output_folder is not None:
        file_out = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{sat_name}_density_inversion.csv"
        pd.DataFrame(density_inversion_dfs).to_csv(f"{path_output_folder}/{sat_name}/{file_out}")
    return pd.DataFrame(density_inversion_dfs)

if __name__ == "__main__":
    # main()
    # daily_indices, kp3hrly, dst_hrly = get_sw_indices()

    #TODO:# # Do a more systematic analysis of the effect of the interpolation window length and polynomial order on the RMS error
    densitydf_gfoa = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-26_01-22-32_GRACE-FO-A_fm12597_density_inversion.csv")
    densitydf_tsx = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/TerraSAR-X/2024-04-26_06-24-57_TerraSAR-X_fm12597_density_inversion.csv")
    densitydf_champ = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/CHAMP/2024-04-24_CHAMP_fm0_density_inversion.csv")
    # # #read in the x,y,z,xv,yv,zv, and UTC from the densitydf_df
    sat_names = ["GRACE-FO-A", "TerraSAR-X", "CHAMP"]
    for df_num, density_df in enumerate([densitydf_gfoa, densitydf_tsx, densitydf_champ]):
        density_dfs = [density_df]
        #SELECT THE SAT NAME IN USING THE NU
        sat_name = sat_names[df_num]
        print(f"sat_name: {sat_name}")
        # plot_density_arglat_diff(density_dfs, 45, sat_name)
        # plot_density_data(density_dfs, 45, sat_name)
        plot_relative_density_change(density_dfs, 45, sat_name)