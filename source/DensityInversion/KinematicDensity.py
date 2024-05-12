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
from .Plotting.PODDerivedDensityPlotting import get_arglat_from_df, plot_density_arglat_diff, plot_density_data, plot_relative_density_change, density_compare_scatter

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
    # daily_indices, kp3hrly, dst_hrly = get_sw_indices()
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
    #TODO:# # Do a more systematic analysis of the effect of the interpolation window length and polynomial order on the RMS error
    # densitydf_gfoa = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-26_01-22-32_GRACE-FO-A_fm12597_density_inversion.csv")
    # densitydf_tsx = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/TerraSAR-X/2024-04-26_06-24-57_TerraSAR-X_fm12597_density_inversion.csv")
    # densitydf_champ = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/CHAMP/2024-04-24_CHAMP_fm0_density_inversion.csv")

    # instead of continuing to manually list the paths just iterate over the list of satellite names in "output/DensityInversion/PODBasedAccelerometry/Data/StormAnalysis/"
    # Base directory for storm analysis
    # base_dir = "output/DensityInversion/PODBasedAccelerometry/Data/StormAnalysis/"

    # # List of satellite names
    # sat_names = ["CHAMP", "GRACE-FO-A", "TerraSAR-X"]

    # for sat_name in sat_names:
    #     # Correctly set the path for the current satellite
    #     storm_analysis_dir = os.path.join(base_dir, sat_name)
        
    #     # Check if the directory exists before listing files
    #     if os.path.exists(storm_analysis_dir):
    #         for storm_file in os.listdir(storm_analysis_dir):
    #             # Form the full path to the storm file
    #             storm_file_path = os.path.join(storm_analysis_dir, storm_file)
                
    #             # Check if it's actually a file
    #             if os.path.isfile(storm_file_path):
    #                 storm_df = pd.read_csv(storm_file_path) 
    #                 plot_relative_density_change([storm_df], 45, sat_name)
    #                 plot_density_arglat_diff([storm_df], 45, sat_name)
    #                 plot_density_data([storm_df], 45, sat_name)
                    # density_compare_scatter([storm_df], 45, sat_name)

#TODO: make megaplot with all 
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    from datetime import datetime, timedelta

    def determine_storm_category(kp_max):
        if kp_max < 5:
            return "Below G1"
        elif kp_max < 6:
            return "G1"
        elif kp_max < 7:
            return "G2"
        elif kp_max < 8:
            return "G3"
        elif kp_max < 9:
            return "G4"
        else:
            return "G5"
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    from datetime import datetime, timedelta
    from matplotlib.dates import DateFormatter

    def plot_computed_density_for_satellite(base_dir, sat_name, moving_avg_minutes=45):
        storm_analysis_dir = os.path.join(base_dir, sat_name)
        if not os.path.exists(storm_analysis_dir):
            print(f"No data directory found for {sat_name}")
            return
        
        _, kp_3hrly, hourly_dst = get_sw_indices()
        kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
        hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

        storm_data = []

        for storm_file in sorted(os.listdir(storm_analysis_dir)):
            storm_file_path = os.path.join(storm_analysis_dir, storm_file)
            if os.path.isfile(storm_file_path):
                df = pd.read_csv(storm_file_path)
                df['UTC'] = pd.to_datetime(df['UTC'], utc=True)
                df.set_index('UTC', inplace=True)
                df.index = df.index.tz_convert('UTC')

                df = get_arglat_from_df(df)

                # Calculate the moving average for density
                window_size = (moving_avg_minutes * 60) // 30
                density_types = ['Computed Density']  # Add other density types if needed
                for density_type in density_types:
                    if density_type in df.columns:
                        df[density_type] = df[density_type].rolling(window=window_size, min_periods=1, center=True).mean()

                start_time = df.index.min()
                end_time = df.index.max()

                kp_filtered = kp_3hrly[(kp_3hrly['DateTime'] >= start_time) & (kp_3hrly['DateTime'] <= end_time)]
                max_kp_value = kp_filtered['Kp'].max() if not kp_filtered.empty else 0

                storm_category = determine_storm_category(max_kp_value)
                storm_number = -int(storm_category[1:]) if storm_category != "Below G1" else 0

                storm_data.append((df, start_time, storm_category, storm_number))

        storm_data.sort(key=lambda x: x[3], reverse=True)

        num_storms = len(storm_data)
        nrows = int(np.ceil(num_storms / 3))
        ncols = 3 if num_storms > 2 else num_storms
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), dpi=100)
        axes = axes.flatten()

        cmap = 'cubehelix'

        all_densities = np.concatenate([df['Computed Density'] for df, _, _, _ in storm_data if 'Computed Density' in df])
        positive_densities = all_densities[all_densities > 0]

        if positive_densities.size == 0:
            raise ValueError("No positive densities found. Check your data.")

        # density_vmin = positive_densities.min()
        # density_vmax = positive_densities.max()
        density_vmin = 5e-13
        density_vmax = 5e-11
        norm = LogNorm(vmin=density_vmin, vmax=density_vmax)

        for i, (df, start_time, storm_category, storm_number) in enumerate(storm_data):
            # Set plot range to 5 days from start_time
            plot_end_time = start_time + timedelta(days=3)
            plot_df = df[(df.index >= start_time) & (df.index <= plot_end_time)]
            
            densities = np.clip(plot_df['Computed Density'], density_vmin, density_vmax)
            sc = axes[i].scatter(plot_df.index, plot_df['arglat'], c=densities, cmap=cmap, alpha=0.6, edgecolor='none', norm=norm)
            axes[i].set_title(f'{start_time.strftime("%Y-%m")}, {storm_category}', fontsize=10)
            axes[i].set_ylabel('')
            axes[i].set_xlabel('')
            
            axes[i].set_xlim([start_time, plot_end_time])
            # axes[i].xaxis.set_major_locator(plt.MaxNLocator(3))
            # axes[i].xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
            #remove the x-axis labels
            axes[i].set_xticklabels([])
            #remove the y-axis labels
            axes[i].set_yticklabels([])
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Computed Density (kg/m³)', rotation=270, labelpad=15)

        plt.savefig(f'{sat_name}_computed_density_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Example Usage
    base_dir = "output/DensityInversion/PODBasedAccelerometry/Data/StormAnalysis/"
    sat_name = "CHAMP"
    plot_computed_density_for_satellite(base_dir, sat_name)