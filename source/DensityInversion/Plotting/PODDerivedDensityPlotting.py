import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm
from pandas.tseries import offsets
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import PVCoordinates
from orekit.pyhelpers import datetime_to_absolutedate
from source.tools.utilities import project_acc_into_HCL, pv_to_kep, interpolate_positions, calculate_acceleration
from source.tools.SWIndices import get_kp_ap_dst_f107, read_ae, read_sym
from org.orekit.frames import FramesFactory
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
from scipy.stats import shapiro

def get_arglat_from_df(densitydf_df):
    frame = FramesFactory.getEME2000()
    use_column = 'UTC' in densitydf_df.columns

    for index, row in densitydf_df.iterrows():
        x = row['x']
        y = row['y']
        z = row['z']
        xv = row['xv']
        yv = row['yv']
        zv = row['zv']
        
        utc = row['UTC'] if use_column else index

        position = Vector3D(float(x), float(y), float(z))
        velocity = Vector3D(float(xv), float(yv), float(zv))
        pvCoordinates = PVCoordinates(position, velocity)
        time = datetime_to_absolutedate(utc)
        kep_els = pv_to_kep(pvCoordinates, frame, time)
        arglat = kep_els[3] + kep_els[5]
        densitydf_df.at[index, 'arglat'] = arglat

    return densitydf_df

def plot_relative_density_change(data_frames, moving_avg_minutes, sat_name):
    sns.set_style("darkgrid", {
        'axes.facecolor': '#2d2d2d', 'axes.edgecolor': 'white',
        'axes.labelcolor': 'white', 'xtick.color': 'white',
        'ytick.color': 'white', 'figure.facecolor': '#2d2d2d', 'text.color': 'white'
    })

    for density_df in data_frames:
        if 'UTC' in density_df.columns:
            density_df['UTC'] = pd.to_datetime(density_df['UTC'], utc=True)
            density_df.set_index('UTC', inplace=True)

    first_time = data_frames[0].index[0]
    thirty_hours = first_time + pd.Timedelta(hours=30)
    for density_df in data_frames:
        density_df = density_df[(density_df.index >= first_time) & (density_df.index <= thirty_hours)]

    density_types = ['Computed Density', 'JB08', 'DTM2000', 'NRLMSISE00']
    titles = ['Rate of Change: Computed vs JB08', 'Rate of Change: Computed vs DTM2000', 'Rate of Change: Computed vs NRLMSISE00']

    fig, axes = plt.subplots(nrows=len(titles), ncols=2, figsize=(10, 3 * len(titles)), dpi=200, constrained_layout=True)

    daily_indices, kp_3hrly, hourly_dst = get_kp_ap_dst_f107()

    for density_df in data_frames:
        density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True) if 'Epoch' in density_df.columns else density_df.index
        first_epoch = density_df['Epoch'].iloc[0]
        density_df = get_arglat_from_df(density_df)
        density_df.set_index('Epoch', inplace=True)

        if 'Computed Density' in density_df.columns:
            window_size = (moving_avg_minutes * 60) // 30
            density_df['Computed Density'] = density_df['Computed Density'].rolling(window=window_size, min_periods=1, center=True).mean()
            median_density = density_df['Computed Density'].median()
            density_df['Computed Density'] = density_df['Computed Density'].apply(lambda x: median_density if x > 20 * median_density or x < median_density / 20 else x)
            density_df['Computed Density'] = savgol_filter(density_df['Computed Density'], 51, 3)
        density_df = density_df.iloc[450:-450]

        for density_type in density_types:
            if density_type in density_df.columns:
                density_df[f'{density_type} Rate of Change'] = density_df[density_type].diff()

        density_df.index = density_df.index.tz_localize(None)
        daily_indices = daily_indices[(daily_indices['Date'] >= density_df.index[0]) & (daily_indices['Date'] <= density_df.index[-1] + offsets.Hour())]
        kp_3hrly = kp_3hrly[(kp_3hrly['DateTime'] >= density_df.index[0]) & (kp_3hrly['DateTime'] <= density_df.index[-1] + offsets.Hour())]
        hourly_dst = hourly_dst[(hourly_dst['DateTime'] >= density_df.index[0]) & (hourly_dst['DateTime'] <= density_df.index[-1] + offsets.Hour())]
        hourly_dst = hourly_dst.sort_values('DateTime')
        kp_3hrly = kp_3hrly.sort_values('DateTime')

        for j, title in enumerate(titles):
            model_density = density_types[j + 1]
            if f'{model_density} Rate of Change' in density_df.columns:
                density_df[f'Relative Change Rate {model_density}'] = (density_df['Computed Density Rate of Change'] - density_df[f'{model_density} Rate of Change']) / density_df['Computed Density']
                
                median_rcr = density_df[f'Relative Change Rate {model_density}'].median()
                iqr_rcr = density_df[f'Relative Change Rate {model_density}'].quantile(0.75) - density_df[f'Relative Change Rate {model_density}'].quantile(0.25)
                lower_bound_rcr = median_rcr - 3 * iqr_rcr
                upper_bound_rcr = median_rcr + 3 * iqr_rcr
                density_df[f'Relative Change Rate {model_density}'] = density_df[f'Relative Change Rate {model_density}'].clip(lower_bound_rcr, upper_bound_rcr)

                sc = axes[j, 0].scatter(density_df.index, density_df['arglat'], c=density_df[f'Relative Change Rate {model_density}'], cmap='coolwarm', alpha=0.6, edgecolor='none')
                axes[j, 0].set_title(title, fontsize=12)
                axes[j, 0].set_xlabel('Time (UTC)')
                for label in axes[j, 0].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                axes[j, 0].set_ylabel('Argument of Latitude')
                cbar = fig.colorbar(sc, ax=axes[j, 0], aspect=10)
                cbar.set_label('Relative rate of change', rotation=270, labelpad=15)

                axes_secondary = axes[j, 0].twinx()
                axes_secondary.plot(hourly_dst['DateTime'], hourly_dst['Value'], label='Dst Index', c='xkcd:purple', linewidth=1)
                axes_secondary.set_ylabel('Dst Index', color='xkcd:purple')
                axes_secondary.tick_params(axis='y', colors='xkcd:purple')  
                axes_secondary.yaxis.label.set_color('xkcd:purple')  

                axes_tertiary = axes[j, 0].twinx()
                axes_tertiary.plot(kp_3hrly['DateTime'], kp_3hrly['Kp'], label='Kp Index', c='xkcd:bright pink', linewidth=1)
                axes_tertiary.set_ylabel('Kp Index', color='xkcd:bright pink')
                axes_tertiary.tick_params(axis='y', colors='xkcd:bright pink')  
                axes_tertiary.yaxis.label.set_color('xkcd:bright pink')  
                axes_tertiary.spines['right'].set_position(('outward', 40))  

                density_df[f'Relative Density {model_density}'] = density_df['Computed Density'] / density_df[model_density]
                
                median_rd = density_df[f'Relative Density {model_density}'].median()
                iqr_rd = density_df[f'Relative Density {model_density}'].quantile(0.75) - density_df[f'Relative Density {model_density}'].quantile(0.25)
                lower_bound_rd = median_rd - 3 * iqr_rd
                upper_bound_rd = median_rd + 3 * iqr_rd
                density_df[f'Relative Density {model_density}'] = density_df[f'Relative Density {model_density}'].clip(lower_bound_rd, upper_bound_rd)

                sc2 = axes[j, 1].scatter(density_df.index, density_df['arglat'], c=density_df[f'Relative Density {model_density}'], cmap='nipy_spectral', alpha=0.6, edgecolor='none')
                axes[j, 1].set_title(f'Ratio: Computed vs {model_density}', fontsize=12)
                axes[j, 1].set_xlabel('Time (UTC)')
                for label in axes[j, 1].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                axes[j, 1].set_ylabel('Argument of Latitude')
                cbar2 = fig.colorbar(sc2, ax=axes[j, 1], aspect=10)
                cbar2.set_label('Density ratio', rotation=270, labelpad=15) 

    start_time = pd.to_datetime(min(density_df.index))
    day, month, year = start_time.day, start_time.month, start_time.year

    plt.suptitle(f'Relative Change and Ratio in Atmospheric Density for {sat_name}\n {day}/{month}/{year}', color='white') 
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/rel_densitydiff_{first_epoch}.jpg', dpi=600)

def plot_density_arglat_diff(data_frames, moving_avg_minutes, sat_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.colors import LogNorm
    from datetime import datetime, timedelta

    sns.set_style("darkgrid", {
        'axes.facecolor': '#2d2d2d', 'axes.edgecolor': 'white', 
        'axes.labelcolor': 'white', 'xtick.color': 'white', 
        'ytick.color': 'white', 'figure.facecolor': '#2d2d2d', 'text.color': 'white'
    })
    
    density_types = ['Computed Density', 'JB08', 'DTM2000', 'NRLMSISE00']
    titles = ['Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']
    density_diff_titles = ['Computed - JB08', 'Computed - DTM2000', 'Computed - NRLMSISE00']

    nrows = len(density_types)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(9, 2 * nrows), dpi=600)

    _, kp_3hrly, hourly_dst = get_kp_ap_dst_f107()
    
    kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
    hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

    for i in range(len(data_frames)):
        if 'UTC' in data_frames[i].columns:
            data_frames[i]['UTC'] = pd.to_datetime(data_frames[i]['UTC'], utc=True)
            data_frames[i].set_index('UTC', inplace=True)
        data_frames[i].index = data_frames[i].index.tz_localize('UTC') if data_frames[i].index.tz is None else data_frames[i].index.tz_convert('UTC')

    start_time = pd.to_datetime(min(df.index.min() for df in data_frames))
    end_time = pd.to_datetime(max(df.index.max() for df in data_frames))

    kp_3hrly = kp_3hrly[(kp_3hrly['DateTime'] >= start_time) & (kp_3hrly['DateTime'] <= end_time)]

    kp_3hrly = kp_3hrly.sort_values(by='DateTime')
    hourly_dst = hourly_dst.sort_values(by='DateTime')

    max_kp_time = kp_3hrly.loc[kp_3hrly['Kp'].idxmax(), 'DateTime']

    analysis_start_time = max_kp_time - timedelta(hours=24)
    analysis_end_time = max_kp_time + timedelta(hours=36)

    kp_3hrly_analysis = kp_3hrly[(kp_3hrly['DateTime'] >= analysis_start_time) & (kp_3hrly['DateTime'] <= analysis_end_time)]
    hourly_dst_analysis = hourly_dst[(hourly_dst['DateTime'] >= analysis_start_time) & (hourly_dst['DateTime'] <= analysis_end_time)]

    for i, density_df in enumerate(data_frames):
        density_df = density_df[(density_df.index >= analysis_start_time) & (density_df.index <= analysis_end_time)].copy()

        window_size = (moving_avg_minutes * 60) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds if moving_avg_minutes > 0 else 1
        shift_periods = (moving_avg_minutes * 30) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds if moving_avg_minutes > 0 else 0

        vmin, vmax = float('inf'), float('-inf')
        diff_vmax_abs = 0
        for density_type in density_types:
            if density_type in density_df.columns:
                density_df = get_arglat_from_df(density_df)
                
                if density_type == 'Computed Density':
                    median_density = density_df[density_type].median()
                    IQR = density_df[density_type].quantile(0.75) - density_df[density_type].quantile(0.25)
                    lower_bound = median_density - 5 * IQR
                    upper_bound = median_density + 5 * IQR
                    density_df.loc[:, 'Computed Density'] = density_df[density_type].apply(lambda x: median_density if x < lower_bound or x > upper_bound else x)
                    smoothed_values = density_df[density_type].rolling(window=window_size, min_periods=1, center=True).mean()
                    density_df.loc[:, 'Computed Density'] = smoothed_values

                if density_type != 'Computed Density':
                    density_df.loc[:, f'{density_type} Difference'] = density_df['Computed Density'] - density_df[density_type]
                    diff_vmax_abs = max(diff_vmax_abs, abs(density_df[f'{density_type} Difference'].max()), abs(density_df[f'{density_type} Difference'].min()))
                    vmax = max(vmax, density_df[density_type].max())
                    vmin = min(vmin, density_df[density_type].min())

        for j, density_type in enumerate(density_types):
            if f'{density_type}' in density_df.columns:
                sc = axes[j, 0].scatter(density_df.index, density_df['arglat'], c=density_df[density_type], cmap='cubehelix', alpha=0.6, edgecolor='none', norm=LogNorm(vmin=vmin, vmax=vmax))
                axes[j, 0].set_title(titles[j], fontsize=12)
                axes[j, 0].set_ylabel('Arg. Lat.')
                axes[j, 0].set_yticks([-180, 0, 180])
                cbar = fig.colorbar(sc, ax=axes[j, 0])
                cbar.set_label('Density (kg/m³)', rotation=270, labelpad=15)
                if j != nrows - 1:
                    axes[j, 0].set_xticklabels([])
                else:
                    axes[j, 0].set_xlabel('Time (UTC)')
                for label in axes[j, 0].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')

            if density_type != 'Computed Density' and f'{density_type} Difference' in density_df.columns:
                sc_diff = axes[j, 1].scatter(density_df.index, density_df['arglat'], c=density_df[f'{density_type} Difference'], cmap='seismic', alpha=0.6, edgecolor='none', vmin=-diff_vmax_abs, vmax=diff_vmax_abs)
                axes[j, 1].set_title(density_diff_titles[j - 1], fontsize=12)
                axes[j, 1].set_ylabel('Arg. Lat.')
                axes[j, 1].set_yticks([-180, 0, 180])
                cbar_diff = fig.colorbar(sc_diff, ax=axes[j, 1])
                cbar_diff.set_label('Δ Density (kg/m³)', rotation=270, labelpad=15)
                if j != nrows - 1:
                    axes[j, 1].set_xticklabels([])
                else:
                    axes[j, 1].set_xlabel('Time (UTC)')
                for label in axes[j, 1].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')

    ax_right_top = axes[0, 1]
    ax_kp = ax_right_top.twinx()

    ax_right_top.plot(hourly_dst_analysis['DateTime'], hourly_dst_analysis['Value'], label='Dst (nT)', linewidth=2, c = 'xkcd:violet')
    ax_kp.plot(kp_3hrly_analysis['DateTime'], kp_3hrly_analysis['Kp'], label='Kp', linewidth=2, c = 'xkcd:hot pink')
    plt.setp(ax_right_top.get_xticklabels(), visible=False)
    ax_right_top.set_ylabel('Dst (nT)', color='xkcd:violet')
    ax_right_top.yaxis.label.set_color('xkcd:violet')
    ax_right_top.set_ylim(50, -300)
    ax_right_top.tick_params(axis='y', colors='xkcd:violet')

    ax_kp.set_ylabel('Kp', color='xkcd:hot pink')
    ax_kp.yaxis.label.set_color('xkcd:hot pink')
    ax_kp.set_ylim(0, 9)
    ax_kp.tick_params(axis='y', colors='xkcd:hot pink')
    ax_kp.set_yticks(np.arange(0, 10, 3))

    max_kp_value = kp_3hrly_analysis['Kp'].max()
    storm_category = "Below G1" if max_kp_value < 5 else "G1" if max_kp_value < 6 else "G2" if max_kp_value < 7 else "G3" if max_kp_value < 8 else "G4" if max_kp_value < 9 else "G5"

    day, month, year = analysis_start_time.day, analysis_start_time.month, analysis_start_time.year
    plt.suptitle(f'Atmospheric Density as Function of Argument of Latitude for {sat_name} - {storm_category} Storm\n{day}/{month}/{year}', color='white')
    plt.tight_layout()
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/SWI_densitydiff_arglat{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', dpi=600)
    plt.close()
    
def plot_densities_and_residuals(data_frames, moving_avg_minutes, sat_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.signal import savgol_filter
    from datetime import datetime, timedelta

    sns.set_style(style="whitegrid")

    custom_palette = ["#FF6347", "#3CB371", "#1E90FF"]  # Tomato, MediumSeaGreen, DodgerBlue

    for density_df in data_frames:
        if 'UTC' in density_df.columns:
            density_df['UTC'] = pd.to_datetime(density_df['UTC'], utc=True)
            density_df.set_index('UTC', inplace=True)
        if density_df.index.tz is None:
            density_df.index = density_df.index.tz_localize('UTC')
        else:
            density_df.index = density_df.index.tz_convert('UTC')
    
    start_time = min(df.index.min() for df in data_frames)
    end_time = max(df.index.max() for df in data_frames)
    max_kp_time = start_time + (end_time - start_time) / 2  # Replace with actual max Kp time if available
    analysis_start_time = max_kp_time - timedelta(hours=24)
    analysis_end_time = max_kp_time + timedelta(hours=36)

    for i, density_df in enumerate(data_frames):
        seconds_per_point = 30
        window_size = (moving_avg_minutes * 60) // seconds_per_point
        density_df['Computed Density'] = density_df['Computed Density'].rolling(window=window_size, center=True).mean()
        median_density = density_df['Computed Density'].median()
        density_df['Computed Density'] = density_df['Computed Density'].apply(lambda x: median_density if x > 20 * median_density or x < median_density / 20 else x)
        density_df['Computed Density'] = savgol_filter(density_df['Computed Density'], 51, 3)
        density_df = density_df[(density_df.index >= analysis_start_time) & (density_df.index <= analysis_end_time)]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})

    sns.lineplot(ax=axs[0], data=data_frames[0], x=data_frames[0].index, y='JB08', label='JB08 Density', color=custom_palette[0], linewidth=1)
    sns.lineplot(ax=axs[0], data=data_frames[0], x=data_frames[0].index, y='DTM2000', label='DTM2000 Density', color=custom_palette[1], linewidth=1)
    sns.lineplot(ax=axs[0], data=data_frames[0], x=data_frames[0].index, y='NRLMSISE00', label='NRLMSISE00 Density', color=custom_palette[2], linewidth=1)

    for i, density_df in enumerate(data_frames):
        sns.lineplot(ax=axs[0], data=density_df, x=density_df.index, y='Computed Density', label='Computed Density', linestyle='--', color="xkcd:hot pink", linewidth=1)

    day, month, year = analysis_start_time.day, analysis_start_time.month, analysis_start_time.year
    axs[0].set_title(f'Model vs. Estimated: {sat_name} \n{day}-{month}-{year}', fontsize=12)
    axs[0].set_xlabel('Time (UTC)', fontsize=12)
    axs[0].set_ylabel('Density (log scale)', fontsize=12)
    axs[0].legend(loc='upper right', frameon=True)
    axs[0].set_yscale('log')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].set_xlim(analysis_start_time, analysis_end_time)

    residuals = []
    model_names = ['JB08', 'DTM2000', 'NRLMSISE00']

    for model_name in model_names:
        residual = data_frames[0][model_name] - data_frames[0]['Computed Density']
        residuals.append(residual)

    max_residual = max([residual.dropna().max() for residual in residuals])
    bins = np.linspace(-max_residual, max_residual, 50)

    sns.histplot(residuals[0].dropna(), bins=bins, color=custom_palette[0], edgecolor='black', alpha=0.5, label=model_names[0], ax=axs[1])
    sns.histplot(residuals[1].dropna(), bins=bins, color=custom_palette[1], edgecolor='black', alpha=0.5, label=model_names[1], ax=axs[1])
    sns.histplot(residuals[2].dropna(), bins=bins, color=custom_palette[2], edgecolor='black', alpha=0.5, label=model_names[2], ax=axs[1])

    axs[1].set_xlim(-max_residual, max_residual)
    axs[1].set_yscale('log')
    axs[1].set_title('Model Densities - Computed')
    axs[1].set_xlabel('Residuals (kg/m³)')
    axs[1].set_ylabel('Frequency (Log)')
    axs[1].legend(loc='upper right', frameon=True)
    axs[1].grid(True, linestyle='--', linewidth=0.5)
    axs[1].set_xlim(analysis_start_time, analysis_end_time)

    plt.tight_layout()
    datenow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/tseries_hist_residuals_{day}_{month}_{year}.png', dpi=600)
    plt.close()

def plot_densities_and_indices(data_frames, moving_avg_minutes, sat_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.signal import savgol_filter
    from datetime import datetime, timedelta

    sns.set_style(style="whitegrid")

    custom_palette = ["#FF6347", "#3CB371", "#1E90FF"]  # Tomato, MediumSeaGreen, DodgerBlue

    _, kp_3hrly, hourly_dst = get_kp_ap_dst_f107()
    
    kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
    hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

    for density_df in data_frames:
        if 'UTC' in density_df.columns:
            density_df['UTC'] = pd.to_datetime(density_df['UTC'], utc=True)
            density_df.set_index('UTC', inplace=True)
        if density_df.index.tz is None:
            density_df.index = density_df.index.tz_localize('UTC')
        else:
            density_df.index = density_df.index.tz_convert('UTC')
    
    start_time = min(df.index.min() for df in data_frames)
    end_time = max(df.index.max() for df in data_frames)

    kp_3hrly = kp_3hrly[(kp_3hrly['DateTime'] >= start_time) & (kp_3hrly['DateTime'] <= end_time)]
    kp_3hrly = kp_3hrly.sort_values(by='DateTime')
    hourly_dst = hourly_dst.sort_values(by='DateTime')
    max_kp_time = kp_3hrly.loc[kp_3hrly['Kp'].idxmax(), 'DateTime']
    analysis_start_time = max_kp_time - timedelta(hours=24)
    analysis_end_time = max_kp_time + timedelta(hours=36)
    kp_3hrly_analysis = kp_3hrly[(kp_3hrly['DateTime'] >= analysis_start_time) & (kp_3hrly['DateTime'] <= analysis_end_time)]
    hourly_dst_analysis = hourly_dst[(hourly_dst['DateTime'] >= analysis_start_time) & (hourly_dst['DateTime'] <= analysis_end_time)]

    start_date_str = (analysis_start_time - timedelta(days=1)).strftime('%Y-%m-%d')
    end_date_str = (analysis_end_time + timedelta(days=1)).strftime('%Y-%m-%d')
    ae = read_ae(start_date_str, end_date_str)
    sym = read_sym(start_date_str, end_date_str)

    if ae is not None:
        ae['Datetime'] = pd.to_datetime(ae['Datetime'], utc=True)
        ae = ae[(ae['Datetime'] >= analysis_start_time) & (ae['Datetime'] <= analysis_end_time)]
    if sym is not None:
        sym['Datetime'] = pd.to_datetime(sym['Datetime'], utc=True)
        sym = sym[(sym['Datetime'] >= analysis_start_time) & (sym['Datetime'] <= analysis_end_time)]

    for i, density_df in enumerate(data_frames):
        seconds_per_point = 30
        window_size = (moving_avg_minutes * 60) // seconds_per_point
        density_df['Computed Density'] = density_df['Computed Density'].rolling(window=window_size, center=True).mean()
        median_density = density_df['Computed Density'].median()
        density_df['Computed Density'] = density_df['Computed Density'].apply(lambda x: median_density if x > 20 * median_density or x < median_density / 20 else x)
        density_df['Computed Density'] = savgol_filter(density_df['Computed Density'], 51, 3)
        density_df = density_df[(density_df.index >= analysis_start_time) & (density_df.index <= analysis_end_time)]

    nrows = 3 + (1 if sym is not None else 0)  # One row for densities, one for Kp and Dst, one for AE, and optionally one for SYM
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1, 1] + ([1] if sym is not None else []), 'hspace': 0.4})

    # Plot Densities
    sns.lineplot(ax=axs[0], data=data_frames[0], x=data_frames[0].index, y='JB08', label='JB08 Density', color=custom_palette[0], linewidth=1)
    sns.lineplot(ax=axs[0], data=data_frames[0], x=data_frames[0].index, y='DTM2000', label='DTM2000 Density', color=custom_palette[1], linewidth=1)
    sns.lineplot(ax=axs[0], data=data_frames[0], x=data_frames[0].index, y='NRLMSISE00', label='NRLMSISE00 Density', color=custom_palette[2], linewidth=1)

    for i, density_df in enumerate(data_frames):
        sns.lineplot(ax=axs[0], data=density_df, x=density_df.index, y='Computed Density', label='Computed Density', linestyle='--', color="xkcd:hot pink", linewidth=1)

    day, month, year = analysis_start_time.day, analysis_start_time.month, analysis_start_time.year
    axs[0].set_title(f'Model vs. Estimated: {sat_name} \n{day}-{month}-{year}', fontsize=12)
    axs[0].set_xlabel('Time (UTC)', fontsize=12)
    axs[0].set_ylabel('Density (log scale)', fontsize=12)
    axs[0].legend(loc='upper right', frameon=True)
    axs[0].set_yscale('log')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].set_xlim(analysis_start_time, analysis_end_time)

    # Plot Kp and Dst
    ax_right_top = axs[1]
    ax_kp = ax_right_top.twinx()

    ax_right_top.plot(hourly_dst_analysis['DateTime'], hourly_dst_analysis['Value'], label='Dst (nT)', linewidth=2, c='xkcd:violet')
    ax_kp.plot(kp_3hrly_analysis['DateTime'], kp_3hrly_analysis['Kp'], label='Kp', linewidth=2, c='xkcd:hot pink')
    ax_right_top.set_ylabel('Dst (nT)', color='xkcd:violet')
    ax_right_top.yaxis.label.set_color('xkcd:violet')
    ax_right_top.set_ylim(50, -300)
    ax_right_top.tick_params(axis='y', colors='xkcd:violet')

    ax_kp.set_ylabel('Kp', color='xkcd:hot pink')
    ax_kp.yaxis.label.set_color('xkcd:hot pink')
    ax_kp.set_ylim(0, 9)
    ax_kp.tick_params(axis='y', colors='xkcd:hot pink')
    ax_kp.set_yticks(np.arange(0, 10, 3))
    ax_right_top.set_xlim(analysis_start_time, analysis_end_time)

    # Plot AE Index
    if ae is not None:
        sns.lineplot(ax=axs[2], data=ae, x='Datetime', y='minute_value', label='AE Index', color='xkcd:orange', linewidth=1)
        axs[2].set_xlim(analysis_start_time, analysis_end_time)
        axs[2].set_title('AE Index')
        axs[2].set_xlabel('Time (UTC)')
        axs[2].set_ylabel('AE (nT)')
        axs[2].grid(True, linestyle='--', linewidth=0.5)

    # Plot SYM Index if available
    if sym is not None:
        sns.lineplot(ax=axs[3], data=sym, x='Datetime', y='minute_value', label='SYM Index', color='xkcd:violet', linewidth=1)
        axs[3].set_xlim(analysis_start_time, analysis_end_time)
        axs[3].set_title('SYM Index')
        axs[3].set_xlabel('Time (UTC)')
        axs[3].set_ylabel('SYM (nT)')
        axs[3].grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout() 
    datenow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/tseries_indices_{day}_{month}_{year}.png', dpi=600)
    plt.close()
    
def density_compare_scatter(density_df, moving_avg_window, sat_name):
    
    save_path = f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert moving average minutes to the number of points based on data frequency
    if not isinstance(density_df.index, pd.DatetimeIndex):
        density_df['UTC'] = pd.to_datetime(density_df['UTC'], utc=True)
        density_df.set_index('UTC', inplace=True)
    
    # Calculate moving average for the Computed Density
    freq_in_seconds = pd.to_timedelta(pd.infer_freq(density_df.index)).seconds
    window_size = (moving_avg_window * 60) // freq_in_seconds
    
    # Compute the moving average for Computed Density
    density_df['Computed Density MA'] = density_df['Computed Density'].rolling(window=window_size, center=False).mean()

    # Calculate the number of points to shift equivalent to half of the moving average window
    shift_periods = int((moving_avg_window / 2 * 60) // freq_in_seconds)

    # Shift the moving average back by the calculated periods
    density_df['Computed Density MA'] = density_df['Computed Density MA'].shift(-shift_periods)

    # Model names to compare
    model_names = ['JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']

    for model in model_names:
        plot_data = density_df.dropna(subset=['Computed Density MA', model])
        plot_data = plot_data[plot_data['Computed Density MA'] > 0]  # Ensure positive values for log scale
        
        f, ax = plt.subplots(figsize=(6, 6))

        # Draw a combo histogram and scatterplot with density contours
        sns.scatterplot(x=plot_data[model], y=plot_data['Computed Density MA'], s=5, color=".15", ax=ax)
        sns.histplot(x=plot_data[model], y=plot_data['Computed Density MA'], bins=50, pthresh=.1, cmap="rocket", cbar=True, ax=ax)
        sns.kdeplot(x=plot_data[model], y=plot_data['Computed Density MA'], levels=4, color="xkcd:white", linewidths=1, ax=ax)
        #log the x and y 
        ax.set_xscale('log')
        ax.set_yscale('log')
        #add a line of y=x
        ax.plot([1e-13, 1e-11], [1e-13, 1e-11], color='black', linestyle='--')
        #constrain the axes to be between 1e-13 and 1e-11 and of same length
        ax.set_xlim(1e-13, 3e-12)
        ax.set_ylim(1e-13, 3e-12)
        ax.set_title(f'Comparison of {model} vs. Computed Density')
        ax.set_xlabel('Model Density')
        ax.set_ylabel('Computed Density')
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        plot_filename = f'comparison_{model.replace(" ", "_")}.png'
        plt.savefig(os.path.join(save_path, plot_filename))
        plt.close()

        # Line plot of density over time for both the model and the computed density
        plt.figure(figsize=(11, 7))
        plt.plot(plot_data.index, plot_data['Computed Density MA'], label='Computed Density')
        plt.plot(plot_data.index, plot_data[model], label=model)
        plt.title(f'{model} vs. Computed Density Over Time')
        plt.xlabel('Epoch (UTC)')
        plt.ylabel('Density')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plot_filename = f'comparison_{model.replace(" ", "_")}_time.png'
        plt.savefig(os.path.join(save_path, plot_filename))
        plt.close()

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

def reldens_sat_megaplot(base_dir, sat_name, moving_avg_minutes=45):
    storm_analysis_dir = os.path.join(base_dir, sat_name)
    if not os.path.exists(storm_analysis_dir):
        print(f"No data directory found for {sat_name}")
        return
    
    _, kp_3hrly, hourly_dst = get_kp_ap_dst_f107()
    kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
    hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

    storm_data = []
    unique_dates = set()

    for storm_file in sorted(os.listdir(storm_analysis_dir)):
        storm_file_path = os.path.join(storm_analysis_dir, storm_file)
        if os.path.isfile(storm_file_path):
            df = pd.read_csv(storm_file_path)
            df['UTC'] = pd.to_datetime(df['UTC'], utc=True)
            df.set_index('UTC', inplace=True)
            df.index = df.index.tz_convert('UTC')

            start_time = df.index.min()
            if start_time.strftime("%Y-%m-%d") in unique_dates:
                continue
            unique_dates.add(start_time.strftime("%Y-%m-%d"))

            df = get_arglat_from_df(df)

            density_types = ['Computed Density']
            for density_type in density_types:
                if density_type in df.columns:
                    df[density_type] = df[density_type].rolling(window=moving_avg_minutes, min_periods=1, center=True).mean()

            kp_filtered = kp_3hrly[(kp_3hrly['DateTime'] >= start_time) & (kp_3hrly['DateTime'] <= start_time + datetime.timedelta(days=3))]
            max_kp_time = kp_filtered.loc[kp_filtered['Kp'].idxmax()]['DateTime'] if not kp_filtered.empty else start_time

            storm_category = determine_storm_category(kp_filtered['Kp'].max())
            storm_number = -int(storm_category[1:]) if storm_category != "Below G1" else 0

            # Adjust the plotting times based on the max Kp time
            adjusted_start_time = max_kp_time - datetime.timedelta(hours=12)
            adjusted_end_time = max_kp_time + datetime.timedelta(hours=32)

            storm_data.append((df, adjusted_start_time, adjusted_end_time, storm_category, storm_number))

    storm_data.sort(key=lambda x: x[4], reverse=True)

    num_storms = len(storm_data)
    ncols = 4
    nrows = (num_storms + ncols - 1) // ncols  # This ensures we don't have any extra blank rows
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.2 * ncols, 1 * nrows), dpi=600)
    axes = axes.flatten()

    # Hide unused axes if the number of plots isn't a perfect multiple of nrows * ncols
    for i in range(len(storm_data), len(axes)):
        axes[i].set_visible(False)

    cmap = 'nipy_spectral'

    for i, (df, adjusted_start_time, adjusted_end_time, storm_category, storm_number) in enumerate(storm_data):
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            first_x, first_y, first_z = df.iloc[0][['x', 'y', 'z']]
            altitude = ((first_x**2 + first_y**2 + first_z**2)**0.5 - 6378137) / 1000
        else:
            altitude = 0  # Default to 0 if x, y, z are not available

        plot_df = df[(df.index >= adjusted_start_time) & (df.index <= adjusted_end_time)]
        
        local_min_density = plot_df['Computed Density'].min()
        local_max_density = plot_df['Computed Density'].max()

        relative_densities = (plot_df['Computed Density'] - local_min_density) / (local_max_density - local_min_density)
        
        sc = axes[i].scatter(plot_df.index, plot_df['arglat'], c=relative_densities, cmap=cmap, alpha=0.7, edgecolor='none', s=5)
        axes[i].set_title(f'{adjusted_start_time.strftime("%Y-%m-%d")}, {storm_category}, {altitude:.0f}km', fontsize=10)
        axes[i].set_ylabel(' ')
        axes[i].set_xlabel(' ')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.subplots_adjust(left=0.055, bottom=0.012, right=0.905, top=0.967, wspace=0.2, hspace=0.288)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Computed Density', rotation=270, labelpad=15)
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}_computed_density_plots.png', dpi=300, bbox_inches='tight')

def model_reldens_sat_megaplot(base_dir, sat_name, moving_avg_minutes=45):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    storm_analysis_dir = os.path.join(base_dir, sat_name)
    if not os.path.exists(storm_analysis_dir):
        return
    
    _, kp_3hrly, hourly_dst = get_kp_ap_dst_f107()
    kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
    hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

    storm_data = []
    unique_dates = set()

    density_diff_titles = ['Computed-JB08', 'Computed-DTM2000', 'Computed-NRLMSISE00']

    for storm_file in sorted(os.listdir(storm_analysis_dir)):
        storm_file_path = os.path.join(storm_analysis_dir, storm_file)
        if os.path.isfile(storm_file_path):
            df = pd.read_csv(storm_file_path)
            df['UTC'] = pd.to_datetime(df['UTC'], utc=True)
            df.set_index('UTC', inplace=True)
            df.index = df.index.tz_convert('UTC')

            start_time = df.index.min()
            if start_time.strftime("%Y-%m-%d") in unique_dates:
                continue
            unique_dates.add(start_time.strftime("%Y-%m-%d"))

            df = get_arglat_from_df(df)

            window_size = (moving_avg_minutes * 60) // pd.to_timedelta(pd.infer_freq(df.index)).seconds if moving_avg_minutes > 0 else 1
            shift_periods = (moving_avg_minutes * 30) // pd.to_timedelta(pd.infer_freq(df.index)).seconds if moving_avg_minutes > 0 else 0

            df['Computed Density'] = df['Computed Density'].rolling(window=window_size, min_periods=1, center=True).mean().shift(-shift_periods)
            for model in ['JB08', 'DTM2000', 'NRLMSISE00']:
                df[f'Computed-{model}'] = df['Computed Density'] - df[model]

            kp_filtered = kp_3hrly[(kp_3hrly['DateTime'] >= start_time) & (kp_3hrly['DateTime'] <= start_time + timedelta(days=3))]
            max_kp_time = kp_filtered.loc[kp_filtered['Kp'].idxmax(), 'DateTime'] if not kp_filtered.empty else start_time

            storm_category = determine_storm_category(kp_filtered['Kp'].max() if not kp_filtered.empty else 0)
            storm_number = -int(storm_category[1:]) if storm_category != "Below G1" else 0

            adjusted_start_time = max_kp_time - timedelta(hours=12)
            adjusted_end_time = max_kp_time + timedelta(hours=32)

            storm_data.append((df, adjusted_start_time, adjusted_end_time, storm_category, storm_number))

    storm_data.sort(key=lambda x: x[4], reverse=True)

    num_storms = len(storm_data)
    ncols = 3
    nrows = num_storms

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.2 * ncols, 1 * nrows), dpi=600)
    if nrows == 1:
        axes = np.array([axes])
    
    for i, (df, adjusted_start_time, adjusted_end_time, storm_category, storm_number) in enumerate(storm_data):
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            first_x, first_y, first_z = df.iloc[0][['x', 'y', 'z']]
            altitude = ((first_x**2 + first_y**2 + first_z**2)**0.5 - 6378137) / 1000
        else:
            altitude = 0

        row_min = np.inf
        row_max = -np.inf
        
        for density_diff_title in density_diff_titles:
            plot_df = df[(df.index >= adjusted_start_time) & (df.index <= adjusted_end_time)]
            row_min = min(row_min, plot_df[density_diff_title].min())
            row_max = max(row_max, plot_df[density_diff_title].max())

        absolute_max = max(abs(row_min), abs(row_max))

        for j, density_diff_title in enumerate(density_diff_titles):
            ax_idx = i * ncols + j
            if ax_idx >= len(axes.flatten()):
                continue

            plot_df = df[(df.index >= adjusted_start_time) & (df.index <= adjusted_end_time)]
            sc = axes[i][j].scatter(plot_df.index, plot_df['arglat'], c=plot_df[density_diff_title], cmap='coolwarm', vmin=-absolute_max, vmax=absolute_max, alpha=0.7, edgecolor='none', s=7)
            axes[i][j].set_title(f'{adjusted_start_time.strftime("%Y-%m-%d")}, {storm_category}, {altitude:.0f}km', fontsize=10)
            axes[i][j].set_ylabel('')
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

            if j == ncols - 1:
                cbar = plt.colorbar(sc, ax=axes[i][j], fraction=0.046, pad=0.04)
                cbar.set_label('Δρ', rotation=270, labelpad=10)

    plt.subplots_adjust(left=0.055, bottom=0.012, right=0.905, top=0.967, wspace=0.2, hspace=0.32)
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}_megadensity_diff_plots.png', dpi=300, bbox_inches='tight')
