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
from source.tools.SWIndices import get_sw_indices
from org.orekit.frames import FramesFactory
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

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

def plot_relative_density_change(data_frames, moving_avg_minutes, sat_name, start_date=None, stop_date=None):
    sns.set_style("darkgrid", {
        'axes.facecolor': '#2d2d2d', 'axes.edgecolor': 'white',
        'axes.labelcolor': 'white', 'xtick.color': 'white',
        'ytick.color': 'white', 'figure.facecolor': '#2d2d2d', 'text.color': 'white'
    })

    #set the UTC column as the index but also keep it as a column
    for density_df in data_frames:
        if 'UTC' in density_df.columns:
            density_df['UTC'] = pd.to_datetime(density_df['UTC'], utc=True)
            density_df.set_index('UTC', inplace=True)

    #use the UTC to stop the data frame after 30 hours from the start date
    first_time = data_frames[0].index[0]
    thirty_hours = first_time + pd.Timedelta(hours=30)
    for density_df in data_frames:
        density_df = density_df[(density_df.index >= first_time) & (density_df.index <= thirty_hours)]

    density_types = ['Computed Density', 'JB08', 'DTM2000', 'NRLMSISE00']
    titles = ['Delta Density: Computed vs JB08', 'Delta Density: Computed vs DTM2000', 'Delta Density: Computed vs NRLMSISE00']

    fig, axes = plt.subplots(nrows=len(titles), ncols=1, figsize=(5, 3 * len(titles)), dpi=200, constrained_layout=True)

    daily_indices, kp_3hrly, hourly_dst = get_sw_indices()

    global_min = float('inf')
    global_max = float('-inf')

    for density_df in data_frames:
        density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True) if 'Epoch' in density_df.columns else density_df.index
        first_epoch = density_df['Epoch'].iloc[0]
        density_df = get_arglat_from_df(density_df)
        density_df.set_index('Epoch', inplace=True)

        window_size = (moving_avg_minutes * 60) // 30
        for density_type in density_types:
            if density_type in density_df.columns:
                density_df[f'{density_type}'] = density_df[density_type].rolling(window=window_size, min_periods=1, center=True).mean()
                density_df = density_df.iloc[450:-450]

        median_density = density_df['Computed Density'].median()
        IQR = density_df['Computed Density'].quantile(0.75) - density_df['Computed Density'].quantile(0.25)
        lower_bound = median_density - 10 * IQR
        upper_bound = median_density + 10 * IQR
        density_df['Computed Density'] = density_df['Computed Density'].apply(lambda x: median_density if x < lower_bound or x > upper_bound else x)
        for density_type in density_types:
            initial_value = density_df[f'{density_type}'].iloc[0]
            density_df[f'{density_type} Delta'] = density_df[f'{density_type}'] - initial_value

        density_df.index = density_df.index.tz_localize(None)
        daily_indices = daily_indices[(daily_indices['Date'] >= density_df.index[0]) & (daily_indices['Date'] <= density_df.index[-1] + offsets.Hour())]
        kp_3hrly = kp_3hrly[(kp_3hrly['DateTime'] >= density_df.index[0]) & (kp_3hrly['DateTime'] <= density_df.index[-1] + offsets.Hour())]
        hourly_dst = hourly_dst[(hourly_dst['DateTime'] >= density_df.index[0]) & (hourly_dst['DateTime'] <= density_df.index[-1] + offsets.Hour())]
        hourly_dst = hourly_dst.sort_values('DateTime')
        kp_3hrly = kp_3hrly.sort_values('DateTime')

        for j, title in enumerate(titles):
            model_density = density_types[j + 1]
            if f'{model_density} Delta' in density_df.columns:
                density_df[f'Relative Change {model_density}'] = density_df['Computed Density Delta'] - density_df[f'{model_density} Delta']
                sc = axes[j].scatter(density_df.index, density_df['arglat'], c=density_df[f'Relative Change {model_density}'], cmap='rainbow', alpha=0.6, edgecolor='none')
                axes[j].set_title(title, fontsize=12)
                axes[j].set_xlabel('Time (UTC)')
                for label in axes[j].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                axes[j].set_ylabel('Argument of Latitude')
                cbar = fig.colorbar(sc, ax=axes[j], aspect=10)
                cbar.set_label('delta drho/dt (kg/m³/s)', rotation=270, labelpad=15)

                # Update global min and max values
                global_min = min(global_min, density_df[f'Relative Change {model_density}'].min())
                global_max = max(global_max, density_df[f'Relative Change {model_density}'].max())

                # Overlay Space Weather Indices
                axes_secondary = axes[j].twinx()
                axes_secondary.plot(hourly_dst['DateTime'], hourly_dst['Value'], label='Dst Index', c='xkcd:purple', linewidth=1)
                axes_secondary.set_ylabel('Dst Index', color='xkcd:purple')
                axes_secondary.tick_params(axis='y', colors='xkcd:purple')  # Adjusted color of y-axis ticks
                axes_secondary.yaxis.label.set_color('xkcd:purple')  # Adjusted color of y-axis label

                axes_tertiary = axes[j].twinx()
                axes_tertiary.plot(kp_3hrly['DateTime'], kp_3hrly['Kp'], label='Kp Index', c='xkcd:bright pink', linewidth=1)
                axes_tertiary.set_ylabel('Kp Index', color='xkcd:bright pink')
                axes_tertiary.tick_params(axis='y', colors='xkcd:bright pink')  # Adjusted color of y-axis ticks
                axes_tertiary.yaxis.label.set_color('xkcd:bright pink')  # Adjusted color of y-axis label
                axes_tertiary.spines['right'].set_position(('outward', 40))  # Offset Kp Index axis

    # Set the colorbar limits
    for ax in axes:
        for c in ax.collections:
            c.set_clim(global_min, global_max)

    plt.suptitle(f'Relative Change in Atmospheric Density for {sat_name}', color='white')  # Adjusted color of title
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

    _, kp_3hrly, hourly_dst = get_sw_indices()
    
    kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
    hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

    for i in range(len(data_frames)):
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
        diff_vmin, diff_vmax = float('inf'), float('-inf')
        for density_type in density_types:
            if density_type in density_df.columns:
                smoothed_values = density_df[density_type].rolling(window=window_size, min_periods=1, center=True).mean().shift(-shift_periods)
                density_df.loc[:, f'{density_type}'] = smoothed_values

                if density_type == 'Computed Density':
                    median_density = density_df[density_type].median()
                    IQR = density_df[density_type].quantile(0.75) - density_df[density_type].quantile(0.25)
                    lower_bound = median_density - 10 * IQR
                    upper_bound = median_density + 10 * IQR
                    density_df.loc[:, density_type] = density_df[density_type].apply(lambda x: median_density if x < lower_bound or x > upper_bound else x)

                if density_type != 'Computed Density':
                    density_df.loc[:, f'{density_type} Difference'] = density_df['Computed Density'] - density_df[density_type]
                    diff_vmax = max(diff_vmax, density_df[f'{density_type} Difference'].max())
                    diff_vmin = min(diff_vmin, density_df[f'{density_type} Difference'].min())
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
                sc_diff = axes[j, 1].scatter(density_df.index, density_df['arglat'], c=density_df[f'{density_type} Difference'], cmap='coolwarm', alpha=0.6, edgecolor='none', vmin=diff_vmin, vmax=diff_vmax)
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
    #set the ticks to be every 1
    ax_kp.set_yticks(np.arange(0, 10, 3))

    max_kp_value = kp_3hrly_analysis['Kp'].max()
    storm_category = "Below G1" if max_kp_value < 5 else "G1" if max_kp_value < 6 else "G2" if max_kp_value < 7 else "G3" if max_kp_value < 8 else "G4" if max_kp_value < 9 else "G5"

    day, month, year = analysis_start_time.day, analysis_start_time.month, analysis_start_time.year
    plt.suptitle(f'Atmospheric Density as Function of Argument of Latitude for {sat_name} - {storm_category} Storm\n{day}/{month}/{year}', color='white')
    plt.tight_layout()
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/SWI_densitydiff_arglat{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', dpi=600)

def plot_density_data(data_frames, moving_avg_minutes, sat_name):
    sns.set_style(style="whitegrid")
    
    # Define color palette for all densities including model densities
    custom_palette = sns.color_palette("Set2", len(data_frames) + 3)  # Adding 3 for model densities

    # First plot for the computed densities MAs
    plt.figure(figsize=(8, 4))
    for i, density_df in enumerate(data_frames):
        seconds_per_point = pd.to_timedelta(pd.infer_freq(density_df.index)).seconds
        window_size = (moving_avg_minutes * 60) // seconds_per_point
        density_df['Computed Density'] = density_df['Computed Density'].rolling(window=window_size, center=False).mean()
        shift_periods = int((moving_avg_minutes / 2 * 60) // seconds_per_point)
        density_df['Computed Density'] = density_df['Computed Density'].shift(-shift_periods)
        mean_density = density_df['Computed Density'].mean()
        density_df['Computed Density'] = density_df['Computed Density'].apply(lambda x: mean_density if x > 100 * mean_density or x < mean_density / 100 else x)

    # Second plot for the first data frame with model densities along with computed densities
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='JB08', label='JB08 Density', color="xkcd:forest green")
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='DTM2000', label='DTM2000 Density', color="xkcd:orange")
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='NRLMSISE00', label='NRLMSISE00 Density', color="xkcd:turquoise")

    # Include computed densities from all data frames again on the same plot
    for i, density_df in enumerate(data_frames):
        sns.lineplot(data=density_df, x=density_df.index, y='Computed Density', label=f'Computed Density', linestyle='--', color="xkcd:hot pink")

    plt.title('Model Densities vs. Computed Densities', fontsize=12)
    plt.xlabel('Time (UTC)', fontsize=12)
    plt.ylabel('Density (log scale)', fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/model_density_vs_computed_density_{datenow}.png')

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
    
    _, kp_3hrly, hourly_dst = get_sw_indices()
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
    storm_analysis_dir = os.path.join(base_dir, sat_name)
    if not os.path.exists(storm_analysis_dir):
        print(f"No data directory found for {sat_name}")
        return
    
    _, kp_3hrly, hourly_dst = get_sw_indices()
    kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
    hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

    storm_data = []
    unique_dates = set()

    density_types = ['JB08', 'DTM2000', 'NRLMSISE00']
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

            # Compute the density differences
            for density_type in density_types:
                df[f'Computed-{density_type}'] = df['Computed Density'] - df[density_type]

            kp_filtered = kp_3hrly[(kp_3hrly['DateTime'] >= start_time) & (kp_3hrly['DateTime'] <= start_time + datetime.timedelta(days=3))]
            max_kp_time = kp_filtered.loc[kp_filtered['Kp'].idxmax(), 'DateTime'] if not kp_filtered.empty else start_time

            storm_category = determine_storm_category(kp_filtered['Kp'].max() if not kp_filtered.empty else 0)
            storm_number = -int(storm_category[1:]) if storm_category != "Below G1" else 0

            adjusted_start_time = max_kp_time - datetime.timedelta(hours=12)
            adjusted_end_time = max_kp_time + datetime.timedelta(hours=32)

            storm_data.append((df, adjusted_start_time, adjusted_end_time, storm_category, storm_number))

    storm_data.sort(key=lambda x: x[4], reverse=True)

    num_storms = len(storm_data)
    ncols = 3
    total_plots = 3 * num_storms
    nrows = (total_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.2 * ncols, 2 * nrows), dpi=600)
    axes = axes.flatten()

    for i, (df, adjusted_start_time, adjusted_end_time, storm_category, storm_number) in enumerate(storm_data):
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            first_x, first_y, first_z = df.iloc[0][['x', 'y', 'z']]
            altitude = ((first_x**2 + first_y**2 + first_z**2)**0.5 - 6378137) / 1000
        else:
            altitude = 0

        for j, density_diff_title in enumerate(density_diff_titles):
            ax_idx = i * ncols + j
            if ax_idx >= len(axes):  # Check if the index is within the range of created axes
                print(f"Trying to access axes[{ax_idx}] but only have {len(axes)} axes.")
                continue

            plot_df = df[(df.index >= adjusted_start_time) & (df.index <= adjusted_end_time)]
            
            local_min_density = plot_df[density_diff_title].min()
            local_max_density = plot_df[density_diff_title].max()

            if local_max_density != local_min_density:
                relative_densities = (plot_df[density_diff_title] - local_min_density) / (local_max_density - local_min_density)
            else:
                relative_densities = np.zeros_like(plot_df[density_diff_title])

            sc = axes[ax_idx].scatter(plot_df.index, plot_df['arglat'], c=relative_densities, cmap='nipy_spectral', alpha=0.7, edgecolor='none', s=5)
            axes[ax_idx].set_title(f'{adjusted_start_time.strftime("%Y-%m-%d")}, {storm_category}, {altitude:.0f}km, {density_diff_title}', fontsize=10)
            axes[ax_idx].set_ylabel(' ')
            axes[ax_idx].set_xlabel(' ')
            axes[ax_idx].set_xticks([])
            axes[ax_idx].set_yticks([])

    # Hide any unused axes
    for k in range(i * ncols + j + 1, len(axes)):
        axes[k].set_visible(False)

    plt.subplots_adjust(left=0.055, bottom=0.012, right=0.905, top=0.967, wspace=0.2, hspace=0.288)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='nipy_spectral')
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Density Difference', rotation=270, labelpad=15)
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}_megadensity_diff_plots.png', dpi=300, bbox_inches='tight')
    # plt.show()