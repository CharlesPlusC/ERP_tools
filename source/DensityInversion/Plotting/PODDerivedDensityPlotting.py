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
    density_diff_titles = ['|Computed - JB08|', '|Computed - DTM2000|', '|Computed - NRLMSISE00|']

    nrows = len(density_types)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(9, 2 * nrows), dpi=600)

    # Get space weather indices
    _, kp_3hrly, hourly_dst = get_sw_indices()
    
    # Ensure DateTime is timezone-aware in UTC
    kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
    hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

    # Ensure all DataFrame indices in data_frames are timezone-aware
    for i in range(len(data_frames)):
        if data_frames[i].index.tz is None:
            data_frames[i].index = data_frames[i].index.tz_localize('UTC')
        else:
            data_frames[i].index = data_frames[i].index.tz_convert('UTC')

    # Determine the overall time frame of data_frames
    start_time = pd.to_datetime(min(df.index.min() for df in data_frames))
    end_time = pd.to_datetime(max(df.index.max() for df in data_frames))

    # Filter kp_3hrly based on this overall time frame
    kp_3hrly = kp_3hrly[(kp_3hrly['DateTime'] >= start_time) & (kp_3hrly['DateTime'] <= end_time)]

    # Sort by DateTime to avoid odd line connections
    kp_3hrly = kp_3hrly.sort_values(by='DateTime')
    hourly_dst = hourly_dst.sort_values(by='DateTime')

    # Find the time of maximum Kp within this frame
    max_kp_time = kp_3hrly.loc[kp_3hrly['Kp'].idxmax(), 'DateTime']

    # Define analysis period from 12 hours before to 24 hours after this max Kp time
    analysis_start_time = max_kp_time - timedelta(hours=12)
    analysis_end_time = max_kp_time + timedelta(hours=32)

    # Filter all data based on this analysis period
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
                axes[j, 0].set_xlabel('Time (UTC)')
                for label in axes[j, 0].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                axes[j, 0].set_ylabel('Argument of Latitude')
                cbar = fig.colorbar(sc, ax=axes[j, 0])
                cbar.set_label('Density (kg/m³)', rotation=270, labelpad=15)

            if density_type != 'Computed Density' and f'{density_type} Difference' in density_df.columns:
                sc_diff = axes[j, 1].scatter(density_df.index, density_df['arglat'], c=density_df[f'{density_type} Difference'], cmap='coolwarm', alpha=0.6, edgecolor='none', vmin=diff_vmin, vmax=diff_vmax)
                axes[j, 1].set_title(density_diff_titles[j - 1], fontsize=12)
                axes[j, 1].set_xlabel('Time (UTC)')
                for label in axes[j, 1].get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                axes[j, 1].set_ylabel('Argument of Latitude')
                cbar_diff = fig.colorbar(sc_diff, ax=axes[j, 1])
                cbar_diff.set_label('Density Difference (kg/m³)', rotation=270, labelpad=15)

    ax_right_top = axes[0, 1]
    ax_kp = ax_right_top.twinx()

    ax_right_top.plot(hourly_dst_analysis['DateTime'], hourly_dst_analysis['Value'], 'b-', label='Dst (nT)', linewidth=2)
    ax_kp.plot(kp_3hrly_analysis['DateTime'], kp_3hrly_analysis['Kp'], 'r-', label='Kp', linewidth=2)

    ax_right_top.set_xlabel('Time (UTC)')
    ax_right_top.set_ylabel('Dst (nT)', color='b')
    ax_right_top.yaxis.label.set_color('blue')  # Ensures the y-axis label is blue
    ax_right_top.set_ylim(-300, 50)  # Fix the y-axis range for Dst

    ax_kp.set_ylabel('Kp', color='r')
    ax_kp.yaxis.label.set_color('red')  # Ensures the y-axis label is red
    ax_kp.set_ylim(0, 9)  # Fix the y-axis range for Kp

    for label in ax_right_top.get_xticklabels() + ax_kp.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    lines, labels = ax_right_top.get_legend_handles_labels()
    lines2, labels2 = ax_kp.get_legend_handles_labels()
    ax_kp.legend(lines + lines2, labels + labels2, loc='upper right')

    # Determine the storm category based on the highest Kp value
    max_kp_value = kp_3hrly_analysis['Kp'].max()
    if max_kp_value < 5:
        storm_category = "Below G1"
    elif max_kp_value < 6:
        storm_category = "G1"
    elif max_kp_value < 7:
        storm_category = "G2"
    elif max_kp_value < 8:
        storm_category = "G3"
    elif max_kp_value < 9:
        storm_category = "G4"
    else:
        storm_category = "G5"

    # Include the storm category in the title
    #get the day, month and year of the analysis_start_time
    day = analysis_start_time.day
    month = analysis_start_time.month
    year = analysis_start_time.year
    plt.suptitle(f'Atmospheric Density as Function of Argument of Latitude for {sat_name} - {storm_category} Storm\n{day}/{month}/{year}', color='white')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/SWI_densitydiff_arglat{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', dpi=600)

def plot_density_data(data_frames, moving_avg_minutes, sat_name):
    sns.set_style(style="whitegrid")
    
    # Define color palette for all densities including model densities
    custom_palette = sns.color_palette("Set2", len(data_frames) + 3)  # Adding 3 for model densities

    # First plot for the computed densities MAs
    plt.figure(figsize=(8, 4))
    for i, density_df in enumerate(data_frames):
        print(f'Processing density_df {i+1}')
        print(f'columns: {density_df.columns}')
        # if density_df['UTC'].dtype != 'datetime64[ns]':
        #     density_df['UTC'] = pd.to_datetime(density_df['UTC'], utc=True)
        # if not isinstance(density_df.index, pd.DatetimeIndex):
        #     density_df.set_index('UTC', inplace=True)
        # if pd.infer_freq(density_df.index) is None:
        #     density_df = density_df.asfreq('infer')
        seconds_per_point = pd.to_timedelta(pd.infer_freq(density_df.index)).seconds
        window_size = (moving_avg_minutes * 60) // seconds_per_point
        density_df['Computed Density'] = density_df['Computed Density'].rolling(window=window_size, center=False).mean()
        shift_periods = int((moving_avg_minutes / 2 * 60) // seconds_per_point)
        density_df['Computed Density'] = density_df['Computed Density'].shift(-shift_periods)
        mean_density = density_df['Computed Density'].mean()
        density_df['Computed Density'] = density_df['Computed Density'].apply(lambda x: mean_density if x > 100 * mean_density or x < mean_density / 100 else x)

        sns.lineplot(data=density_df, x=density_df.index, y='Computed Density', label=f'Computed Density', linestyle='--')
        sns.lineplot(data=density_df, x=density_df.index, y='JB08', label=f'JB08 Density')
        sns.lineplot(data=density_df, x=density_df.index, y='DTM2000', label=f'DTM2000 Density')
        sns.lineplot(data=density_df, x=density_df.index, y='NRLMSISE00', label=f'NRLMSISE00 Density')

    plt.title(f'Computed and Modelled Atmospheric Density for {sat_name}', fontsize=14)
    plt.xlabel('Time (UTC)', fontsize=14)
    plt.ylabel('Density (log scale)', fontsize=14)
    plt.legend(loc='upper right', frameon=True)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    datenow = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/computed_density_moving_averages_{datenow}.png', dpi=600)

    # Second plot for the first data frame with model densities along with computed densities
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='JB08', label='JB08 Density', color=custom_palette[-3])
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='DTM2000', label='DTM2000 Density', color=custom_palette[-2])
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='NRLMSISE00', label='NRLMSISE00 Density', color=custom_palette[-1])

    # Include computed densities from all data frames again on the same plot
    for i, density_df in enumerate(data_frames):
        sns.lineplot(data=density_df, x=density_df.index, y='Computed Density', label=f'Computed Density {i+1}', linestyle='--', palette=[custom_palette[i]])

    plt.title('Model Densities vs. Computed Density Moving Averages', fontsize=14)
    plt.xlabel('Time (UTC)', fontsize=14)
    plt.ylabel('Density (log scale)', fontsize=14)
    plt.legend(loc='upper right', frameon=True)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/model_density_vs_computed_density_{datenow}.png')

def density_compare_scatter(density_df, moving_avg_window, sat_name):
    
    save_path = f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert moving average minutes to the number of points based on data frequency
    if not isinstance(density_df.index, pd.DatetimeIndex):
        density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True)
        density_df.set_index('Epoch', inplace=True)
    
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