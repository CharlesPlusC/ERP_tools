import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm
from pandas.tseries import offsets
from SWIndices import get_sw_indices
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import PVCoordinates
from orekit.pyhelpers import datetime_to_absolutedate
from tools.utilities import pv_to_kep
from org.orekit.frames import FramesFactory
import os

def get_arglat_from_df(densitydf_df):
    frame = FramesFactory.getEME2000()
    use_column = 'Epoch' in densitydf_df.columns

    for index, row in densitydf_df.iterrows():
        x = row['x']
        y = row['y']
        z = row['z']
        xv = row['xv']
        yv = row['yv']
        zv = row['zv']
        
        if use_column:
            UTC = row['Epoch']
            # Check if UTC needs to be converted from string to datetime
            if isinstance(UTC, str):
                UTC = datetime.datetime.strptime(UTC, '%Y-%m-%d %H:%M:%S')
        else:
            UTC = index  # Use the index directly, which should already be in datetime format

        position = Vector3D(float(x), float(y), float(z))
        velocity = Vector3D(float(xv), float(yv), float(zv))
        pvCoordinates = PVCoordinates(position, velocity)
        time = datetime_to_absolutedate(UTC)
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

    density_types = ['Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']
    titles = ['Delta Density: Computed vs JB08', 'Delta Density: Computed vs DTM2000', 'Delta Density: Computed vs NRLMSISE00']

    fig, axes = plt.subplots(nrows=len(titles), ncols=1, figsize=(5, 3 * len(titles)), dpi=200, constrained_layout=True)

    daily_indices, kp_3hrly, hourly_dst = get_sw_indices()

    global_min = float('inf')
    global_max = float('-inf')

    for density_df in data_frames:
        density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True) if 'Epoch' in density_df.columns else density_df.index
        density_df = get_arglat_from_df(density_df)
        density_df.set_index('Epoch', inplace=True)

        window_size = (moving_avg_minutes * 60) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds if moving_avg_minutes > 0 else 1
        for density_type in density_types:
            if density_type in density_df.columns:
                density_df[f'{density_type} MA'] = density_df[density_type].rolling(window=window_size, min_periods=1, center=True).mean()
                density_df = density_df.iloc[450:-450]

        for density_type in density_types:
            initial_value = density_df[f'{density_type} MA'].iloc[0]
            density_df[f'{density_type} Delta'] = density_df[f'{density_type} MA'] - initial_value

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
                axes_secondary.plot(hourly_dst['DateTime'], hourly_dst['Value'], label='Dst Index', c='xkcd:purple', linewidth=2)
                axes_secondary.set_ylabel('Dst Index', color='xkcd:purple')
                axes_secondary.tick_params(axis='y', colors='xkcd:purple')  # Adjusted color of y-axis ticks
                axes_secondary.yaxis.label.set_color('xkcd:purple')  # Adjusted color of y-axis label

                axes_tertiary = axes[j].twinx()
                axes_tertiary.plot(kp_3hrly['DateTime'], kp_3hrly['Kp'], label='Kp Index', c='xkcd:bright pink', linewidth=2)
                axes_tertiary.set_ylabel('Kp Index', color='xkcd:bright pink')
                axes_tertiary.tick_params(axis='y', colors='xkcd:bright pink')  # Adjusted color of y-axis ticks
                axes_tertiary.yaxis.label.set_color('xkcd:bright pink')  # Adjusted color of y-axis label
                axes_tertiary.spines['right'].set_position(('outward', 40))  # Offset Kp Index axis

    # Set the colorbar limits
    for ax in axes:
        for c in ax.collections:
            c.set_clim(global_min, global_max)

    plt.suptitle(f'Relative Change in Atmospheric Density for {sat_name}', color='white')  # Adjusted color of title
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/rel_densitydiff_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', dpi=600)

def plot_density_arglat_diff(data_frames, moving_avg_minutes, sat_name):
    sns.set_style("darkgrid", {
        'axes.facecolor': '#2d2d2d', 'axes.edgecolor': 'white', 
        'axes.labelcolor': 'white', 'xtick.color': 'white', 
        'ytick.color': 'white', 'figure.facecolor': '#2d2d2d', 'text.color': 'white'
    })
    
    density_types = ['Computed Density', 'JB08 Density', 'DTM2000 Density', 'NRLMSISE00 Density']
    titles = ['Computed Density', 'JB08 Model Density', 'DTM2000 Model Density', 'NRLMSISE00 Model Density']
    density_diff_titles = ['|Computed - JB08|', '|Computed - DTM2000|', '|Computed - NRLMSISE00|']

    nrows = len(density_types)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 5 * nrows), dpi=200)

    vmin, vmax = 3e-13, 2e-12
    diff_vmin, diff_vmax = 1e-15, 1e-11

    for i, density_df in enumerate(data_frames):
        density_df = get_arglat_from_df(density_df)
        density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True)
        density_df.set_index('Epoch', inplace=True)

        window_size = (moving_avg_minutes * 60) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds if moving_avg_minutes > 0 else 1
        shift_periods = (moving_avg_minutes * 30) // pd.to_timedelta(pd.infer_freq(density_df.index)).seconds if moving_avg_minutes > 0 else 0

        for density_type in density_types:
            if density_type in density_df.columns:
                density_df[f'{density_type} MA'] = density_df[density_type].rolling(window=window_size, min_periods=1, center=True).mean().shift(-shift_periods)
                if density_type != 'Computed Density':
                    density_df[f'{density_type} Difference'] = abs(density_df['Computed Density MA'] - density_df[f'{density_type} MA'])

        for j, density_type in enumerate(density_types):
            if f'{density_type} MA' in density_df.columns:
                sc = axes[j, 0].scatter(density_df.index, density_df['arglat'], c=density_df[f'{density_type} MA'], cmap='cubehelix', alpha=0.6, edgecolor='none', norm=LogNorm(vmin=vmin, vmax=vmax))
                axes[j, 0].set_title(titles[j], fontsize=12)
                axes[j, 0].set_xlabel('Time (UTC)')
                axes[j, 0].set_ylabel('Argument of Latitude')
                cbar = fig.colorbar(sc, ax=axes[j, 0])
                cbar.set_label('Density (kg/m³)', rotation=270, labelpad=15)
            if density_type != 'Computed Density' and f'{density_type} Difference' in density_df.columns:
                sc_diff = axes[j, 1].scatter(density_df.index, density_df['arglat'], c=density_df[f'{density_type} Difference'], cmap='coolwarm', alpha=0.6, edgecolor='none', norm=LogNorm(vmin=diff_vmin, vmax=diff_vmax))
                axes[j, 1].set_title(density_diff_titles[j - 1], fontsize=12)
                axes[j, 1].set_xlabel('Time (UTC)')
                axes[j, 1].set_ylabel('Argument of Latitude')
                cbar_diff = fig.colorbar(sc_diff, ax=axes[j, 1])
                cbar_diff.set_label('Density Difference (kg/m³)', rotation=270, labelpad=15)

    plt.suptitle(f'Atmospheric Density as Function of Argument of Latitude for {sat_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/densitydiff_arglat{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', dpi=600)

def plot_density_data(data_frames, moving_avg_minutes, sat_name):
    sns.set_style(style="whitegrid")
    
    # Define color palette for all densities including model densities
    custom_palette = sns.color_palette("Set2", len(data_frames) + 3)  # Adding 3 for model densities

    # First plot for the computed densities MAs
    plt.figure(figsize=(10, 6))
    for i, density_df in enumerate(data_frames):
        print(f'Processing density_df {i+1}')
        print(f'columns: {density_df.columns}')
        if density_df['Epoch'].dtype != 'datetime64[ns]':
            density_df['Epoch'] = pd.to_datetime(density_df['Epoch'], utc=True)
        if not isinstance(density_df.index, pd.DatetimeIndex):
            density_df.set_index('Epoch', inplace=True)
        if pd.infer_freq(density_df.index) is None:
            density_df = density_df.asfreq('infer')
        seconds_per_point = pd.to_timedelta(pd.infer_freq(density_df.index)).seconds
        window_size = (moving_avg_minutes * 60) // seconds_per_point
        density_df['Computed Density MA'] = density_df['Computed Density'].rolling(window=window_size, center=False).mean()
        shift_periods = int((moving_avg_minutes / 2 * 60) // seconds_per_point)
        density_df['Computed Density MA'] = density_df['Computed Density MA'].shift(-shift_periods)
        
        sns.lineplot(data=density_df, x=density_df.index, y='Computed Density MA', label=f'Computed Density {i+1}', linestyle='--', palette=[custom_palette[i]])

    plt.title(f'Computed and Modelled Atmospheric Density for {sat_name}', fontsize=14)
    plt.xlabel('Epoch (UTC)', fontsize=14)
    plt.ylabel('Density (log scale)', fontsize=14)
    plt.legend(loc='upper right', frameon=True)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    datenow = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.savefig(f'output/DensityInversion/PODBasedAccelerometry/Plots/{sat_name}/computed_density_moving_averages_{datenow}.png')

    # Second plot for the first data frame with model densities along with computed densities
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='JB08 Density', label='JB08 Density', color=custom_palette[-3])
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='DTM2000 Density', label='DTM2000 Density', color=custom_palette[-2])
    sns.lineplot(data=data_frames[0], x=data_frames[0].index, y='NRLMSISE00 Density', label='NRLMSISE00 Density', color=custom_palette[-1])

    # Include computed densities from all data frames again on the same plot
    for i, density_df in enumerate(data_frames):
        sns.lineplot(data=density_df, x=density_df.index, y='Computed Density MA', label=f'Computed Density {i+1} (overlay)', linestyle='--', palette=[custom_palette[i]])

    plt.title('Model Densities vs. Computed Density Moving Averages', fontsize=14)
    plt.xlabel('Epoch (UTC)', fontsize=14)
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