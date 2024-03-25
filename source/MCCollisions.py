import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

import os
from tools.utilities import get_satellite_info
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import propagate_state
from tools.BatchLeastSquares import OD_BLS
from tools.collision_tools import generate_collision_trajectory
import numpy as np
import numpy.random as npr
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 60.0
POSITION_TOLERANCE = 1e-2 # 1 cm

def interpolate_ephemeris(df, start_time, end_time, freq='0.0001S', stitch=False):
    # Ensure UTC is the index and not duplicated
    df = df.drop_duplicates(subset='UTC').set_index('UTC')
    #sort by UTC
    df = df.sort_index()
    # Create a new DataFrame with resampled frequency between the specified start and end times
    df_resampled = df.reindex(pd.date_range(start=start_time, end=end_time, freq=freq), method='nearest').asfreq(freq)
    # Interpolate values using a linear method
    interp_funcs = {col: interp1d(df.index.astype(int), df[col], fill_value='extrapolate') for col in ['x', 'y', 'z']}
    for col in ['x', 'y', 'z']:
        df_resampled[col] = interp_funcs[col](df_resampled.index.astype(int))
    # Filter out the part of the resampled DataFrame within the start and end time
    df_filtered = df_resampled.loc[start_time:end_time].reset_index().rename(columns={'index': 'UTC'})
    if stitch:
        # Concatenate the original DataFrame with the filtered resampled DataFrame
        # This ensures that data outside the interpolation range is kept intact
        df_stitched = pd.concat([
            df.loc[:start_time - pd.Timedelta(freq), :],  # Data before the interpolation interval
            df_filtered.set_index('UTC'),
            df.loc[end_time + pd.Timedelta(freq):, :]    # Data after the interpolation interval
        ]).reset_index()
        return df_stitched
    return df_filtered

def generate_random_vectors(eigenvalues, num_samples):
    random_vectors = []
    for lambda_val in eigenvalues:
        for _ in range(num_samples):
            vector = [npr.normal(0, np.sqrt(lambda_val)) for _ in range(6)]
            random_vectors.append(vector)
    return random_vectors

def apply_perturbations(states, vectors, rotation_matrices):
    perturbed_states = []
    for state, vector_set, rotation_matrix in zip(states, vectors, rotation_matrices):
        for vector in vector_set:
            perturbation = np.dot(rotation_matrix, vector)
            perturbed_states.append(state + perturbation)
    return perturbed_states

def generate_perturbed_states(optimized_state_cov, state, num_samples):
    eig_vals, rotation_matrix = np.linalg.eigh(optimized_state_cov)
    random_vectors = generate_random_vectors(eig_vals, num_samples)
    perturbed_states = apply_perturbations([state], [random_vectors], [rotation_matrix])
    return perturbed_states

def main():
    sat_names_to_test = ["GRACE-FO-A"]
    # , "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"
    arc_length = 5  # min
    num_arcs = 1
    prop_length = 60 * 60 * 1
    prop_length_days = prop_length / (60 * 60 * 24)
    force_model_configs = [
                        {'36x36gravity': True, '3BP': True},
                        # {'120x120gravity': True, '3BP': True},
                        # {'120x120gravity': True, '3BP': True, 'SRP': True, 'nrlmsise00drag': True},
                        {'120x120gravity': True, '3BP': True,'SRP': True, 'jb08drag': True}]
    
    MC_ephem_folder = "output/Collisions/MC/interpolated_MC_ephems" #folder to save the interpolated ephemeris dataframes  
    if not os.path.exists(MC_ephem_folder):
        os.makedirs(MC_ephem_folder)
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        sat_info = get_satellite_info(sat_name)
        cd = sat_info['cd']
        cr = sat_info['cr']
        cross_section = sat_info['cross_section']
        mass = sat_info['mass']
        ephemeris_df = ephemeris_df.iloc[::2, :] # downsample to 60 second intervals
        obs_time_step = (ephemeris_df['UTC'].iloc[1] - ephemeris_df['UTC'].iloc[0]).total_seconds() / 60.0  # in minutes
        # find the time of the first time step
        t0 = ephemeris_df['UTC'].iloc[0]
        t_col = t0 + datetime.timedelta(days=prop_length_days) #this is the time at which the collision will occur
        #end of propagation window is 15 minutes after the collision
        t_end = t0 + datetime.timedelta(days=prop_length_days + 15/(60*24))
        collision_df = generate_collision_trajectory(ephemeris_df, t_col, dt=5.0, post_col_t=15.0) #Generate the collision trajectory with hi resolution for the interpolation
        # now make downsample collision_df to return only every minute (use the previous dt value to find how often the slice should be taken)
        collision_df_minute = collision_df.iloc[::12, :]
        collision_df_interp = interpolate_ephemeris(collision_df, t_col - datetime.timedelta(seconds=7), t_col + datetime.timedelta(seconds=7), stitch=True) #interpolate the collision trajectory very finely around the TCA
        arc_step = int(arc_length / obs_time_step)
        for arc in range(num_arcs):
            
            start_index = arc * arc_step
            end_index = start_index + arc_step
            arc_df = ephemeris_df.iloc[start_index:end_index]

            observations_df = arc_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
            initial_values = arc_df.iloc[0][['x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
            initial_t = arc_df.iloc[0]['UTC']
            initial_vals = np.array(initial_values.tolist() + [cd, cr, cross_section, mass], dtype=float)

            a_priori_estimate = np.concatenate(([initial_t.timestamp()], initial_vals))

            for fm_num, force_model_config in enumerate(force_model_configs):
                optimized_states, cov_mats, _, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag=False, max_patience=1)
                min_RMS_index = np.argmin(RMSs)
                optimized_state = optimized_states[min_RMS_index]
                optimized_state_cov = cov_mats[min_RMS_index]
                print("optimized_state_cov: ", optimized_state_cov)
                primary_states_perturbed_ephem = []
                perturbed_states_primary = generate_perturbed_states(optimized_state_cov, optimized_state, 5)

                for i, primary_state in enumerate(perturbed_states_primary):
                    print(f"propagating perturbed state {i} of {len(perturbed_states_primary)}")
                    primary_state_perturbed_df = propagate_state(start_date=t0, end_date=t_end, initial_state_vector=primary_state, cr=cr, cd=cd, cross_section=cross_section, mass=mass,boxwing=None,ephem=True,dt=5, **force_model_config)
                    primary_states_perturbed_ephem.append(primary_state_perturbed_df)

                for i, primary_state_perturbed_df in enumerate(primary_states_perturbed_ephem):
                    primary_states_perturbed_ephem[i] = interpolate_ephemeris(primary_state_perturbed_df, t_col - datetime.timedelta(seconds=7), t_col + datetime.timedelta(seconds=7), stitch=True)
                    #save the interpolated ephemeris dataframes to a folder
                    primary_states_perturbed_ephem[i].to_csv(f"{MC_ephem_folder}/{sat_name}_arc_{arc}_FM_{fm_num}_sample_{i}_interpolated.csv")

                #go through the list of dataframes and calculate the distance between every combination of primary and secondary statesu8
                distance_dfs = []
                # Iterate through each primary ephemeris dataframe
                #plot the time stamps of the interpolated ephemeris dataframes and the collision trajectory vs row number

                for i, primary_state_perturbed_df in enumerate(primary_states_perturbed_ephem):
                    # Align the primary and collision DataFrames by resampling both to a common time grid.
                    distances = np.linalg.norm(primary_state_perturbed_df[['x', 'y', 'z']].values - collision_df_interp[['x', 'y', 'z']].values, axis=1)
                    distance_df = pd.DataFrame({'UTC': primary_state_perturbed_df['UTC'], f'Distance_{i}': distances})
                    distance_dfs.append(distance_df)

                # Benchmark: calculate the distance between unperturbed primary and secondary states
                # get the subset of the ephemeris_df that is within the time window of the collision_df
                #make a new dataframe with the distance between ephemeris_df and collision_df (merge on UTC)
                col_to_ephem_distances = pd.merge(ephemeris_df, collision_df, on='UTC', suffixes=('_ephem', '_col'))
                #now take the diffeerence between the position vectors and put that in a new column called 'distance'
                col_to_ephem_distances['distance'] = np.linalg.norm(col_to_ephem_distances[['x_ephem', 'y_ephem', 'z_ephem']].values - col_to_ephem_distances[['x_col', 'y_col', 'z_col']].values, axis=1)

                #TODO: make the interpolation be a function of the rate of change of the distance between the two states
                #TODO: stitch the interpolated and non interpolated to get higher resolution around TCA
                #TODO: use the interpolated to get a better estimate fo the TCA (find DCA and invert for TCA)
                #TODO: make the plot from the PateraCollision.py script of propagated orbit to SP3 orbit

                # Concatenate all individual distance dataframes to get a single dataframe with all distances
                distances_df = pd.concat(distance_dfs, axis=1)

                print(f"simulation end. Total trajectories: {len(primary_states_perturbed_ephem)}")

                # plot_distance_time_series_heatmap_bokeh(distances_df, col_to_ephem_distances, t_col, sat_name, arc, fm_num)
                plot_distance_time_series_bokeh_datashader(distances_df, col_to_ephem_distances, t_col, sat_name, arc, fm_num)
                # plot_distance_time_series(distances_df, col_to_ephem_distances, t_col, sat_name, arc, fm_num)
                # plot_distance_time_series_heatmap(distances_df, col_to_ephem_distances, t_col, sat_name, arc, fm_num)
                # plot_minimum_distance_histogram(distances_df.min(axis=1), sat_name, arc, fm_num)

# num_distance_columns = len([col for col in filtered_df.columns if 'Distance' in col])

import pandas as pd
import numpy as np
import datetime
import os
import holoviews as hv
from holoviews.operation.datashader import datashade, dynspread
import datashader as ds
from bokeh.plotting import output_file, save

hv.extension('bokeh')

import pandas as pd
import numpy as np
import datetime
import os
import holoviews as hv
from holoviews.operation.datashader import datashade, dynspread
from bokeh.plotting import output_file, save

hv.extension('bokeh')



def validate_lengths(df):
    # Ensure all columns have the same length
    utc_len = len(df['UTC'])
    for col in df.columns:
        if 'Distance' in col and len(df[col]) != utc_len:
            raise ValueError(f"Length mismatch in column {col}")

import pandas as pd
import numpy as np
import datetime
import os
import holoviews as hv
from holoviews.operation.datashader import datashade, dynspread
from bokeh.plotting import output_file, save

hv.extension('bokeh')

def plot_distance_time_series_bokeh_datashader(distances_df, collision_df, t_col, sat_name, arc, fm_num, t_window=10, hbr=12.0, folder="output/Collisions/MC"):
    # Check and convert 'UTC' column to datetime format
    if not pd.to_datetime(distances_df['UTC'], errors='coerce').isnull().all():
        distances_df['UTC'] = pd.to_datetime(distances_df['UTC'], errors='coerce')
    else:
        raise ValueError("UTC column cannot be converted to datetime format")

    if not pd.to_datetime(collision_df['UTC'], errors='coerce').isnull().all():
        collision_df['UTC'] = pd.to_datetime(collision_df['UTC'], errors='coerce')
    else:
        raise ValueError("UTC column in collision_df cannot be converted to datetime format")

    t_col = pd.to_datetime(t_col)
    time_lower = t_col - datetime.timedelta(seconds=t_window)
    time_upper = t_col + datetime.timedelta(seconds=t_window)

    # Filter data within the time window
    distances_df_filtered = distances_df[(distances_df['UTC'] >= time_lower) & (distances_df['UTC'] <= time_upper)]
    collision_df_filtered = collision_df[(collision_df['UTC'] >= time_lower) & (collision_df['UTC'] <= time_upper)]

    # Melt the DataFrame for easier plotting with Datashader
    distances_df_melted = distances_df_filtered.melt(id_vars=['UTC'], var_name='Simulation', value_name='Distance')
    hv_dist = hv.Dataset(distances_df_melted, ['UTC', 'Simulation'], 'Distance')
    hv_collision = hv.Dataset(collision_df_filtered, ['UTC'], ['distance'])

    # Datashading
    curve = datashade(hv.Curve(hv_dist, 'UTC', 'Distance'), dynamic=False)
    collision_curve = datashade(hv.Curve(hv_collision, 'UTC', 'distance'), dynamic=False, color='green')

    # Create and combine plots
    hline = hv.HLine(hbr).opts(color='red', line_dash='dashed')
    plot = curve * collision_curve * hline

    # Render and save the plot
    renderer = hv.renderer('bokeh').instance(mode='server')
    plot_state = renderer.get_plot(plot).state

    if not os.path.exists(folder):
        os.makedirs(folder)
    timenow = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    output_file(f"{folder}/MC_{sat_name}_arc_{arc}_FM_{fm_num}_{timenow}_datashader.html")
    save(plot_state)
def plot_distance_time_series_bokeh(distances_df, collision_df, t_col, sat_name, arc, fm_num, t_window=10, hbr=12.0, folder="output/Collisions/MC"):
    print(f"Plotting time series for {sat_name} arc {arc} FM {fm_num}")

    # Ensure the correct attributes are used: width and height
    p = figure(title=f"+/- {t_window} Seconds from TCA for {sat_name} arc {arc} FM {fm_num}",
               x_axis_label='Time', y_axis_label='Distance', x_axis_type='datetime',
               y_axis_type="log", width=700, height=400)
    
    # Set the x-range to display around t_col
    p.x_range = Range1d(t_col - datetime.timedelta(seconds=t_window), t_col + datetime.timedelta(seconds=t_window))

    time_lower = t_col - datetime.timedelta(seconds=t_window)
    time_upper = t_col + datetime.timedelta(seconds=t_window)
    # Filter distances_df to the specified time window around t_col
    distances_df = distances_df[distances_df['UTC'].iloc[:, 0].between(time_lower, time_upper)]
    
    num_below_hbr = 0
    for column in distances_df.columns:
        if column != 'UTC':
            print(f"Plotting {column}")
            p.line(distances_df['UTC'], distances_df[column], legend_label=column, line_width=0.2, color='blue', alpha=1)
            if distances_df[column].min() < hbr:
                num_below_hbr += 1
    
    # Plot original distance data from collision_df
    p.line(collision_df['UTC'], collision_df['distance'], legend_label='Original Distance', line_dash='dotted', line_color='green')
    
    # Horizontal line for HBR
    hbr_line = Span(location=hbr, dimension='width', line_color='red', line_dash='dashed', line_width=1)
    p.add_layout(hbr_line)

    # Add labels for HBR and number of samples
    below_hbr_percentage = round((num_below_hbr / (len(distances_df.columns) - 1)) * 100, 2)
    collision_label = Label(x=0.25, y=0.5, text=f'% collision: {below_hbr_percentage}%', x_units='data', y_units='data')
    sample_label = Label(x=0.25, y=0.45, text=f'No. of Samples: {len(distances_df.columns) - 1}', x_units='data', y_units='data')
    p.add_layout(collision_label)
    p.add_layout(sample_label)

    if not os.path.exists(folder):
        os.makedirs(folder)
    timenow = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    output_file(f"{folder}/MC_{sat_name}_arc_{arc}_FM_{fm_num}_{timenow}.html")
    # export_png(p, filename=f"{folder}/MC_{sat_name}_arc_{arc}_FM_{fm_num}_{timenow}_bokeh.png")
    show(p)

def plot_distance_time_series_heatmap_bokeh(distances_df, collision_df, t_col, sat_name, arc, fm_num, t_window=20, hbr=12.0, folder="output/Collisions/MC"):
    print(f'Plotting heatmap for {sat_name} arc {arc} FM {fm_num}')
    t_col = pd.to_datetime(t_col) if isinstance(t_col, str) else t_col
    time_lower = t_col - datetime.timedelta(seconds=t_window)
    time_upper = t_col + datetime.timedelta(seconds=t_window)
    filtered_df = distances_df[distances_df['UTC'].iloc[:, 0].between(time_lower, time_upper)]

    utc_seconds = (filtered_df['UTC'].iloc[:, 0] - t_col).dt.total_seconds().to_numpy()

    x_flattened = []
    y_flattened = []
    colors = []
    num_distance_columns = len([col for col in filtered_df.columns if 'Distance' in col])

    for i, row in filtered_df.iterrows():
        utc_second = utc_seconds[i - filtered_df.index[0]]  # Adjust index offset
        for j in range(num_distance_columns):  
            column_name = f'Distance_{j}'
            if pd.notnull(row[column_name]):
                x_flattened.append(float(utc_second))
                y_flattened.append(float(row[column_name]))
                colors.append(float(row[column_name]))

    mapper = LinearColorMapper(palette="Plasma256", low=min(colors), high=max(colors))
    p = figure(title=f"+/- {t_window} Seconds from TCA", x_axis_label='Time (s from TCA)', y_axis_label='Distance', y_axis_type="log", width=700, height=400, tools="pan,wheel_zoom,box_zoom,reset", toolbar_location="above")
    p.hexbin(x=np.array(x_flattened), y=np.array(y_flattened), size=0.1, fill_color={'field': 'colors', 'transform': mapper}, line_color=None)

    color_bar = ColorBar(color_mapper=mapper, label_standoff=12, location=(0,0), ticker=LogTicker())
    p.add_layout(color_bar, 'right')

    hbr_line_x = [min(x_flattened), max(x_flattened)]
    p.line(hbr_line_x, [hbr, hbr], line_color="red", line_dash="dashed", legend_label="HBR")

    if not os.path.exists(folder):
        os.makedirs(folder)
    output_file(f"{folder}/MC_{sat_name}_arc_{arc}_FM_{fm_num}_heatmap.html")
    export_png(column(p), filename=f"{folder}/MC_{sat_name}_arc_{arc}_FM_{fm_num}_heatmap.png")
    show(p)


def plot_distance_time_series_heatmap(distances_df, collision_df, t_col, sat_name, arc, fm_num, t_window=20, hbr=12.0, folder="output/Collisions/MC"):
    print(f'Plotting heatmap for {sat_name} arc {arc} FM {fm_num}')
    print(f"Memory usage before plotting (in MB): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}")
    plt.figure(figsize=(7, 4))
    sns.set_theme(style="whitegrid")

    t_col = pd.to_datetime(t_col) if isinstance(t_col, str) else t_col

    time_lower = t_col - datetime.timedelta(seconds=t_window)
    time_upper = t_col + datetime.timedelta(seconds=t_window)

    filtered_df = distances_df[distances_df['UTC'].iloc[:, 0].between(time_lower, time_upper)]

    x_flattened = []
    y_flattened = []
    utc_column = filtered_df['UTC']
    x = mdates.date2num(utc_column)

    for column in filtered_df.columns:
        if 'Distance' in column:
            distance_values = filtered_df[column].values
            x_flattened.extend(x)
            y_flattened.extend(distance_values)

    # Reduced gridsize for less memory usage
    hb = plt.hexbin(x_flattened, y_flattened, gridsize=300, cmap='plasma', mincnt=1, xscale='linear', yscale='log')
    cb = plt.colorbar(hb, label='Number of Data Points')

    plt.axhline(y=hbr, color='r', linestyle='--', label='HBR')
    plt.title(f'+/- {t_window} Seconds from TCA')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.yscale('log')
    plt.ylim(0.01, 10e6)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if not os.path.exists(folder):
        os.makedirs(folder)
    timenow = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    plt.savefig(f"{folder}/MC_{sat_name}_arc_{arc}_FM_{fm_num}_{timenow}_heatmap.png", dpi=600)
    plt.close()
    print(f"Memory usage after plotting (in MB): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}")

def plot_distance_time_series(distances_df, collision_df, t_col, sat_name, arc, fm_num, t_window=10, hbr=12.0, folder="output/Collisions/MC"):
    print(f"Plotting time series for {sat_name} arc {arc} FM {fm_num}")
    plt.figure(figsize=(7, 4))
    sns.set_theme(style="whitegrid")
    plt.xlim(t_col - datetime.timedelta(seconds=t_window), t_col + datetime.timedelta(seconds=t_window))
    #slice the dataframe to only include the time window around the TCA
    distances_df = distances_df[distances_df['UTC'].between(t_col - datetime.timedelta(seconds=t_window), t_col + datetime.timedelta(seconds=t_window))]
    num_below_hbr = 0
    for column in distances_df.columns:
        if column != 'UTC':
            print(f"Plotting {column}")
            plt.plot(distances_df['UTC'], distances_df[column], label=column, alpha=1, linewidth=0.2, c='xkcd:blue')
            #if the minimum distance is below the HBR, increment the counter
            if distances_df[column].min() < hbr:
                num_below_hbr += 1
    print(f"num_below_hbr: {num_below_hbr}")
    plt.plot(collision_df['UTC'], collision_df['distance'], label='Original Distance', linestyle='dotted')
    plt.title('+/- 2Min from TCA')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.yscale('log')
    plt.ylim(0.01, 10e6)
    plt.axhline(y=hbr, color='r', linestyle='--', label='HBR')
    #as text, on the HBR line, write the percentage of the time series that is below the HBR
    below_hbr = num_below_hbr / (len(distances_df.columns) - 1)
    percentage_below_hbr = round(below_hbr * 100, 2)
    plt.text(0.25, 0.5, f" %collsion: {percentage_below_hbr}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.25, 0.45, f"No. of Samples: {len(distances_df.columns) - 1}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.xticks(rotation=45)
    #calculate the percentage of the time series that is below the HBR
    num_below_hbr = len([d for d in collision_df['distance'] if d < hbr])
    #plot the finer y axis ticks and lines
    plt.yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
    plt.tight_layout()
    if not os.path.exists(folder):
        os.makedirs(folder)
    timenow = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    plt.savefig(f"{folder}/MC_{sat_name}_arc_{arc}_FM_{fm_num}_{timenow}.png")

def plot_minimum_distance_histogram(min_distances, sat_name, arc, fm_num, folder="output/Collisions/MC"):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    plt.hist(min_distances, bins=20, color='blue', edgecolor='black')
    plt.title('Minimum Distance Histogram')
    plt.xlabel('Minimum Distance')
    plt.ylabel('Frequency')
    thresholds = [1000, 100, 10, 5, 1]
    for i, threshold in enumerate(thresholds, start=1):
        num_below_threshold = len([d for d in min_distances if d < threshold])
        plt.text(0.5, 1 - 0.05*i, f"Below {threshold}m: {num_below_threshold}/{len(min_distances)}", ha='center', transform=plt.gca().transAxes)
    if not os.path.exists(folder):
        os.makedirs(folder)
    timenow = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    plt.savefig(f"{folder}/hist_TCA_{sat_name}_arc_{arc}_FM_{fm_num}_{timenow}.png")

if __name__ == "__main__":
    main()
    #TODO: add a functuion to use the exisiting interpolated ephemeris if available