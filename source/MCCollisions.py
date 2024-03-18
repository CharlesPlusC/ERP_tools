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
                perturbed_states_primary = generate_perturbed_states(optimized_state_cov, optimized_state, 25)

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
                print("number of columns in distances_df: ", len(distances_df.columns) - 1)
                min_distances = []
                # Set the plot size
                plt.figure(figsize=(10, 6))
                #plot only 2min before and after the collision time
                plt.xlim(t_col - datetime.timedelta(minutes=1), t_col + datetime.timedelta(minutes=1))
                # Loop through each distance column (ignoring the 'UTC' column) to plot
                for column in distances_df.columns:
                    if column != 'UTC':
                        plt.plot(distances_df['UTC'], distances_df[column], label=column)
                        #print the minimum distance for each perturbed state
                        min_distance = distances_df[column].min()
                        min_distances.append(min_distance)

                print("DCAs: ", min_distances)
                print(f"closest distance: {min(min_distances)}")
                print(f"simulation end. Total trajectories: {len(min_distances)}")
                plt.plot(col_to_ephem_distances['UTC'], col_to_ephem_distances['distance'], label='Original Distance', linestyle='dotted')
                # Set plot title and labels
                plt.title('Distance Time Series')
                plt.xlabel('Time')
                plt.ylabel('Distance')
                plt.yscale('log')
                #make the y axis start at 0m
                plt.ylim(0.01, 10e7)
                #put a horizontal line at 
                # Rotate date labels for better readability
                plt.xticks(rotation=45)
                # Show the plot
                plt.tight_layout()  # Adjust layout to not cut off labels
        
                folder = "output/Collisions/MC"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                timenow = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                plt.savefig(f"{folder}/MC_{sat_name}_arc_{arc}_FM_{fm_num}_sample_{timenow}.png")
                # plt.show()

                #plot a histogram of the minimum distances
                plt.figure(figsize=(10, 6))
                plt.hist(min_distances, bins=20)
                plt.title('Minimum Distance Histogram')
                plt.xlabel('Minimum Distance')
                plt.ylabel('Frequency')
                #in text write how many out of how many were below 1km, 100m, 10m, 5m and 1m
                num_below_1km = len([d for d in min_distances if d < 1000])
                num_below_100m = len([d for d in min_distances if d < 100])
                num_below_10m = len([d for d in min_distances if d < 10])
                num_below_5m = len([d for d in min_distances if d < 5])
                num_below_1m = len([d for d in min_distances if d < 1])
                smallest_distance = min(min_distances)
                plt.text(0.5, 0.9, f"Below 1km: {num_below_1km}/{len(min_distances)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                plt.text(0.5, 0.85, f"Below 100m: {num_below_100m}/{len(min_distances)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                plt.text(0.5, 0.8, f"Below 10m: {num_below_10m}/{len(min_distances)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                plt.text(0.5, 0.75, f"Below 5m: {num_below_5m}/{len(min_distances)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                plt.text(0.5, 0.7, f"Below 1m: {num_below_1m}/{len(min_distances)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                plt.text(0.5, 0.65, f"Smallest Distance: {smallest_distance}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                plt.savefig(f"{folder}/hist_TCA_{sat_name}_arc_{arc}_FM_{fm_num}_sample_{timenow}.png")
                # plt.show()
                print(f"simulation end. Total trajectories: {len(min_distances)}")
                plot_distance_time_series(distances_df, col_to_ephem_distances, t_col, sat_name, arc, fm_num)
                plot_distance_time_series_heatmap(distances_df, col_to_ephem_distances, t_col, sat_name, arc, fm_num)

import matplotlib.dates as mdates
import matplotlib.colors as colors
def plot_distance_time_series_heatmap(distances_df, collision_df, t_col, sat_name, arc, fm_num, t_window=20, hbr=12.0, folder="output/Collisions/MC"):
    plt.figure(figsize=(7, 4))
    sns.set_theme(style="whitegrid")

    if isinstance(t_col, str):
        t_col = pd.to_datetime(t_col)

    time_lower = t_col - datetime.timedelta(seconds=t_window)
    time_upper = t_col + datetime.timedelta(seconds=t_window)

    filtered_df = distances_df[distances_df['UTC'].iloc[:, 0].between(time_lower, time_upper)]

    x_flattened = []
    y_flattened = []
    utc_column = filtered_df['UTC'].iloc[:, 0]
    x = mdates.date2num(utc_column)

    # Calculate the number of Distance_i columns
    num_distance_columns = sum('Distance' in col for col in filtered_df.columns)

    for i in range(num_distance_columns):
        distance_column = filtered_df[f'Distance_{i}']
        x_flattened.extend(x)
        y_flattened.extend(distance_column)

    # Plot using hexbin
    hb = plt.hexbin(x_flattened, y_flattened, gridsize=1000, cmap='plasma', mincnt=1, xscale='linear', yscale='log')
    cb = plt.colorbar(hb, label='Number of Data Points')

    # Define the colorbar ticks based on the number of Distance_i columns
    tick_interval = max(1, num_distance_columns // 5)  # Aim for up to 5 ticks
    tick_values = range(0, num_distance_columns + 1, tick_interval)
    cb.set_ticks(tick_values)
    cb.set_ticklabels([str(val) for val in tick_values])

    # Set other plot features
    plt.axhline(y=hbr, color='r', linestyle='--', label='HBR')
    plt.text(0.2, hbr, f"HBR: {hbr}", horizontalalignment='center', transform=plt.gca().get_yaxis_transform())
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

def plot_distance_time_series(distances_df, collision_df, t_col, sat_name, arc, fm_num, t_window=20, hbr=12.0, folder="output/Collisions/MC"):
    plt.figure(figsize=(7, 4))
    sns.set_theme(style="whitegrid")
    plt.xlim(t_col - datetime.timedelta(seconds=t_window), t_col + datetime.timedelta(seconds=t_window))
    num_below_hbr = 0
    for column in distances_df.columns:
        if column != 'UTC':
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