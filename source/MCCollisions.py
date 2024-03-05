import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit, PositionAngleType
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from orekit import JArray_double
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants
from org.orekit.orbits import PositionAngleType, OrbitType
from org.orekit.utils import IERSConventions, PVCoordinates
from org.hipparchus.linear import MatrixUtils
from org.orekit.propagation import StateCovariance
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.ssa.collision.shorttermencounter.probability.twod import Patera2005

import os
from tools.utilities import get_satellite_info, pos_vel_from_orekit_ephem
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import configure_force_models, propagate_state, propagate_STM
from tools.BatchLeastSquares import OD_BLS
from tools.collision_tools import generate_collision_trajectory
import numpy as np
import numpy.random as npr
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 60.0
POSITION_TOLERANCE = 1e-2 # 1 cm

#covariance associated with the collision trajectory (assuming good covariance)
secondary_covariance = [
    [4.7440894789163000000000, -1.2583279067770000000000, -1.2583279067770000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [-1.2583279067770000000000,6.1279552605419000000000, 2.1279552605419000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [-1.2583279067770000000000, 2.1279552605419000000000, 6.1279552605419000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000010000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000001, 0.0000010000000000000000, -0.0000000000000000000001],
    [0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000001, -0.0000000000000000000001, 0.0000010000000000000000]
]

def generate_random_vectors(eigenvalues):
    random_vectors = []
    for lambda_val in eigenvalues:
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

def generate_perturbed_states(optimized_state_cov, state):
    eig_vals, rotation_matrix = np.linalg.eigh(optimized_state_cov)
    random_vectors = generate_random_vectors(eig_vals)
    perturbed_states = apply_perturbations([state], [random_vectors], [rotation_matrix])
    return perturbed_states

def main():
    p_o_c_list = []
    sat_names_to_test = ["GRACE-FO-A"]
    arc_length = 25  # mins
    num_arcs = 1
    prop_length = 60 * 60 * 1.5  # around 2 orbital periods
    prop_length_days = prop_length / (60 * 60 * 24)
    force_model_configs = [{'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'jb08drag': True}]

    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        sat_info = get_satellite_info(sat_name)
        cd = sat_info['cd']
        cr = sat_info['cr']
        cross_section = sat_info['cross_section']
        mass = sat_info['mass']
        ephemeris_df = ephemeris_df.iloc[::2, :] # downsample to 60 second intervals
        time_step = (ephemeris_df['UTC'].iloc[1] - ephemeris_df['UTC'].iloc[0]).total_seconds() / 60.0  # in minutes
        # find the time of the first time step
        t0 = ephemeris_df['UTC'].iloc[0]
        print(f't0: {t0}')
        t_col = t0 + datetime.timedelta(days=prop_length_days) #this is the time at which the collision will occur
        #end of propagation window is 15 minutes after the collision
        t_end = t0 + datetime.timedelta(days=prop_length_days + 15/(60*24))
        collision_df = generate_collision_trajectory(ephemeris_df, t_col)
        
        arc_step = int(arc_length / time_step)
        for arc in range(num_arcs):
            
            start_index = arc * arc_step
            end_index = start_index + arc_step
            arc_df = ephemeris_df.iloc[start_index:end_index]

            observations_df = arc_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
            initial_values = arc_df.iloc[0][['x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
            initial_t = arc_df.iloc[0]['UTC']
            initial_vals = np.array(initial_values.tolist() + [cd, cr, cross_section, mass], dtype=float)

            a_priori_estimate = np.concatenate(([initial_t.timestamp()], initial_vals))

            for i, force_model_config in enumerate(force_model_configs):
                optimized_states, cov_mats, residuals, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag=False, max_patience=1)
                min_RMS_index = np.argmin(RMSs)
                optimized_state = optimized_states[min_RMS_index]
                optimized_state_cov = cov_mats[min_RMS_index]

            print("optimized_state_cov: ", optimized_state_cov)

            ##### Perturb the estimated state ("primary state") and propagate those perturbed states
            #make a list of dataframes for each perturbed state
            primary_states_perturbed_ephem = []
            perturbed_states_primary = generate_perturbed_states(optimized_state_cov, optimized_state)
            for primary_state in perturbed_states_primary:
                print(f"propagating primary_state: {primary_state}")
                print(f"force_model_config: {force_model_config}")
                print(f"from t0: {t0}")
                print(f"to t_end: {t_end}")
                primary_state_perturbed_df = propagate_state(start_date=t0, end_date=t_end, initial_state_vector=primary_state, cr=cr, cd=cd, cross_section=cross_section, mass=mass,boxwing=None,ephem=True, **force_model_config)
                primary_states_perturbed_ephem.append(primary_state_perturbed_df)
            print(f"prop perturbed primary states: {primary_states_perturbed_ephem}")

            secondary_state = collision_df.iloc[0][["x_col", "y_col", "z_col", "xv_col", "yv_col", "zv_col"]].values
            secondary_states_perturbed_ephem = []
            perturbed_states_secondary = generate_perturbed_states(secondary_covariance, secondary_state)
            for secondary_state in perturbed_states_secondary:
                secondary_states_perturbed_df = propagate_state(start_date=t0, end_date=t_end, initial_state_vector=secondary_state, cr=cr, cd=cd, cross_section=cross_section, mass=mass,boxwing=None,ephem=True, **force_model_config)
                secondary_states_perturbed_ephem.append(secondary_states_perturbed_df)

            #go through the list of dataframes and calculate the distance between every combination of primary and secondary states
            #make a dataframe of the n1 by n2 distances
            # Initialize a list to hold all distance dataframes
            distance_dfs = []

            # Iterate through each primary ephemeris dataframe
            for i, primary_state_perturbed_df in enumerate(primary_states_perturbed_ephem):
                # Iterate through each secondary ephemeris dataframe
                for j, secondary_state_perturbed_df in enumerate(secondary_states_perturbed_ephem):
                    # Calculate the Euclidean distance for the position vectors (x, y, z) at each time point
                    distances = np.linalg.norm(primary_state_perturbed_df[['x', 'y', 'z']].values - secondary_state_perturbed_df[['x', 'y', 'z']].values, axis=1)
                    # Create a dataframe with distances and corresponding UTC timestamps
                    distance_df = pd.DataFrame({'UTC': primary_state_perturbed_df['UTC'], f'Distance_{i}_{j}': distances})
                    distance_dfs.append(distance_df)

            # Concatenate all individual distance dataframes to get a single dataframe with all distances
            distances_df = pd.concat(distance_dfs, axis=1)

            # Removing duplicate UTC columns if they exist
            distances_df = distances_df.loc[:,~distances_df.columns.duplicated()]

            #calculate the distance between unperturbed primary and secondary states
            # get the subset of the ephemeris_df that is within the time window of the collision_df
            ephemeris_df = ephemeris_df[(ephemeris_df['UTC'] >= collision_df['UTC'].iloc[0]) & (ephemeris_df['UTC'] <= collision_df['UTC'].iloc[-1])]
            collision_distances = np.linalg.norm(ephemeris_df[['x', 'y', 'z']].values - collision_df[['x_col', 'y_col', 'z_col']].values, axis=1)
            #add it to a dataframe time stamped with the UTC
            collision_df['distance'] = collision_distances

            # Set the plot size
            plt.figure(figsize=(10, 6))

            # Loop through each distance column (ignoring the 'UTC' column) to plot
            for column in distances_df.columns:
                if column != 'UTC':
                    plt.plot(distances_df['UTC'], distances_df[column], label=column)
                    #print the minimum distance for each perturbed state
                    min_distance = distances_df[column].min()
                    print(f"min_distance: {min_distance}")

            #add the original distance to the plot with dotted line
            plt.plot(collision_df['UTC'], collision_df['distance'], label='Original Distance', linestyle='dotted')

            # Set plot title and labels
            plt.title('Distance Time Series')
            plt.xlabel('Time')
            plt.ylabel('Distance')
            plt.yscale('log')
            # Rotate date labels for better readability
            plt.xticks(rotation=45)

            # Enable legend
            plt.legend()

            # Show the plot
            plt.tight_layout()  # Adjust layout to not cut off labels
            plt.show()

                # calculate the DCA between the collision_df and the perturbed states
                # plot the DCA for each perturbed state

                #calculate the distance between every combination of primary and secondary states
                # calculate the DCA between the collision_df and the perturbed states
                # plot the DCA for each perturbed state
            #calculate the distance between every combination of primary and secondary states


                
                # calculate the DCA between the collision_df and the perturbed states
                # plot the DCA for each perturbed state

            # propagate all the perturbed states
            # calculate the DCA between the collision_df and the perturbed states
            # plot the DCA for each perturbed state

        # Replace the usage of optimized_state with each of perturbed_states_primary in further computations
        # such as propagating states, computing distances, and calculating the probability of collision.

    print(p_o_c_list)

if __name__ == "__main__":
    main()