import orekit
from orekit.pyhelpers import setup_orekit_curdir
import os
import numpy as np
import numpy.random as npr
import datetime
import pandas as pd
from scipy.interpolate import interp1d
from tools.utilities import get_satellite_info, load_force_model_configs
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import propagate_state
from tools.BatchLeastSquares import OD_BLS
from tools.collision_tools import generate_collision_trajectory
import json
vm = orekit.initVM()

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
    # Ensure that only num_samples vectors are generated in total
    random_vectors = []
    for _ in range(num_samples):
        vector = [npr.normal(0, np.sqrt(lambda_val)) for lambda_val in eigenvalues]
        random_vectors.append(vector)
    return random_vectors

def apply_perturbations(state, vectors, rotation_matrix):
    # Apply perturbations to the state
    perturbed_states = []
    for vector in vectors:
        perturbation = np.dot(rotation_matrix, vector)
        perturbed_state = state + perturbation
        perturbed_states.append(perturbed_state)
    return perturbed_states

def generate_perturbed_states(optimized_state_cov, state, num_samples):
    # Generate perturbed states based on the covariance matrix
    eig_vals, rotation_matrix = np.linalg.eigh(optimized_state_cov)
    random_vectors = generate_random_vectors(eig_vals, num_samples)
    perturbed_states = apply_perturbations(state, random_vectors, rotation_matrix)
    return perturbed_states

def generate_perturbed_states(optimized_state_cov, state, num_samples):
    # Generate perturbed states based on the covariance matrix
    eig_vals, rotation_matrix = np.linalg.eigh(optimized_state_cov)
    random_vectors = generate_random_vectors(eig_vals, num_samples)
    perturbed_states = apply_perturbations(state, random_vectors, rotation_matrix)
    return perturbed_states

def create_and_submit_job_scripts(sat_name, fm_num, num_perturbations):
    import os

    user_home_dir = os.path.expanduser("~")

    folder_for_jobs = f"{user_home_dir}/Scratch/MCCollisions/sge_jobs"
    output_folder = f"{user_home_dir}/Scratch/MCCollisions/{sat_name}/results_fm{fm_num}"
    logs_folder = f"{user_home_dir}/Scratch/MCCollisions/{sat_name}/logs_fm{fm_num}"
    work_dir = f"{user_home_dir}/Scratch/MCCollisions/{sat_name}/propagation_fm{fm_num}"

    os.makedirs(folder_for_jobs, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Creating one script for all perturbed states as an array job
    script_filename = f"{folder_for_jobs}/prop_fm{fm_num}_{sat_name}.sh"

    script_content = f"""#!/bin/bash -l
#$ -l h_rt=3:0:0
#$ -l mem=6G
#$ -l tmpfs=10G
#$ -N Prop_fm{fm_num}_{sat_name}
#$ -t 1-{num_perturbations}
#$ -wd {work_dir}
#$ -o {logs_folder}/out_$TASK_ID.txt
#$ -e {logs_folder}/err_$TASK_ID.txt

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
cp {user_home_dir}/mc_collisions/ERP_tools/erp_tools_env.yml $TMPDIR/erp_tools_env.yml

cp -r {user_home_dir}/mc_collisions/ERP_tools $TMPDIR

cd $TMPDIR/ERP_tools

# The TASK_ID environment variable is used to differentiate between array job tasks
/home/{os.getenv('USER')}/.conda/envs/erp_tools_env/bin/python $TMPDIR/ERP_tools/source/individual_MC_job.py {sat_name} {fm_num} $SGE_TASK_ID {output_folder}/{sat_name}_fm{fm_num}_perturbed_state${{SGE_TASK_ID}}_distances.csv
"""
    with open(script_filename, 'w') as file:
        file.write(script_content)

    os.system(f"qsub {script_filename}")

def generate_nominal_and_perturbed_states(sat_name, num_perturbations=20):

    ## Define Force Model Configurations To Test
    force_model_configs = load_force_model_configs('misc/fm_configs.json')

    ## Load the nominal ephemeris and satellite information
    ephemeris_df = sp3_ephem_to_df(sat_name)
    sat_info = get_satellite_info(sat_name)
    cd = sat_info['cd']
    cr = sat_info['cr']
    cross_section = sat_info['cross_section']
    mass = sat_info['mass']
    t0 = ephemeris_df['UTC'].iloc[0]

    ## Generate Collision Trajectory and Interpolate around nominal TCA
    t_col = t0 + datetime.timedelta(hours=12)  
    collision_df = generate_collision_trajectory(ephemeris_df, t_col, dt=5.0, post_col_t=15.0)
    collision_df_interp = interpolate_ephemeris(collision_df, t_col - datetime.timedelta(seconds=7), t_col + datetime.timedelta(seconds=7), stitch=True) #interpolate the collision trajectory very finely around the TCA
    output_folder = f"output/Collisions/MC/interpolated_MC_ephems/{sat_name}"
    os.makedirs(output_folder, exist_ok=True)
    collision_df_interp.to_csv(f"{output_folder}/{sat_name}_nominal_collision.csv", index=False)

    #slice ephemeris df to be every other row (minutely data) and take only the first 35 minutes
    observations_df = ephemeris_df.iloc[::2].head(35)
    init_pos_vel_sigmas = np.array(observations_df[['x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']].iloc[0].tolist(), dtype=float)
    initial_vals = np.array(init_pos_vel_sigmas.tolist() + [cd, cr, cross_section, mass], dtype=float)
    initial_t = observations_df['UTC'].iloc[0]
    a_priori_estimate = np.concatenate(([initial_t.timestamp()], initial_vals))

    for fm_num, force_model_config in enumerate(force_model_configs):
        optimized_states, cov_mats, _, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag=False, max_patience=1)
        min_RMS_index = np.argmin(RMSs)
        optimized_state = optimized_states[min_RMS_index]
        optimized_state_cov = cov_mats[min_RMS_index]
        primary_states_perturbed_ephem = []
        perturbed_states_primary = generate_perturbed_states(optimized_state_cov, optimized_state, num_perturbations)
        primary_states_perturbed_ephem.extend(perturbed_states_primary)
        perturbed_states_file = f"{output_folder}/{sat_name}_fm{fm_num}_perturbed_states.csv"
        np.savetxt(perturbed_states_file, primary_states_perturbed_ephem, delimiter=",")
        create_and_submit_job_scripts(sat_name, fm_num, num_perturbations)

def main():
    sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    num_perturbations = 10000

    for sat_name in sat_names_to_test:
        generate_nominal_and_perturbed_states(sat_name, num_perturbations)

if __name__ == "__main__":
    main()
