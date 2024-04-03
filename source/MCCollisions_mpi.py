
from mpi4py import MPI
import orekit
from orekit.pyhelpers import setup_orekit_curdir
import os
import numpy as np
import numpy.random as npr
import datetime
import pandas as pd
from scipy.interpolate import interp1d
from tools.utilities import get_satellite_info
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import propagate_state
from tools.BatchLeastSquares import OD_BLS
from tools.collision_tools import generate_collision_trajectory

vm = orekit.initVM()

def create_and_submit_job_script(sat_name, force_model_config, arc, job_id):
    script_content = f"""#!/bin/bash -l
    #$ -l h_rt=1:0:0
    #$ -l mem=1G
    #$ -l tmpfs=15G
    #$ -N MCC_{sat_name}{job_id}
    #$ -wd /home/$USER/Scratch/MCCollisions/{sat_name}/{job_id}
    cd $TMPDIR
    cp /home/$USER/path/to/your_script.py $TMPDIR
    cp /home/$USER/path/to/misc/orekit-data.zip $TMPDIR
    module load python/3.7
    python your_script.py {sat_name} {force_model_config} {arc}
    cp * /home/$USER/Scratch/MCCollisions/{sat_name}/{job_id}/
    """
    script_filename = f"job{sat_name}_{job_id}.sh"
    with open(script_filename, 'w') as file:
        file.write(script_content)
    os.system(f"qsub {script_filename}")

def interpolate_ephemeris(df, start_time, end_time, freq='0.0001S', stitch=False):
    df = df.drop_duplicates(subset='UTC').set_index('UTC')
    df = df.sort_index()
    df_resampled = df.reindex(pd.date_range(start=start_time, end=end_time, freq=freq), method='nearest').asfreq(freq)
    interp_funcs = {col: interp1d(df.index.astype(int), df[col], fill_value='extrapolate') for col in ['x', 'y', 'z']}
    for col in ['x', 'y', 'z']:
    df_resampled[col] = interp_funcscol
    df_filtered = df_resampled.loc[start_time:end_time].reset_index().rename(columns={'index': 'UTC'})
    if stitch:
    df_stitched = pd.concat([df.loc[:start_time - pd.Timedelta(freq), :], df_filtered.set_index('UTC'), df.loc[end_time + pd.Timedelta(freq):, :]]).reset_index()
    return df_stitched
    return df_filtered

def generate_random_vectors(eigenvalues, num_samples):
return [npr.normal(0, np.sqrt(lambda_val), 6) for lambda_val in eigenvalues for _ in range(num_samples)]

def apply_perturbations(states, vectors, rotation_matrices):
return [state + np.dot(rotation_matrix, vector) for state, vector_set, rotation_matrix in zip(states, vectors, rotation_matrices) for vector in vector_set]

def generate_perturbed_states(optimized_state_cov, state, num_samples):
eig_vals, rotation_matrix = np.linalg.eigh(optimized_state_cov)
random_vectors = generate_random_vectors(eig_vals, num_samples)
return apply_perturbations([state], [random_vectors], [rotation_matrix])

def main():
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

python
Copy code
sat_names_to_test = ["GRACE-FO-A"]
force_model_configs = [{'120x120gravity': True, '3BP': True}, {'120x120gravity': True, '3BP': True, 'SRP': True, 'nrlmsise00drag': True}, {'120x120gravity': True, '3BP': True, 'SRP': True, 'jb08drag': True}]
num_arcs = 1

MC_ephem_folder = "output/Collisions/MC/interpolated_MC_ephems"
if rank == 0 and not os.path.exists(MC_ephem_folder):
    os.makedirs(MC_ephem_folder)
comm.Barrier()

job_id = 0
for i, sat_name in enumerate(sat_names_to_test):
    for j, force_model_config in enumerate(force_model_configs):
        for k in range(num_arcs):
            if job_id % size == rank:
                create_and_submit_job_script(sat_name, j, k, job_id)
            job_id += 1
if name == "main":
main()