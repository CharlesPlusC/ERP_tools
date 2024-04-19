# from tools.sp3_2_ephemeris import sp3_ephem_to_df
# from ForceModelBenchmark import benchmark
# import os

# def setup_job_inputs():
#     print("force model benchmarking script")
#     sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
#     dates_to_test = ["2019-01-01", "2023-05-04"]
#     num_arcs = 10 #number of OD arcs that will be run
#     arc_length = 25 # length of each arc in minutes
#     prop_length = 60 * 60 * 12 # length of propagation in seconds
#     force_model_configs = [
#         {'36x36gravity': True, '3BP': True},
#         {'120x120gravity': True, '3BP': True},
#         {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True},
#         {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True},
#         {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True},
#         {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},
#         {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'jb08drag': True},
#         {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'dtm2000drag': True},
#         {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'nrlmsise00drag': True}
#     ]

#     for force_model_config in force_model_configs:
#         print(f"force model config: {force_model_config}")
#         for sat_name in sat_names_to_test:
#             for date in dates_to_test:
#                 ephemeris_df = sp3_ephem_to_df(sat_name, date)
#                 # take every other row of the ephemeris to get to 1-minute intervals
#                 ephemeris_df = ephemeris_df.iloc[::2]
#                 for arc_number in range(num_arcs):
#                     # slice the ephemeris into the observations arc
#                     start_index = arc_number * arc_length
#                     end_index = start_index + arc_length
#                     OD_points = ephemeris_df.iloc[start_index:end_index]
#                     # Now the OP arc starts at the same time as the OD arc, but ends prop_length seconds later
#                     prop_end_index = start_index + prop_length // 60  # convert seconds to minutes
#                     prop_end_index = min(prop_end_index, len(ephemeris_df) - 1)  # ensure we do not go out of bounds
#                     OP_reference_trajectory = ephemeris_df.iloc[start_index:prop_end_index]
#                     print(f"OP arc head: {OP_reference_trajectory.head()}")
#                     print(f"time difference between first and last UTC stamp: {OP_reference_trajectory.iloc[-1]['UTC'] - OP_reference_trajectory.iloc[0]['UTC']}")
#                     print(f"selected force model config: {force_model_config}")
#                     benchmark(sat_name, OD_points, OP_reference_trajectory, prop_length, arc_number, force_model_config)

# def create_and_submit_job_scripts(sat_name, fm_num, num_arcs):

#     user_home_dir = os.path.expanduser("~")

#     folder_for_jobs = f"{user_home_dir}/Scratch/FMBenchmark/sge_jobs"
#     logs_folder = f"{user_home_dir}/Scratch/FMBenchmark/{sat_name}/logs_fm{fm_num}"
#     work_dir = f"{user_home_dir}/Scratch/FMBenchmark/{sat_name}/propagation_fm{fm_num}"

#     os.makedirs(folder_for_jobs, exist_ok=True)
#     os.makedirs(logs_folder, exist_ok=True)
#     os.makedirs(work_dir, exist_ok=True)

#     # Creating one script for all perturbed states as an array job
#     script_filename = f"{folder_for_jobs}/prop_fm{fm_num}_{sat_name}.sh"

#     script_content = f"""#!/bin/bash -l
# #$ -l h_rt=3:0:0
# #$ -l mem=6G
# #$ -l tmpfs=10G
# #$ -N Prop_fm{fm_num}_{sat_name}
# #$ -t 1-{num_arcs}
# #$ -wd {work_dir}
# #$ -o {logs_folder}/out_$TASK_ID.txt
# #$ -e {logs_folder}/err_$TASK_ID.txt

# module load python/miniconda3/4.10.3
# source $UCL_CONDA_PATH/etc/profile.d/conda.sh
# cp {user_home_dir}/fm_benchmark/ERP_tools/erp_tools_env.yml $TMPDIR/erp_tools_env.yml

# cp -r {user_home_dir}/fm_benchmark/ERP_tools $TMPDIR

# cd $TMPDIR/ERP_tools

# # The TASK_ID environment variable is used to differentiate between array job tasks
# /home/{os.getenv('USER')}/.conda/envs/erp_tools_env/bin/python $TMPDIR/ERP_tools/source/ForceModelBenchmark.py 
# """
#     with open(script_filename, 'w') as file:
#         file.write(script_content)

#     os.system(f"qsub {script_filename}")

# if __name__ == "__main__":
#     create_and_submit_job_scripts()

from tools.sp3_2_ephemeris import sp3_ephem_to_df
import os
import json
import pandas as pd

def setup_and_submit_jobs():
    print("force model benchmarking script")
    sat_names_to_test = ["GRACE-FO-A"]
    # sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    dates_to_test = ["2019-01-01", "2023-05-04"]
    num_arcs = 1  # number of OD arcs that will be run
    arc_length = 25  # length of each arc in minutes
    prop_length = 60 * 60 * 12  # length of propagation in seconds
    force_model_configs = [
        {'36x36gravity': True, '3BP': True},
        {'120x120gravity': True, '3BP': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True}]

    for fm_num, force_model_config in enumerate(force_model_configs):
        for sat_name in sat_names_to_test:
            for date in dates_to_test:
                ephemeris_df = sp3_ephem_to_df(sat_name, date)
                ephemeris_df = ephemeris_df.iloc[::2]
                for arc_number in range(num_arcs):
                    start_index = arc_number * arc_length
                    end_index = start_index + arc_length
                    OD_points = ephemeris_df.iloc[start_index:end_index]
                    prop_end_index = min(start_index + (prop_length // 60), len(ephemeris_df) - 1)
                    OP_reference_trajectory = ephemeris_df.iloc[start_index:prop_end_index]
                    
                    user_home_dir = os.path.expanduser("~")
                    folder_for_jobs = f"{user_home_dir}/Scratch/FMBenchmark/sge_jobs"
                    logs_folder = f"{user_home_dir}/Scratch/FMBenchmark/{sat_name}/logs_fm{fm_num}"
                    work_dir = f"{user_home_dir}/Scratch/FMBenchmark/{sat_name}/propagation_fm{fm_num}"

                    os.makedirs(folder_for_jobs, exist_ok=True)
                    os.makedirs(logs_folder, exist_ok=True)
                    os.makedirs(work_dir, exist_ok=True)

                    script_filename = f"{folder_for_jobs}/prop_fm{fm_num}_{sat_name}_arc{arc_number}.sh"
                    force_model_config_json = json.dumps(force_model_config)
                    OD_points_json = OD_points.to_json(orient='split')
                    OP_reference_trajectory_json = OP_reference_trajectory.to_json(orient='split')

                    script_content = f"""#!/bin/bash -l
#$ -l h_rt=3:0:0
#$ -l mem=6G
#$ -l tmpfs=6G
#$ -N Prop_fm{fm_num}_{sat_name}_arc{arc_number}
#$ -wd {work_dir}
#$ -o {logs_folder}/out_{arc_number}.txt
#$ -e {logs_folder}/err_{arc_number}.txt

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate erp_tools_env
cp {user_home_dir}/mc_collisions/ERP_tools/erp_tools_env.yml $TMPDIR/erp_tools_env.yml
cp -r {user_home_dir}/mc_collisions/ERP_tools $TMPDIR
cd $TMPDIR/ERP_tools

echo '{OD_points_json}' > {work_dir}/OD_points.json
echo '{OP_reference_trajectory_json}' > {work_dir}/OP_reference_trajectory.json
echo '{force_model_config_json}' > {work_dir}/force_model_config.json

/home/{os.getenv('USER')}/.conda/envs/erp_tools_env/bin/python $TMPDIR/ERP_tools/source/ForceModelBenchmark.py {sat_name} {work_dir}/OD_points.json {work_dir}/OP_reference_trajectory.json {prop_length} {arc_number} {work_dir}/force_model_config.json
"""
                    with open(script_filename, 'w') as file:
                        file.write(script_content)

                    os.system(f"qsub {script_filename}")

if __name__ == "__main__":
    setup_and_submit_jobs()