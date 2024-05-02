from tools.sp3_2_ephemeris import sp3_ephem_to_df
import os
import json

def generate_json_files(sat_names_to_test, dates_to_test, num_arcs, arc_length, prop_length):
    user_home_dir = os.path.expanduser("~")
    output_base = "output/FMBench"
    scratch_output = f"{user_home_dir}/Scratch/FMBenchmark"

    os.makedirs(output_base, exist_ok=True)

    force_model_configs = [
        {'36x36gravity': True, '3BP': True},
        {'120x120gravity': True, '3BP': True},
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True},
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True},
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True},
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'jb08drag': True},
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'dtm2000drag': True},
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'nrlmsise00drag': True}
    ]

    for sat_name in sat_names_to_test:
        for date in dates_to_test:
            ephemeris_df = sp3_ephem_to_df(sat_name, date)
            ephemeris_df = ephemeris_df.iloc[::2]
            for arc_number in range(num_arcs):
                arc_folder = f"{output_base}/{sat_name}/arc_{arc_number}"
                scratch_arc_folder = f"{scratch_output}/{sat_name}/arc_{arc_number}"
                os.makedirs(arc_folder, exist_ok=True)
                os.makedirs(scratch_arc_folder, exist_ok=True)

                for fm_num in range(len(force_model_configs)):
                    scratch_output_arc_folder = f"{scratch_arc_folder}/output_fm{fm_num}_arc{arc_number}"
                    os.makedirs(scratch_output_arc_folder, exist_ok=True)  # Correct placement for folder creation

                logs_folder = f"{scratch_output}/{sat_name}/logs_fm{fm_num}"
                os.makedirs(logs_folder, exist_ok=True)

                start_index = arc_number * arc_length
                end_index = start_index + arc_length
                prop_end_index = min(start_index + (prop_length // 60), len(ephemeris_df))

                OD_points = ephemeris_df.iloc[start_index:end_index]
                OP_reference_trajectory = ephemeris_df.iloc[start_index:prop_end_index]

                OD_points.to_json(f"{arc_folder}/OD_points.json", orient='split')
                OP_reference_trajectory.to_json(f"{arc_folder}/OP_reference_trajectory.json", orient='split')

                for config_num, force_model_config in enumerate(force_model_configs):
                    with open(f"{arc_folder}/force_model_config_{config_num}.json", 'w') as f:
                        json.dump(force_model_config, f)

import pandas as pd
# Example function adjustments for reading JSON data correctly:
def read_json_data(file_path):
    try:
        return pd.read_json(file_path, orient='split')
    except ValueError as e:
        print(f"Error reading JSON data from {file_path}: {e}")
        raise

def write_array_scripts(sat_names_to_test, num_arcs, prop_length):
    user_home_dir = os.path.expanduser("~")
    local_output = "output/FMBench"
    scratch_output = f"{user_home_dir}/Scratch/FMBenchmark"
    folder_for_jobs = f"{scratch_output}/sge_jobs"
    os.makedirs(folder_for_jobs, exist_ok=True)
    sat_names_short = ["GFOA", "GFOB", "TSX", "TDX", "S1A", "S2A", "S2B", "S3A", "S3B", "S6A"]
    force_model_configs = 9

    for sat_name in sat_names_to_test:
        sat_short_name = sat_names_short[sat_names_to_test.index(sat_name)]
        for fm_num in range(force_model_configs):
            logs_folder = f"{scratch_output}/{sat_name}/logs_fm{fm_num}"
            os.makedirs(logs_folder, exist_ok=True)
            script_filename = f"{folder_for_jobs}/prop_fm{fm_num}_{sat_name}_array_job.sh"
            script_content = f"""#!/bin/bash -l
#$ -t 1-{num_arcs}
#$ -l h_rt=40:0:0
#$ -l mem=16G
#$ -l tmpfs=20G
#$ -N fm{fm_num}{sat_short_name}
#$ -wd {scratch_output}
#$ -o {logs_folder}/out_$TASK_ID.txt
#$ -e {logs_folder}/err_$TASK_ID.txt

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate erp_tools_env

cp -r {user_home_dir}/FMBenchmark/ERP_tools $TMPDIR
cp -r {user_home_dir}/FMBenchmark/ERP_tools/output/FMBench/{sat_name} $TMPDIR/ERP_tools
cd $TMPDIR/ERP_tools
"""
            for arc_num in range(num_arcs):
                scratch_arc_folder = f"{scratch_output}/{sat_name}/arc_{arc_num}"
                output_arc_folder = f"{scratch_arc_folder}/output_fm{fm_num}_arc{arc_num}"
                script_content += f"""/home/{os.getenv('USER')}/.conda/envs/erp_tools_env/bin/python $TMPDIR/ERP_tools/source/ForceModelBenchmark.py \\
    {output_arc_folder} \\
    {sat_name} \\
    {local_output}/{sat_name}/arc_{arc_num}/OD_points.json \\
    {local_output}/{sat_name}/arc_{arc_num}/OP_reference_trajectory.json \\
    {prop_length} \\
    {arc_num} \\
    {local_output}/{sat_name}/arc_{arc_num}/force_model_config_{fm_num}.json
"""
            with open(script_filename, 'w') as file:
                file.write(script_content)
            os.system(f"qsub {script_filename}")
if __name__ == "__main__":
    sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X", "Sentinel-1A", "Sentinel-2A", "Sentinel-2B", "Sentinel-3A", "Sentinel-3B", "Sentinel-6A"]
    dates_to_test = ["2023-05-04"]
    num_arcs = 20
    arc_length = 25
    prop_length = 60 * 60 * 12
    print("Generating JSON files ")
    generate_json_files(sat_names_to_test, dates_to_test, num_arcs, arc_length, prop_length)
    print("JSON files generated")
    print("Writing array scripts")
    write_array_scripts(sat_names_to_test, num_arcs, prop_length)
    print("Array scripts written")