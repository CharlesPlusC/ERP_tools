from tools.sp3_2_ephemeris import sp3_ephem_to_df
import os
import json

def generate_json_files(sat_names_to_test, dates_to_test, num_arcs, arc_length, prop_length):
    output_folder = f"output/FMBench" #relative to the ERP_tools directory
    os.makedirs(output_folder, exist_ok=True)

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

    for fm_num, force_model_config in enumerate(force_model_configs):
        for sat_name in sat_names_to_test:
            for date in dates_to_test:
                ephemeris_df = sp3_ephem_to_df(sat_name, date)
                ephemeris_df = ephemeris_df.iloc[::2]

                for arc_number in range(num_arcs):
                    start_index = arc_number * arc_length
                    end_index = start_index + arc_length
                    prop_end_index = min(start_index + (prop_length // 60), len(ephemeris_df))

                    OD_points = ephemeris_df.iloc[start_index:end_index]
                    OP_reference_trajectory = ephemeris_df.iloc[start_index:prop_end_index]

                    OD_points.to_json(f"{output_folder}/OD_points_{arc_number}.json", orient='split')
                    OP_reference_trajectory.to_json(f"{output_folder}/OP_reference_trajectory_{arc_number}.json", orient='split')
                    with open(f"{output_folder}/force_model_config_{arc_number}.json", 'w') as f:
                        json.dump(force_model_config, f)

def write_array_scripts(sat_names_to_test, dates_to_test, num_arcs, arc_length, prop_length):
    user_home_dir = os.path.expanduser("~")
    scratch_output = f"{user_home_dir}/Scratch/FMBenchmark"
    local_output = f"output/FMBench"
    sat_names_short = ["GFOA", "GFOB", "TSX", "TDX", "S1A", "S2A", "S2B", "S3A", "S3B", "S6A"]

    folder_for_jobs = f"{scratch_output}/sge_jobs"
    os.makedirs(folder_for_jobs, exist_ok=True)

    for fm_num in range(len(sat_names_to_test)):
        for sat_name in sat_names_to_test:
            script_filename = f"{folder_for_jobs}/prop_fm{fm_num}_{sat_name}_array_job.sh"
            script_content = f"""#!/bin/bash -l
#$ -t 1-{num_arcs}
#$ -l h_rt=40:0:0
#$ -l mem=16G
#$ -l tmpfs=20G
#$ -N fm{fm_num}_{sat_names_short[sat_names_to_test.index(sat_name)]}
#$ -wd {scratch_output}
#$ -o {scratch_output}/{sat_name}/logs_fm{fm_num}/out_$TASK_ID.txt
#$ -e {scratch_output}/{sat_name}/logs_fm{fm_num}/err_$TASK_ID.txt

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate erp_tools_env

/home/{os.getenv('USER')}/.conda/envs/erp_tools_env/bin/python $TMPDIR/ERP_tools/source/ForceModelBenchmark.py \\
    {scratch_output}/output_fm{fm_num}_arc$SGE_TASK_ID \\
    {sat_name} \\
    {local_output}/OD_points_$SGE_TASK_ID.json \\
    {local_output}/OP_reference_trajectory_$SGE_TASK_ID.json \\
    {prop_length} \\
    $SGE_TASK_ID \\
    {local_output}/force_model_config_$SGE_TASK_ID.json
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
    generate_json_files(sat_names_to_test, dates_to_test, num_arcs, arc_length, prop_length)
    write_array_scripts(sat_names_to_test, dates_to_test, num_arcs, arc_length, prop_length)
