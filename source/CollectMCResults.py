import os

def create_and_submit_stats_job(sat_name):
    user_home_dir = os.path.expanduser("~")
    folder_for_jobs = f"{user_home_dir}/Scratch/MCCollisions/statscollect_jobs"
    output_folder = f"{user_home_dir}/Scratch/MCCollisions/{sat_name}/stats"
    logs_folder = f"{user_home_dir}/Scratch/MCCollisions/{sat_name}/logs"
    python_script = f"{user_home_dir}/mc_collisions/ERP_tools/source/GetStatsFromMCJobs.py"

    os.makedirs(folder_for_jobs, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    # Creating one script for each satellite
    script_filename = f"{folder_for_jobs}/stats_{sat_name}.sh"
    script_content = f"""#!/bin/bash -l
#$ -l h_rt=3:0:0
#$ -l mem=16G
#$ -l tmpfs=16G
#$ -N Stats_{sat_name}
#$ -wd {output_folder}
#$ -o {logs_folder}/out.txt
#$ -e {logs_folder}/err.txt

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

# Run the Python script for the specific satellite
/home/{os.getenv('USER')}/.conda/envs/erp_tools_env/bin/python {python_script} {sat_name}
"""
    with open(script_filename, 'w') as file:
        file.write(script_content)

    os.system(f"qsub {script_filename}")

def main():
    sat_names_to_process = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]

    for sat_name in sat_names_to_process:
        create_and_submit_stats_job(sat_name)

if __name__ == "__main__":
    main()
