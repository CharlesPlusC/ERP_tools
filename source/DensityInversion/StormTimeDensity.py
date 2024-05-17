from ..tools.Get_SP3_from_GFZ_FTP import download_sp3
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from .PODDensity import density_inversion
from ..tools.utilities import project_acc_into_HCL, get_satellite_info, interpolate_positions, calculate_acceleration

import os
import json
import ftplib
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

def load_sp3_codes(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def download_files(ftp_server, path, local_directory):
    try:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login()
            ftp.cwd(path)
            files = ftp.nlst()
            for filename in files:
                local_path = os.path.join(local_directory, filename)
                with open(local_path, 'wb') as local_file:
                    ftp.retrbinary('RETR ' + filename, local_file.write)
            print(f"Downloaded files to {local_directory}")
    except ftplib.all_errors as e:
        print(f"FTP error: {e}")

def download_storm_time_ephems(selected_storms_file='output/DensityInversion/PODBasedAccelerometry/selectedG5storms.txt'):
    with open(selected_storms_file, 'r') as file:
        data = file.read()

    # Dictionary to store satellite data
    satellite_data = {}

    # Extract and process the date lists
    lines = data.splitlines()
    satellite_name = None
    for line in lines:
        line = line.strip()
        if line.endswith("Satellite:"):
            # Correctly parse the satellite name from the line
            satellite_name = line.split()[0].strip()
            satellite_data[satellite_name] = []  # Initialize a list for this satellite
        elif 'G' in line and '[' in line and satellite_name:  # Looking for lines that contain dates
            storm_level = line.split(':')[0].strip()
            date_list_str = line[line.find('[') + 1:line.find(']')]
            # Parsing date strings formatted as datetime.date
            date_strs = date_list_str.split('datetime.date')
            dates = []
            for date_str in date_strs:
                if date_str:
                    date_str = date_str.strip("() ,")
                    if date_str:
                        year, month, day = map(int, date_str.split(","))
                        date = datetime(year, month, day).date()
                        dates.append(date)

            # Adjust the date to create the start_date and end_date
            for date in dates:
                start_date = date - timedelta(days=1)
                end_date = date + timedelta(days=1)
                satellite_data[satellite_name].append((storm_level, start_date, end_date))
                download_sp3(start_date, end_date, satellite_name)

    return satellite_data

def load_storm_sp3(storm_data):
    all_data = {}
    print(f"storm_data: {storm_data}")
    for satellite, storm_periods in storm_data.items():
        # Initialize a list to store DataFrames for each period for the satellite
        all_data[satellite] = []
        
        for storm_index, (storm_level, start_date, end_date) in enumerate(storm_periods):
            df_list = []
            df = sp3_ephem_to_df(satellite, date=str(end_date))
            df_list.append(df)
            all_data[satellite].append(df_list)

    return all_data

def filter_non_empty_dataframes(data_groups):
    return [df for df in data_groups if not df.empty]

def main():
    force_model_config = {
        '120x120gravity': True, '3BP': True, 'solid_tides': True,
        'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True
    }

    storm_data = download_storm_time_ephems()
    storm_ephem_data = load_storm_sp3(storm_data)
    seen_identifiers = set()

    for satellite, data_groups in storm_ephem_data.items():
        new_df_list = []
        for df_group in data_groups:
            filtered_group = filter_non_empty_dataframes(df_group)
            if filtered_group:  # Only add non-empty groups
                new_df_list.append(filtered_group)
        storm_ephem_data[satellite] = new_df_list

    # for satellite, df_list in storm_ephem_data.items():
    #same line as above but using tqdm to show progress
    for satellite, df_list in tqdm(storm_ephem_data.items(), desc="Density Inversion"):
        print(f"Processing {satellite}")
        for storm_period_index, df_period in enumerate(df_list):
            print(f"Processing storm period {storm_period_index} for {satellite}")
            for storm_df_index, storm_df in enumerate(df_period):
                if not storm_df.empty:
                    identifier = tuple(storm_df['UTC'].tolist())
                    if identifier in seen_identifiers:
                        print(f"Duplicate detected: Skipping storm {storm_df} for {satellite}")
                        continue
                    seen_identifiers.add(identifier)
                    print(f"Processing storm {storm_df_index} for {satellite}")
                    print(f"storm start: {storm_df['UTC'].iloc[0]}")
                    print(f"storm end: {storm_df['UTC'].iloc[-1]}")
                    interp_ephemeris_df = interpolate_positions(storm_df, '0.01S')
                    velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
                
                    density_inversion_df = density_inversion("GRACE-FO-A", velacc_ephem, 'vel_acc_x', 'vel_acc_y', 'vel_acc_z', force_model_config, nc_accs=False, 
                                models_to_query=['JB08', 'DTM2000', "NRLMSISE00"], density_freq='15S')

                    datenow = datetime.now().strftime("%Y%m%d%H%M%S")
                    savepath = f"output/DensityInversion/PODBasedAccelerometry/Data/StormAnalysis/{satellite}"
                    os.makedirs(savepath, exist_ok=True)
                    output_path = savepath + f"/{satellite}_storm_density_{storm_period_index}_{datenow}.csv"
                    density_inversion_df.to_csv(output_path)
                    print(f"Data saved to {output_path}")
                else:
                    print(f"Skipping processing for {satellite} storm {storm_df_index} due to empty DataFrame.")

# if __name__ == "__main__":
#     # main()
#     storm_data = download_storm_time_ephems()
#     storm_ephem_data = load_storm_sp3(storm_data)
#     print(f"storm_ephem_data: {storm_ephem_data}")
#     index = 0
#     for satellite, periods in storm_ephem_data.items():
#         for period_index, df_group in enumerate(periods):
#             if df_group:
#                 print(f"Processing {satellite} period {period_index}")
#                 print(f"head of df_group: {df_group[0].head()}")
    #then load each of the downloaded sp3 files using the datetimes in the selected_storms.txt file and the sp3_epheme_to_df function
    # Use ephemeris_to_density to:
        #1. ingest the ephemeris
        #2. interpolate the ephemeris
        #3. calculate the acceleration
        #4. perform the density inversion
    # Check if the interpolated ephemeris already exists. 
    # If it does, load it and skip the interpolation step. Use the density_inversion function to perform the density inversion directly
    # Save the density inversion output to a CSV file for each storm period
    # Calculate the delta drho/dt between the inverted-for density and the JB08 density


def create_and_submit_density_jobs():
    import os
    import json

    user_home_dir = os.getenv("HOME")
    project_root_dir = f"{user_home_dir}/Rhoin/ERP_tools/"
    folder_for_jobs = f"{user_home_dir}/Scratch/Rhoin/sge_jobs"
    work_dir = f"{user_home_dir}/Scratch/Rhoin/working"
    logs_folder = f"{user_home_dir}/Scratch/Rhoin/logs"
    output_folder = f"{user_home_dir}/Scratch/Rhoin/output"

    os.makedirs(folder_for_jobs, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    storm_data = download_storm_time_ephems()
    storm_ephem_data = load_storm_sp3(storm_data)

    index = 0
    for satellite, periods in storm_ephem_data.items():
        for period_index, df_group in enumerate(periods):
            if df_group:
                script_filename = f"{folder_for_jobs}/{satellite}_period{period_index}.sh"
                script_content = f"""#!/bin/bash -l
#$ -l h_rt=24:0:0
#$ -l mem=8G
#$ -l tmpfs=15G
#$ -N {satellite}_period{period_index}
#$ -t 1-{len(df_group)}
#$ -wd {work_dir}
#$ -o {logs_folder}/out_{satellite}_period{period_index}_$TASK_ID.txt
#$ -e {logs_folder}/err_{satellite}_period{period_index}_$TASK_ID.txt

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate erp_tools_env
export PYTHONPATH="{project_root_dir}:$PYTHONPATH"

cp -r {user_home_dir}/Rhoin/ERP_tools $TMPDIR

cd $TMPDIR/ERP_tools

/home/{os.getenv('USER')}/.conda/envs/erp_tools_env/bin/python -m source.DensityInversion.StormTimeDensity {satellite} {period_index} $SGE_TASK_ID {output_folder}
"""
                with open(script_filename, 'w') as file:
                    file.write(script_content)

                os.system(f"qsub {script_filename}")
                index += 1

def main_script(satellite, period_index, df_index, output_folder):
    import pandas as pd
    from datetime import datetime
    from tqdm import tqdm
    import json

    storm_data = download_storm_time_ephems()
    storm_ephem_data = load_storm_sp3(storm_data)

    df_list = storm_ephem_data[satellite][int(period_index)]
    if 0 <= int(df_index) - 1 < len(df_list):
        df = df_list[int(df_index) - 1]
        if not df.empty:
            force_model_config = {
                '120x120gravity': True, '3BP': True, 'solid_tides': True,
                'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True
            }

            interp_ephemeris_df = interpolate_positions(df, '0.01S')
            velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
            
            density_inversion_df = density_inversion(satellite, velacc_ephem, 'vel_acc_x', 'vel_acc_y', 'vel_acc_z', force_model_config, nc_accs=False, 
                        models_to_query=[None], density_freq='15S')

            datenow = datetime.now().strftime("%Y%m%d%H%M%S")
            savepath = f"{output_folder}/StormAnalysis/{satellite}"
            os.makedirs(savepath, exist_ok=True)
            output_path = savepath + f"/{satellite}_storm_density_{period_index}_{df_index}_{datenow}.csv"
            density_inversion_df.to_csv(output_path)
            print(f"Data saved to {output_path}")
        else:
            print(f"Skipping processing for {satellite} storm {df_index} due to empty DataFrame.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5:
        main_script(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        create_and_submit_density_jobs()