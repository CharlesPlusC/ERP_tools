import os
import pandas as pd
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor
import sys

def check_disk_space(directory):
    total, used, free = shutil.disk_usage(directory)
    return free

def extract_min_distance(file_path):
    try:
        df = pd.read_csv(file_path, usecols=['UTC', 'Distance'])
        min_distance_row = df.loc[df['Distance'].idxmin()]
        print(f"extracted TCA: {min_distance_row['UTC']}; DCA: {min_distance_row['Distance']} from {file_path}")
        return min_distance_row['UTC'], min_distance_row['Distance']
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

def process_file(file_path):
    tca, dca = extract_min_distance(file_path)
    if tca and dca:
        return {'TCA': tca, 'DCA': dca}
    return None

def process_spacecraft(sat_name, fm_num, base_directory):
    pattern = os.path.join(base_directory, sat_name, f'results_{fm_num}', '*_distances.csv')
    distance_files = glob.glob(pattern)

    if not distance_files:
        return

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, distance_files)

    summary_data = [result for result in results if result]

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        stats_dir = os.path.join(base_directory, sat_name, 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        output_file_path = os.path.join(stats_dir, f'sc_{sat_name}_fm_{fm_num}_TCADCA.csv')
        
        # Check disk space before writing the file
        free_space = check_disk_space(stats_dir)
        estimated_space_needed = summary_df.memory_usage(index=True, deep=True).sum() * 10  # Rough estimate

        if free_space > estimated_space_needed:
            summary_df.to_csv(output_file_path, index=False)
            print(f"Output file created: {output_file_path}")
        else:
            print("Not enough disk space to write the output file.")

def main(base_directory="/home/zcesccc/Scratch/MCCollisions", specific_sat_name=None):
    if not os.path.exists(base_directory):
        print("Base directory does not exist.")
        return

    if specific_sat_name:
        sat_names = [specific_sat_name]
    else:
        sat_names = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    for sat_name in sat_names:
        fm_dirs = [d for d in glob.glob(os.path.join(base_directory, sat_name, 'results_fm*')) if os.path.isdir(d)]

        for fm_dir in fm_dirs:
            fm_num = fm_dir.split('_')[-1]
            process_spacecraft(sat_name, fm_num, base_directory)

if __name__ == "__main__":
    specific_sat = sys.argv[1] if len(sys.argv) > 1 else None
    main(specific_sat_name=specific_sat)