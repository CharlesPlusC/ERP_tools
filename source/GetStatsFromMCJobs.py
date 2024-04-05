import os
import pandas as pd
import glob

def extract_min_distance(file_path):
    """
    Extract the smallest distance and its corresponding time from a CSV file.
    """
    df = pd.read_csv(file_path)
    if df.empty:
        print(f"Warning: The file {file_path} is empty.")
        return None, None
    min_distance_row = df.loc[df['Distance'].idxmin()]
    return min_distance_row['UTC'], min_distance_row['Distance']

def process_spacecraft(sat_name, fm_num, base_directory):
    """
    Process all distance files for a specific spacecraft and force model,
    creating a summary CSV file with the minimum distance and time of closest approach.
    """
    pattern = os.path.join(base_directory, sat_name, f'results_{fm_num}', '*_distances.csv')
    distance_files = glob.glob(pattern)
    print(f"looking for files with pattern: {pattern}")

    if not distance_files:
        print(f"No distance files found for spacecraft {sat_name} and force model {fm_num}.")
        return

    print(f"Processing {len(distance_files)} files for spacecraft {sat_name} and force model {fm_num}.")
    summary_data = []
    for i, file_path in enumerate(distance_files):
        print(f"Processing file: {file_path}")
        tca, dca = extract_min_distance(file_path)
        if tca is not None and dca is not None:
            summary_data.append({'TCA': tca, 'DCA': dca, 'sample_number': i})
        else:
            print(f"No valid data in file: {file_path}")

    if not summary_data:
        print(f"Warning: No data was added for spacecraft {sat_name} and force model {fm_num}. CSV will be empty.")
    else:
        summary_df = pd.DataFrame(summary_data)
        output_file_path = os.path.join(base_directory, sat_name, 'stats', f'sc_{sat_name}_fm_{fm_num}_TCADCA.csv')
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        summary_df.to_csv(output_file_path, index=False)
        print(f"Output file created: {output_file_path}")

def main(base_directory="/home/zcesccc/Scratch/MCCollisions"):
    """
    Iterate through all spacecraft and force models, processing distance files.
    """
    if not os.path.exists(base_directory):
        print(f"Base directory {base_directory} does not exist.")
        return

    sat_names = os.listdir(base_directory)
    if not sat_names:
        print("No spacecraft directories found.")
        return

    for sat_name in sat_names:
        fm_dirs = glob.glob(os.path.join(base_directory, sat_name, 'results_fm*'))
        if not fm_dirs:
            print(f"No force model directories found for spacecraft {sat_name}.")
            continue

        for fm_dir in fm_dirs:
            fm_num = fm_dir.split('_')[-1]
            process_spacecraft(sat_name, fm_num, base_directory)

if __name__ == "__main__":
    main()
