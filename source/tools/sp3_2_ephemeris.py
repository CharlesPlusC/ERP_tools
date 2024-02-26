# for all the satellites below, this script will take the sp3 files in sp3_cache, and concatenate them into a single dataframe per spacecraft. 
# the datatframe will then be used to write out an ephemeris file for each spacecraft that will be saved in the specified directory.
import numpy as np
import json
import pandas as pd
import sp3
import gzip
import tempfile
import os
import glob
from source.tools.utilities import SP3_to_EME2000, utc_to_mjd
#run from CLI from root using: python source/tools/sp3_2_ephemeris.py

def read_sp3_file(sp3_file_path):
    print(f"reading sp3 file: {sp3_file_path}")

    product = sp3.Product.from_file(sp3_file_path)
    
    satellite = product.satellites[0]
    records = satellite.records

    times = []
    positions = []
    velocities = []

    for record in records:
        utc_time = record.time
        times.append(utc_time)
        positions.append(record.position)
        # Check if velocity data is available; assign NaNs if not
        velocities.append(record.velocity if record.velocity is not None else (np.nan, np.nan, np.nan))

    print("first position: ", positions[0])
    print("first velocity: ", velocities[0] if velocities[0] is not None else "None")
    print("first time: ", times[0])

    df = pd.DataFrame({
        'Time': times,
        'Position_X': [pos[0] / 1000 for pos in positions],
        'Position_Y': [pos[1] / 1000 for pos in positions],
        'Position_Z': [pos[2] / 1000 for pos in positions],
        # Use NaN for velocities if they are None
        'Velocity_X': [vel[0] / 1000 if vel is not None else np.nan for vel in velocities],
        'Velocity_Y': [vel[1] / 1000 if vel is not None else np.nan for vel in velocities],
        'Velocity_Z': [vel[2] / 1000 if vel is not None else np.nan for vel in velocities]
    })

    print(f"Read {len(df)} records from {sp3_file_path}")
    return df

def read_sp3_gz_file(sp3_gz_file_path):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.sp3')
    temp_file_path = temp_file.name

    with gzip.open(sp3_gz_file_path, 'rb') as gz_file:
        temp_file.write(gz_file.read())
    temp_file.close()

    print(f"reading sp3 file: {sp3_gz_file_path}")

    product = sp3.Product.from_file(temp_file_path)
    
    satellite = product.satellites[0]
    records = satellite.records

    times = []
    positions = []
    velocities = []

    for record in records:
        utc_time = record.time
        times.append(utc_time)
        positions.append(record.position)
        velocities.append(record.velocity)

    df = pd.DataFrame({
        'Time': times,
        'Position_X': [pos[0]/1000 for pos in positions],
        'Position_Y': [pos[1]/1000 for pos in positions],
        'Position_Z': [pos[2]/1000 for pos in positions],
        'Velocity_X': [vel[0]/1000 for vel in velocities],
        'Velocity_Y': [vel[1]/1000 for vel in velocities],
        'Velocity_Z': [vel[2]/1000 for vel in velocities]
    })

    print(f"Read {len(df)} records from {temp_file_path}")
    os.remove(temp_file_path)
    return df

def process_sp3_files(base_path, sat_list):
    all_dataframes = {sat_name: [] for sat_name in sat_list}

    for sat_name, sat_info in sat_list.items():
        sp3_c_code = sat_info['sp3-c_code']
        satellite_path = os.path.join(base_path, sp3_c_code)
        print(f"Processing {sat_name} from {satellite_path}")

        if not os.path.exists(satellite_path):
            print(f"Directory does not exist: {satellite_path}")
            continue

        for day_folder in glob.glob(f"{satellite_path}/*"):
            print(f"Looking in {day_folder}")
            sp3_files_found = glob.glob(f"{day_folder}/*sp3*")
            if not sp3_files_found:
                print(f"No SP3 files found in {day_folder}")
                continue

            for sp3_file in sp3_files_found:
                if sp3_file.endswith('.gz'):
                    print(f"Found .gz file: {sp3_file}")
                    df = read_sp3_gz_file(sp3_file)
                elif sp3_file.endswith('.sp3'):
                    print(f"Found .sp3 file: {sp3_file}")
                    df = read_sp3_file(sp3_file)
                else:
                    print(f"Incorrect or missing file extension for {sp3_file}")
                    continue

                print(f"Appending DataFrame for {sat_name} from {sp3_file}")  # Debug print
                all_dataframes[sat_name].append(df)

    concatenated_dataframes = {}
    for sat_name, dfs in all_dataframes.items():
        if dfs:
            concatenated_df = pd.concat(dfs).drop_duplicates(subset='Time').set_index('Time').sort_index()
            concatenated_dataframes[sat_name] = concatenated_df
            print(f"Concatenated {len(dfs)} DataFrames for {sat_name}")  # Confirmation print
        else:
            print(f"No data found for concatenation for {sat_name}, skipping.")

    return concatenated_dataframes

def write_ephemeris_file(satellite, df, sat_dict, output_dir="external/ephems"):
    # Create directory for satellite if it doesn't exist
    sat_dir = os.path.join(output_dir, satellite)
    os.makedirs(sat_dir, exist_ok=True)

    # Define the file name
    start_day = df.index.min().strftime("%Y-%m-%d")
    end_day = df.index.max().strftime("%Y-%m-%d")
    norad_id = sat_dict[satellite]['norad_id']
    file_name = f"NORAD{norad_id}-{start_day}-{end_day}.txt"
    file_path = os.path.join(sat_dir, file_name)

    # Write data to the ephemeris file
    with open(file_path, 'w') as file:
        for idx, row in df.iterrows():
            # Convert index to UTC string without timezone information
            utc = idx.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S.%f")
            line1 = f"{utc} {row['pos_x_eci']} {row['pos_y_eci']} {row['pos_z_eci']} {row['vel_x_eci']} {row['vel_y_eci']} {row['vel_z_eci']}\n"
            line2 = f"{row['sigma_x']} {row['sigma_y']} {row['sigma_z']} {row['sigma_xv']} {row['sigma_yv']} {row['sigma_zv']}\n"
            file.write(line1)
            file.write(line2)

def sp3_ephem_to_df(satellite, ephemeris_dir="external/ephems"):
    # Path to the directory containing the ephemeris files for the satellite
    sat_dir = os.path.join(ephemeris_dir, satellite)

    # List of all ephemeris files for the satellite
    ephemeris_files = glob.glob(os.path.join(sat_dir, "*.txt"))

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    for file_path in ephemeris_files:
        with open(file_path, 'r') as file:
            data = []
            while True:
                line1 = file.readline()
                line2 = file.readline()
                if not line2:  # Check if second line is empty (end of file)
                    break
                
                # Split lines and handle UTC timestamp
                line1_parts = line1.strip().split()
                utc = ' '.join(line1_parts[:2])
                ephemeris_values = line1_parts[2:]
                sigma_values = line2.strip().split()

                if len(ephemeris_values) != 6 or len(sigma_values) != 6:
                    raise ValueError("Incorrect number of values in ephemeris or sigma lines.")

                # Convert positions and velocities from km to m
                converted_values = [float(val) * 1000 if i < 6 else float(val) 
                                    for i, val in enumerate(ephemeris_values)]
                
                # Combine all values and convert to appropriate types
                row = [pd.to_datetime(utc)] + converted_values + sigma_values
                data.append(row)

            # Create a DataFrame from the current file data
            file_df = pd.DataFrame(data, columns=['UTC', 'x', 'y', 'z', 
                                                  'xv', 'yv', 'zv', 
                                                  'sigma_x', 'sigma_y', 'sigma_z', 
                                                  'sigma_xv', 'sigma_yv', 'sigma_zv'])

            # Append to the main DataFrame
            df = pd.concat([df, file_df])

    # Reset index
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    sat_list_path = "misc/sat_list.json"
    sp3_files_path = "external/sp3_files"
    with open(sat_list_path, 'r') as file:
        sat_dict = json.load(file)

    #printh the keys
    print(f"processing {sat_dict.keys()}")

    sp3_dataframes = process_sp3_files(sp3_files_path, sat_dict)

    # Assuming 'concatenated_dfs' is your dictionary of DataFrames
    for satellite, df in sp3_dataframes.items():
        # Convert the index to datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Convert time to MJD
        mjd_times = [utc_to_mjd(dt) for dt in df.index]
        df['MJD'] = mjd_times

        # Prepare CTS coordinates (ITRF 2014)
        itrs_positions = df[['Position_X', 'Position_Y', 'Position_Z']].values
        itrs_velocities = df[['Velocity_X', 'Velocity_Y', 'Velocity_Z']].values

        # Convert to EME2000
        icrs_positions, icrs_velocities = SP3_to_EME2000(itrs_positions, itrs_velocities, df['MJD'])
        # Add new columns for ECI coordinates
        df['pos_x_eci'], df['pos_y_eci'], df['pos_z_eci'] = icrs_positions.T
        df['vel_x_eci'], df['vel_y_eci'], df['vel_z_eci'] = icrs_velocities.T

    # Now add in  'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv' columns to each dataframe
    # for the time being we will input dummy values consistent with 5cm and 1mm/s for position and velocity respectively
    #TODO: replace this with actual values eventually
    for satellite, df in sp3_dataframes.items():
        df['sigma_x'] = 5e-1 #in km
        df['sigma_y'] = 5e-1 #in km
        df['sigma_z'] = 5e-1 #in km
        df['sigma_xv'] = 1e-3 #in km/s
        df['sigma_yv'] = 1e-3 #in km/s
        df['sigma_zv'] = 1e-3 #in km/s

    # After adding sigma columns to each dataframe
    for satellite, df in sp3_dataframes.items():
        write_ephemeris_file(satellite, df, sat_dict)

if __name__ == "__main__":
    main()
