# for all the satellites below, this script will take the sp3 files in sp3_cache, and concatenate them into a single dataframe per spacecraft. 
# the datatframe will then be used to write out an ephemeris file for each spacecraft that will be saved in the specified directory.

import json
import pandas as pd
import sp3
import gzip
import tempfile
import os
import glob

from tools.utilities import utc_jd_date, itrs_to_gcrs

def read_sp3_gz_file(sp3_gz_file_path):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.sp3')
    temp_file_path = temp_file.name

    with gzip.open(sp3_gz_file_path, 'rb') as gz_file:
        temp_file.write(gz_file.read())
    temp_file.close()

    product = sp3.Product.from_file(temp_file_path)
    satellite = product.satellites[0]
    records = satellite.records

    times = []
    positions = []
    velocities = []

    for record in records:
        times.append(record.time)
        positions.append(record.position)
        velocities.append(record.velocity if record.velocity else (None, None, None))

    df = pd.DataFrame({
        'Time': times,
        'Position_X': [pos[0]/1000 for pos in positions],
        'Position_Y': [pos[1]/1000 for pos in positions],
        'Position_Z': [pos[2]/1000 for pos in positions],
        'Velocity_X': [vel[0]/1000 for vel in velocities],
        'Velocity_Y': [vel[1]/1000 for vel in velocities],
        'Velocity_Z': [vel[2]/1000 for vel in velocities]
    })

    os.remove(temp_file_path)
    return df

def process_sp3_files(base_path, sat_list):
    all_dataframes = {sat_name: [] for sat_name in sat_list}

    for sat_name, sat_info in sat_list.items():
        sp3_c_code = sat_info['sp3-c_code']
        satellite_path = os.path.join(base_path, sp3_c_code)
        for day_folder in glob.glob(f"{satellite_path}/*"):
            for sp3_gz_file in glob.glob(f"{day_folder}/*.sp3.gz"):
                df = read_sp3_gz_file(sp3_gz_file)
                all_dataframes[sat_name].append(df)

    # Concatenating dataframes for each spacecraft
    concatenated_dataframes = {}
    for sat_name, dfs in all_dataframes.items():
        concatenated_df = pd.concat(dfs).drop_duplicates(subset='Time').set_index('Time').sort_index()
        
        # Checking for consistent time intervals (30 seconds), ignoring NaN values
        time_diffs = concatenated_df.index.to_series().diff().dt.total_seconds()
        if not time_diffs[time_diffs.notna()].eq(30).all():
            print(f"found a time interval of {time_diffs[time_diffs.notna()].unique()} seconds in data for {sat_name}")
            raise ValueError(f"Inconsistent time intervals found in data for {sat_name}")

        concatenated_dataframes[sat_name] = concatenated_df

    return concatenated_dataframes

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
            print

        # Convert time to MJD
        mjd_times = [utc_jd_date(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, mjd=True) for dt in df.index]
        df['MJD'] = mjd_times

        # Prepare ITRS coordinates
        itrs_positions = df[['Position_X', 'Position_Y', 'Position_Z']].values
        itrs_velocities = df[['Velocity_X', 'Velocity_Y', 'Velocity_Z']].values

        # Convert to ICRS (ECI)
        icrs_positions, icrs_velocities = itrs_to_gcrs(itrs_positions, itrs_velocities, df['MJD'].iloc[0])

        # Add new columns for ECI coordinates
        df['pos_x_eci'], df['pos_y_eci'], df['pos_z_eci'] = icrs_positions.T
        df['vel_x_eci'], df['vel_y_eci'], df['vel_z_eci'] = icrs_velocities.T

if __name__ == "__main__":
    main()