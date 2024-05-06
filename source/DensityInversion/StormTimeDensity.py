from ..tools.Get_SP3_from_GFZ_FTP import download_sp3
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from ..DensityInversion.KinematicDensity import density_inversion, ephemeris_to_density

import os
import json
import ftplib
from datetime import datetime, timedelta
import pandas as pd
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

def parse_dates_and_download():
    with open('output/DensityInversion/PODBasedAccelerometry/selected_storms_test.txt', 'r') as file:
        data = file.read()

    # Dictionary to store satellite data
    satellite_data = {}

    # Extract and process the date lists
    lines = data.splitlines()
    satellite_name = None
    for line in lines:
        if 'Satellite:' in line:
            # Correctly parse the satellite name from the line
            satellite_name = line.split('Satellite:')[0].strip()
            satellite_data[satellite_name] = []  # Initialize a list for this satellite
            print(f"Satellite: {satellite_name}")
        elif 'G' in line and '[' in line and satellite_name:  # Looking for lines that contain dates
            storm_level = line.strip().split(':')[0].strip()
            date_list_str = line[line.find('[') + 1:line.find(']')]
            # Parsing date strings formatted as tuples
            date_strs = date_list_str.split('datetime.date')
            dates = []
            for date_str in date_strs:
                if date_str:
                    date_str = date_str.strip("() ,")
                    if date_str:
                        year, month, day = map(int, date_str.split(","))
                        date = datetime(year, month, day).date()
                        dates.append(date)

            for date in dates:
                start_date = date - timedelta(days=1)
                end_date = date + timedelta(days=2)
                # Storing geomagnetic storm level and time period in satellite data
                satellite_data[satellite_name].append((storm_level, start_date, end_date))
                # Assuming download_sp3 is a defined function elsewhere
                # download_sp3(start_date, end_date, satellite_name)

    return satellite_data

def load_sp3_data_for_storms(storm_data):
    all_data = {}
    for satellite, storm_periods in storm_data.items():
        # Initialize a list to store DataFrames for each period for the satellite
        all_data[satellite] = []
        
        for storm_index, (storm_level, start_date, end_date) in enumerate(storm_periods):

            df_list = []
            df = sp3_ephem_to_df(satellite, date=str(start_date))
            df_list.append(df)
            all_data[satellite].append(df_list)

    return all_data

if __name__ == "__main__":
    force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
    storm_data = parse_dates_and_download()
    storm_ephem_data = load_sp3_data_for_storms(storm_data)
    for satellite, df_list in storm_ephem_data.items():
        for df in df_list:
            for i, df in enumerate(df):
                density_inversion_df = density_inversion(satellite, df, force_model_config)
                #save density_inversion_df to a CSV file
                density_inversion_df.to_csv(f"output/DensityInversion/PODBasedAccelerometry/{satellite}_storm_{i}.csv")
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

