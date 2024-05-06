from ..tools.Get_SP3_from_GFZ_FTP import download_sp3

# Download data for the storm time instances. From 1 day before to 2 days after the storm time instance
# Run sp3_3_ephemeris.py on all of them
# Then run KinematicDensity.py on all of them
# ❯ python -m source.DensityInversion.GetStormTimeSP3
import os
import json
import ftplib
from datetime import datetime, timedelta

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
    with open('output/DensityInversion/PODBasedAccelerometry/selected_storms.txt', 'r') as file:
        data = file.read()

    # Extract and process the date lists
    lines = data.splitlines()
    satellite_name = None
    for line in lines:
        if 'Satellite:' in line:
            satellite_name = line.split()[0].replace(":", "")  # Remove colon from satellite name
            print(f"Satellite: {satellite_name}")
        elif 'G' in line and '[' in line:  # looking for lines that contain dates
            storm_level = line.split()[0]
            date_list_str = line[line.find('[') + 1:line.find(']')]
            # Parsing date strings formatted as tuples
            date_strs = date_list_str.split('datetime.date')
            dates = []
            for date_str in date_strs:
                if date_str:
                    # Cleaning and extracting date components
                    date_str = date_str.strip("() ,")
                    if date_str:  # Checking if not empty after strip
                        year, month, day = map(int, date_str.split(","))
                        date = datetime(year, month, day).date()
                        dates.append(date)

            for date in dates:
                start_date = date - timedelta(days=1)
                end_date = date + timedelta(days=2)
                download_sp3(start_date, end_date, satellite_name)

if __name__ == "__main__":
    parse_dates_and_download()