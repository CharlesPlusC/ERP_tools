import os
import json
import ftplib
from datetime import datetime

def load_sp3_codes(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def create_ftp_url(spacecraft, sp3_code, year, day):
    day_str = f"{day:03d}"
    return f"{spacecraft}/ORBIT/{sp3_code}/RSO/{year}/{day_str}/"

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

def main():
    spacecraft_name = "TerraSAR-X"
    year = 2019
    day_of_year = 335
    json_path = "misc/sat_list.json"
    local_directory = "external/sp3_files"
    sp3_codes = load_sp3_codes(json_path)
    if spacecraft_name in sp3_codes:
        sp3_code = sp3_codes[spacecraft_name]["sp3-c_code"]
        base_url = "isdcftp.gfz-potsdam.de"
        spacecraft_folder = {
            "CHAMP": "champ",
            "GRACE-FO-A": "grace-fo",
            "GRACE-FO-B": "grace-fo",
            "TerraSAR-X": "tsxtdx",
            "TanDEM-X": "tsxtdx"
        }.get(spacecraft_name, "")
        ftp_path = create_ftp_url(spacecraft_folder, sp3_code, year, day_of_year)
        download_files(base_url, ftp_path, local_directory)
    else:
        print(f"No SP3-C code found for {spacecraft_name}")

if __name__ == "__main__":
    main()
