import datetime
import numpy as np
import pandas as pd
from astropy.time import Time
from typing import Tuple
from tools.utilities import yyyy_mm_dd_hh_mm_ss_to_jd, doy_to_dom_month, jd_to_utc, std_dev_from_lower_triangular

def spacex_ephem_to_df_w_cov(ephem_path: str) -> pd.DataFrame:
    """
    Convert SpaceX ephemeris data, including covariance terms, into a pandas DataFrame.

    Parameters
    ----------
    ephem_path : str
        Path to the ephemeris file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing parsed SpaceX ephemeris data, including covariance terms.
    """
    with open(ephem_path) as f:
        lines = f.readlines()

    # Remove header lines and select every 4th line starting from the first data line
    t_xyz_uvw = lines[4::4]

    # Extract t, x, y, z, u, v, w
    t = [float(i.split()[0]) for i in t_xyz_uvw]
    x = [float(i.split()[1]) for i in t_xyz_uvw]
    y = [float(i.split()[2]) for i in t_xyz_uvw]
    z = [float(i.split()[3]) for i in t_xyz_uvw]
    u = [float(i.split()[4]) for i in t_xyz_uvw]
    v = [float(i.split()[5]) for i in t_xyz_uvw]
    w = [float(i.split()[6]) for i in t_xyz_uvw]

    # Extract the 21 covariance terms (3 lines after each primary data line)
    covariance_data = {f'cov_{i+1}': [] for i in range(21)}
    for i in range(5, len(lines), 4):  # Start from the first covariance line
        cov_lines = lines[i:i+3]  # Get the three lines of covariance terms
        cov_values = ' '.join(cov_lines).split()
        for j, value in enumerate(cov_values):
            covariance_data[f'cov_{j+1}'].append(float(value))

    # Convert timestamps to strings and call parse_spacex_datetime_stamps
    t_str = [str(int(i)) for i in t]  # Convert timestamps to string
    parsed_timestamps = parse_spacex_datetime_stamps(t_str)

    # Calculate Julian Dates for each timestamp
    jd_stamps = np.zeros(len(parsed_timestamps))
    for i in range(len(parsed_timestamps)):
        jd_stamps[i] = yyyy_mm_dd_hh_mm_ss_to_jd(int(parsed_timestamps[i][0]), int(parsed_timestamps[i][1]), 
                                                 int(parsed_timestamps[i][2]), int(parsed_timestamps[i][3]), 
                                                 int(parsed_timestamps[i][4]), int(parsed_timestamps[i][5]), 
                                                 int(parsed_timestamps[i][6]))

    # Initialize lists for averaged position and velocity standard deviations
    sigma_x, sigma_y, sigma_z, sigma_xv, sigma_yv, sigma_zv = [], [], [], [], [], []

    # Calculate averaged standard deviations for each row
    for _, row in pd.DataFrame(covariance_data).iterrows():
        std_devs = std_dev_from_lower_triangular(row.values)
        sigma_x.append(std_devs[0])
        sigma_y.append(std_devs[1])
        sigma_z.append(std_devs[2])
        sigma_xv.append(std_devs[3])
        sigma_yv.append(std_devs[4])
        sigma_zv.append(std_devs[5])

    # Construct the DataFrame with all data
    spacex_ephem_df = pd.DataFrame({
        't': t,
        'x': x,
        'y': y,
        'z': z,
        'xv': u,
        'yv': v,
        'zv': w,
        'JD': jd_stamps,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'sigma_z': sigma_z,
        'sigma_xv': sigma_xv,
        'sigma_yv': sigma_yv,
        'sigma_zv': sigma_zv,
        **covariance_data
    })

    # Multiply all the values except the times by 1000 to convert from km to m
    columns_to_multiply = ['x', 'y', 'z', 'xv', 'yv', 'zv', 
                        'sigma_x', 'sigma_y', 'sigma_z', 
                        'sigma_xv', 'sigma_yv', 'sigma_zv']

    for col in columns_to_multiply:
        spacex_ephem_df[col] *= 1000

    covariance_columns = [f'cov_{i+1}' for i in range(21)]
    for col in covariance_columns:
        spacex_ephem_df[col] *= 1000**2

    spacex_ephem_df['hours'] = (spacex_ephem_df['JD'] - spacex_ephem_df['JD'][0]) * 24.0 # hours since first timestamp
    spacex_ephem_df['UTC'] = spacex_ephem_df['JD'].apply(jd_to_utc)
    # TODO: I am gaining 3 milisecond per minute in the UTC time. Why?
    return spacex_ephem_df

def parse_spacex_datetime_stamps(timestamps: list) -> np.ndarray:
    """
    Parse SpaceX ephemeris datetime stamps into year, day of year, hour, minute, and second.

    Parameters
    ----------
    timestamps : List[str]
        List of SpaceX ephemeris datetime stamps.

    Returns
    -------
    np.ndarray
        Parsed timestamps (year, month, day, hour, minute, second, millisecond).
    """
    parsed_tstamps = np.zeros((len(timestamps), 7))

    for i, tstamp in enumerate(timestamps):
        # Ensure the timestamp string is at least 16 characters long
        tstamp_str = str(tstamp)
        if len(tstamp_str) < 16:
            # Pad zeros at the end if necessary
            tstamp_str = tstamp_str.ljust(16, '0')

        # Extract year, day of year, hour, minute, second, and millisecond
        year = tstamp_str[0:4]
        dayofyear = tstamp_str[4:7]
        # Convert day of year to month and day
        day_of_month, month = doy_to_dom_month(int(year), int(dayofyear))
        hour = tstamp_str[7:9]
        minute = tstamp_str[9:11]
        second = tstamp_str[11:13]
        millisecond = tstamp_str[13:16]

        # Convert millisecond to integer
        millisecond = int(millisecond) if millisecond else 0

        # Add the parsed timestamp to the array
        parsed_tstamps[i] = [int(year), int(month), int(day_of_month), int(hour), int(minute), int(second), millisecond]

    return parsed_tstamps

def read_spacex_ephemeris(ephem_path: str) -> Tuple[float, float, int]:
    """
    Read a SpaceX ephemeris file and extracts start time, end time, and step size.

    Parameters
    ----------
    ephem_path : str
        Path to the ephemeris file.

    Returns
    -------
    Tuple[float, float, int]
        Start time (JD), end time (JD), step size (seconds).
    """
    # read the first 5 lines of the operator ephem file
    with open(ephem_path) as f:
        ephem_lines = f.readlines()
    ephem_lines = ephem_lines[:5]
    ephem_utc_start = str(ephem_lines[1][16:16+19]) # start time
    ephem_utc_end = str(ephem_lines[1][55:55+19]) # end time
    ephem_step_size = int(ephem_lines[1][89:89+2]) # step size
    #convert to datetime object
    ephem_utc_dt_obj_start = datetime.datetime.strptime(ephem_utc_start, '%Y-%m-%d %H:%M:%S')
    ephem_utc_dt_obj_end = datetime.datetime.strptime(ephem_utc_end, '%Y-%m-%d %H:%M:%S')
    # convert to julian date
    ephem_start_jd_dt_obj = Time(ephem_utc_dt_obj_start).jd
    ephem_end_jd_dt_obj = Time(ephem_utc_dt_obj_end).jd

    return ephem_start_jd_dt_obj, ephem_end_jd_dt_obj, ephem_step_size

def spacex_ephem_to_dataframe(ephem_path: str) -> pd.DataFrame:
    """
    Convert SpaceX ephemeris data into a pandas DataFrame.

    Parameters
    ----------
    ephem_path : str
        Path to the ephemeris file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing parsed SpaceX ephemeris data.
    """
    # read in the text file 
    with open(ephem_path) as f:
        lines = f.readlines()
    # remove the header lines
    lines = lines[4:]
    # select every 4th line
    t_xyz_uvw = lines[::4]
    # from all the lines in t_xyz_uvw select the first float in each line and append that to a list
    t = [float(i.split()[0]) for i in t_xyz_uvw]
    x = [float(i.split()[1]) for i in t_xyz_uvw]
    y = [float(i.split()[2]) for i in t_xyz_uvw]
    z = [float(i.split()[3]) for i in t_xyz_uvw]
    u = [float(i.split()[4]) for i in t_xyz_uvw]
    v = [float(i.split()[5]) for i in t_xyz_uvw]
    w = [float(i.split()[6]) for i in t_xyz_uvw]
    
    # make all the values in the list 't' into a numpy array
    tstamp_array = np.array(t)
    # parse the timestamps into year, day of year, hour, minute, and second
    parsed_tstamps = parse_spacex_datetime_stamps(tstamp_array)
    # convert the parsed timestamps into julian dates
    jd_stamps = np.zeros(len(parsed_tstamps))
    for i in range(0, len(parsed_tstamps), 1):
        jd_stamps[i] = yyyy_mm_dd_hh_mm_ss_to_jd(int(parsed_tstamps[i][0]), int(parsed_tstamps[i][1]), int(parsed_tstamps[i][2]), int(parsed_tstamps[i][3]), int(parsed_tstamps[i][4]), int(parsed_tstamps[i][5]), int(parsed_tstamps[i][6]))

    # take t, x, y, z, u, v, w and put them into a dataframe
    spacex_ephem_df = pd.DataFrame({'jd_time':jd_stamps, 'x':x, 'y':y, 'z':z, 'u':u, 'v':v, 'w':w})
    # use the function meme_2_teme() to convert the x, y, z, u, v, w values from the MEME frame to the TEME frame
    # remove the last row from spacex_ephem_df
    spacex_ephem_df = spacex_ephem_df[:-1]

    return spacex_ephem_df