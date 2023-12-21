import datetime
import numpy as np
import pandas as pd
from astropy.time import Time
from typing import Tuple, List

from tools.utilities import yyyy_mm_dd_hh_mm_ss_to_jd, doy_to_dom_month

def parse_spacex_datetime_stamps(timestamps: List[str]) -> np.ndarray:
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
    # make an array where we will store the year, day of year, hour, minute, and second for each timestamp
    parsed_tstamps = np.zeros((len(timestamps), 7))
    
    for i in range(0, len(timestamps), 1):
        tstamp_str = str(timestamps[i])
        # year is first 4 digits
        year = tstamp_str[0:4]
        # day of year is next 3 digits
        dayofyear = tstamp_str[4:7]
        #convert day of year to day of month and month number
        day_of_month, month = doy_to_dom_month(int(year), int(dayofyear))
        # hour is next 2 digits
        hour = tstamp_str[7:9]
        # minute is next 2 digits
        minute = tstamp_str[9:11]
        # second is next 2 digits
        second = tstamp_str[11:13]
        # milisecond is next 3 digits
        milisecond = tstamp_str[14:16]
        # add the parsed timestamp to the array
        parsed_tstamps[i] = ([int(year), int(month), int(day_of_month), int(hour), int(minute), int(second), int(milisecond)])

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