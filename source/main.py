import numpy as np
import netCDF4 as nc
import os
from itertools import islice
from tools.data_processing import calculate_satellite_fov, process_trajectory, sgp4_prop_TLE ,combine_lw_sw_data, extract_hourly_ceres_data
from tools.plotting import plot_radiance_animation, plot_radiance_geiger

def main(dataset_path, TLE, jd_start, jd_end, dt, number_of_tsteps=150):
    # Load the CERES dataset
    data = nc.Dataset(dataset_path)

    # TLE for OneWeb satellite
    sgp4_ephem = sgp4_prop_TLE(TLE=TLE, jd_start=jd_start, jd_end=jd_end, dt=dt)

    # Extract data from the CERES dataset
    ceres_times, lat, lon, lw_radiation_data, sw_radiation_data = extract_hourly_ceres_data(data)

    # Combine longwave and shortwave radiation data
    combined_radiation_data = combine_lw_sw_data(lw_radiation_data, sw_radiation_data)

    # Process the satellite trajectory
    lats, lons, alts, ceres_indices = process_trajectory(sgp4_ephem, ceres_times)

    # # Call the plot_radiance_animation function with the necessary parameters
    # plot_radiance_animation(alts=alts, lats=lats, lons=lons, ceres_indices=ceres_indices, 
    #                         lw_radiation_data=lw_radiation_data, sw_radiation_data=sw_radiation_data, 
    #                         combined_radiation_data=combined_radiation_data, lat=lat, lon=lon, ceres_times=ceres_times, 
    #                         number_of_tsteps=number_of_tsteps, lw=True, sw=True, lwsw=True, output_folder=output_folder)

    plot_radiance_geiger(alts, lats, lons, ceres_indices, lw_radiation_data, sw_radiation_data, 
                         combined_radiation_data, lat, lon, ceres_times, number_of_tsteps, 
                         lw=True, sw=True, lwsw=True, output_folder="output/FOV_sliced_data/geiger_plots")

if __name__ == "__main__":
    # Configuration for OneWeb TLE and CERES dataset
    TLE = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993\n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"
    jd_start = 2460069.5000000  # Force time to be within the CERES dataset
    jd_end = jd_start + 1
    dt = 60  # Seconds
    dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc'  # Hourly data

    # Execute the main function
    main(dataset_path, TLE, jd_start, jd_end, dt, number_of_tsteps=500)