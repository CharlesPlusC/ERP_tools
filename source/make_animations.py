import numpy as np
import netCDF4 as nc
import os
from itertools import islice
from tools.data_processing import calculate_satellite_fov, process_trajectory, sgp4_prop_TLE ,combine_lw_sw_data, extract_hourly_ceres_data
from tools.plotting import plot_radiance_animation, plot_radiance_geiger

def main(dataset_path, TLE, jd_start, jd_end, dt, number_of_tsteps, output_folder):
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

    # Call the plot_radiance_animation function with the necessary parameters
    # plot_radiance_animation(alts=alts, lats=lats, lons=lons, ceres_indices=ceres_indices, 
    #                         lw_radiation_data=lw_radiation_data, sw_radiation_data=sw_radiation_data, 
    #                         combined_radiation_data=combined_radiation_data, lat=lat, lon=lon, ceres_times=ceres_times, 
    #                         number_of_tsteps=number_of_tsteps, lw=False, sw=False, lwsw=True, output_folder=output_folder)

    plot_radiance_geiger(alts, lats, lons, ceres_indices, lw_radiation_data, sw_radiation_data, 
                         combined_radiation_data, lat, lon, ceres_times, number_of_tsteps, 
                         lw=True, sw=True, lwsw=True, output_folder=output_folder)

if __name__ == "__main__":
    # Configuration for OneWeb TLE and CERES dataset
    #oneweb TLE
    TLE_1 = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993\n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"
    # a different oneweb satellite
    TLE_2 = "1 56716U 23068G   23345.58334491  .00000013  00000+0  00000+0 0  9990\n2 56716  87.9381  66.8814 0002382 101.3173 258.1798 13.24936606 30448"
    # starlink TLE
    TLE_3 = "1 58214U 23170J   23345.43674150  .00003150  00000+0  17305-3 0  9997\n2 58214  42.9996 329.1219 0001662 255.3130 104.7534 15.15957346  7032"
    # a different starlink satellite
    TLE_4 = "1 56534U 23065AH  23345.43622091  .00010158  00000+0  76519-3 0  9998\n2 56534  43.0036 329.2697 0001189 268.3661  91.7045 15.02572222 32751"
    jd_start = 2460069.5000000  # Force time to be within the CERES dataset
    jd_end = jd_start + 1 # 1 day later
    dt = 60  # Seconds
    dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc'  # Hourly data

    # # Execute the main function
    main(dataset_path, TLE_1, jd_start, jd_end, dt, number_of_tsteps=1000, output_folder='output/FOV_sliced_data/geiger_plots/oneweb_1/')
    main(dataset_path, TLE_2, jd_start, jd_end, dt, number_of_tsteps=1000, output_folder='output/FOV_sliced_data/geiger_plots/oneweb_2/')
    main(dataset_path, TLE_3, jd_start, jd_end, dt, number_of_tsteps=1000, output_folder='output/FOV_sliced_data/geiger_plots/starlink_1/')
    main(dataset_path, TLE_4, jd_start, jd_end, dt, number_of_tsteps=1000, output_folder='output/FOV_sliced_data/geiger_plots/starlink_2/')