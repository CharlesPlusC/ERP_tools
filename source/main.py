import numpy as np
import netCDF4 as nc
import os
from itertools import islice
from tools.data_processing import calculate_satellite_fov, process_trajectory, sgp4_prop_TLE ,combine_lw_sw_data, extract_hourly_ceres_data
from tools.plotting import plot_fov_radiance, create_animation

def main(dataset_path, TLE, jd_start, jd_end, dt, number_of_items=150):
    # Load the CERES dataset
    
    data = nc.Dataset(dataset_path)

    # TLE for OneWeb satellite
    sgp4_ephem = sgp4_prop_TLE(TLE = TLE, jd_start = jd_start, jd_end = jd_end, dt = dt)

    ceres_times, lat, lon, lw_radiation_data, sw_radiation_data = extract_hourly_ceres_data(data)

    combined_radiation_data = combine_lw_sw_data(lw_radiation_data, sw_radiation_data)

    lats, lons, alts, ceres_indices = process_trajectory(sgp4_ephem, ceres_times)

    # Create animation frames
    output_folder = "output/FOV_sliced_data"
    os.makedirs(output_folder, exist_ok=True)

    # Loop and call the function for combined data
    for idx, (alt, sat_lat, sat_lon, ceres_idx) in islice(enumerate(zip(alts, lats, lons, ceres_indices)), number_of_items):
        print("idx:", idx)
        horizon_distance_km = calculate_satellite_fov(alt)

        # # Call for LW radiation
        # lw_plot_filename = os.path.join(output_folder, f'fov_lw_rad_{idx}.png')
        # plot_fov_radiance("CERES_lw_flux", ceres_idx, lw_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_distance_km, lw_plot_filename, ceres_times)

        # # Call for SW radiation
        # sw_plot_filename = os.path.join(output_folder, f'fov_sw_rad_{idx}.png')
        # plot_fov_radiance("CERES_sw_flux", ceres_idx, sw_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_distance_km, sw_plot_filename, ceres_times)

        # Call for combined radiation
        combined_plot_filename = os.path.join(output_folder, f'fov_lwsw_rad_{idx}.png')
        plot_fov_radiance("CERES_combined_flux", ceres_idx, combined_radiation_data, lat, lon, sat_lat, sat_lon, alt, horizon_distance_km, combined_plot_filename, ceres_times)

    # Create filenames lists
    lw_filenames = [os.path.join(output_folder, f'fov_lw_rad_{idx}.png') for idx in range(number_of_items)]
    sw_filenames = [os.path.join(output_folder, f'fov_sw_rad_{idx}.png') for idx in range(number_of_items)]
    combined_filenames = [os.path.join(output_folder, f'fov_lwsw_rad_{idx}.png') for idx in range(number_of_items)]

    # # Create LW animation
    # lw_animation_path = os.path.join(output_folder, 'lw_flux_animation.gif')
    # create_animation(lw_filenames, lw_animation_path)

    # # Create SW animation
    # sw_animation_path = os.path.join(output_folder, 'sw_flux_animation.gif')
    # create_animation(sw_filenames, sw_animation_path)

    # Create combined animation
    combined_animation_path = os.path.join(output_folder, 'combined_flux_animation_nipy.gif')
    create_animation(combined_filenames, combined_animation_path)

if __name__ == "__main__":
    #OneWeb TLE
    TLE = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993 \n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"
    # oneweb_tle_time = TLE_time(oneweb_test_tle)
    jd_start = 2460069.5000000 #force time to be within the CERES dataset
    jd_end = jd_start + 1
    dt = 60 #seconds
    dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc' # Hourly data
    main(dataset_path, TLE, jd_start, jd_end, dt, number_of_items=150)