import netCDF4 as nc
from itertools import islice
from source.tools.ceres_data_processing import process_trajectory ,combine_lw_sw_data, extract_hourly_ceres_data
from tools.plotting import plot_radiance_animation, plot_radiance_geiger
from tools.TLE_tools import sgp4_prop_TLE
import concurrent.futures

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

def process_satellite(dataset_path, TLE, jd_start, jd_end, dt, number_of_tsteps, output_folder):
    # Assuming 'main' is your function that processes the satellite data
    main(dataset_path, TLE, jd_start, jd_end, dt, number_of_tsteps=number_of_tsteps, output_folder=output_folder)

if __name__ == "__main__":

    #oneweb TLE
    TLE_1 = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993\n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"
    # a different oneweb satellite
    TLE_2 = "1 56716U 23068G   23345.58334491  .00000013  00000+0  00000+0 0  9990\n2 56716  87.9381  66.8814 0002382 101.3173 258.1798 13.24936606 30448"
    # starlink TLE
    TLE_3 = "1 58214U 23170J   23345.43674150  .00003150  00000+0  17305-3 0  9997\n2 58214  42.9996 329.1219 0001662 255.3130 104.7534 15.15957346  7032"
    # a different starlink satellite
    TLE_4 = "1 56534U 23065AH  23345.43622091  .00010158  00000+0  76519-3 0  9998\n2 56534  43.0036 329.2697 0001189 268.3661  91.7045 15.02572222 32751"

    # Configuration for satellites
    satellites = [
        {"TLE": TLE_1, "output_folder": "output/FOV_sliced_data/geiger_plots/oneweb_1/"},
        {"TLE": TLE_2, "output_folder": "output/FOV_sliced_data/geiger_plots/oneweb_2/"},
        {"TLE": TLE_3, "output_folder": "output/FOV_sliced_data/geiger_plots/starlink_1/"},
        {"TLE": TLE_4, "output_folder": "output/FOV_sliced_data/geiger_plots/starlink_2/"}
    ]

    jd_start = 2460069.5000000
    jd_end = jd_start + 1
    dt = 60
    no_steps = 100
    dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc'

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_satellite, dataset_path, sat["TLE"], jd_start, jd_end, dt, no_steps, sat["output_folder"]) for sat in satellites]

        for future in concurrent.futures.as_completed(futures):
            # You can add error handling or additional processing here
            print(f"Task completed with result: {future.result()}")
