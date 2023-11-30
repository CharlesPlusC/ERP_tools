import numpy as np
import netCDF4 as nc
import os
from tools.data_processing import process_trajectory, sgp4_prop_TLE ,combine_lw_sw_data, extract_hourly_ceres_data
from tools.utilities import eci2ecef_astropy, ecef_to_lla, julian_day_to_ceres_time, find_nearest_index, calculate_satellite_fov
from tools.plotting import plot_fov_radiance, create_animation

# Load the CERES dataset
dataset_path = 'data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc' # Hourly data
data = nc.Dataset(dataset_path)

#Load the spacecraft trajectory data
oneweb_test_tle = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993 \n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"

# oneweb_tle_time = TLE_time(oneweb_test_tle)
oneweb_tle_time = 2460069.5000000 #force time to be within the CERES dataset
oneweb_sgp4_ephem = sgp4_prop_TLE(TLE = oneweb_test_tle, jd_start = oneweb_tle_time, jd_end = oneweb_tle_time + 1, dt = 60)


ceres_times, lat, lon, lw_radiation_data, sw_radiation_data = extract_hourly_ceres_data(data)

combined_radiation_data = combine_lw_sw_data(lw_radiation_data, sw_radiation_data)

lats, lons, alts, ceres_indices = process_trajectory(oneweb_sgp4_ephem, ceres_times)

ecef_oneweb_pos_list = []
ecef_oneweb_vel_list = []

for i in range(len(oneweb_sgp4_ephem)):
    ecef_oneweb_pos, ecef_oneweb_vel = eci2ecef_astropy(eci_pos = np.array([oneweb_sgp4_ephem[i][1]]), eci_vel = np.array([oneweb_sgp4_ephem[i][2]]), mjd = oneweb_sgp4_ephem[i][0]-2400000.5)
    ecef_oneweb_pos_list.append(ecef_oneweb_pos)
    ecef_oneweb_vel_list.append(ecef_oneweb_vel)

ow_lats, ow_lons, ow_alts = [], [], []

for i in range(len(ecef_oneweb_pos_list)):
    ow_lat, ow_lon, ow_alt = ecef_to_lla(ecef_oneweb_pos_list[i][0][0], ecef_oneweb_pos_list[i][0][1], ecef_oneweb_pos_list[i][0][2])
    ow_lats.append(ow_lat)
    ow_lons.append(ow_lon)
    ow_alts.append(ow_alt)

# Calculate satellite positions and times
# (Assuming you have already calculated sl_lats, sl_lons, sl_alts, and ow_lats, ow_lons, ow_alts)
ow_ceres_time = [julian_day_to_ceres_time(jd) for jd in (ephem[0] for ephem in oneweb_sgp4_ephem)]

# sl_ceres_indices = [find_nearest_index(ceres_times, t) for t in sl_ceres_time]
ow_ceres_indices = [find_nearest_index(ceres_times, t) for t in ow_ceres_time]

# Create animation frames
output_folder = "FOV_sliced_data"
os.makedirs(output_folder, exist_ok=True)

from itertools import islice
number_of_items = 220

# Extract the time, latitude, and longitude variables
ceres_times = data.variables['time'][:]  # Array of time points in the CERES dataset
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
lw_radiation_data = data.variables['toa_lw_all_1h'][:] 
sw_radiation_data = data.variables['toa_sw_all_1h'][:] 

# Check dimensions
assert lw_radiation_data.shape == sw_radiation_data.shape, "Data dimensions do not match"

# Combine the data
combined_radiation_data = lw_radiation_data + sw_radiation_data

# Loop and call the function for combined data
for idx, (alt, sat_lat, sat_lon, ceres_idx) in islice(enumerate(zip(ow_alts, ow_lats, ow_lons, ow_ceres_indices)), number_of_items):
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
lw_filenames = [os.path.join(output_folder, f'fov_lw_rad_{idx}.png') for idx in range(120)]
sw_filenames = [os.path.join(output_folder, f'fov_sw_rad_{idx}.png') for idx in range(120)]
combined_filenames = [os.path.join(output_folder, f'fov_lwsw_rad_{idx}.png') for idx in range(220)]



# # Create LW animation
# lw_animation_path = os.path.join(output_folder, 'lw_flux_animation.gif')
# create_animation(lw_filenames, lw_animation_path)

# # Create SW animation
# sw_animation_path = os.path.join(output_folder, 'sw_flux_animation.gif')
# create_animation(sw_filenames, sw_animation_path)

# Create combined animation
combined_animation_path = os.path.join(output_folder, 'combined_flux_animation_nipy.gif')
create_animation(combined_filenames, combined_animation_path)