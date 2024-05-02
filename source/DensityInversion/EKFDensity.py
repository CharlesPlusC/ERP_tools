from ..tools.EKF_UKF import EKF
from ..tools.utilities import project_acc_into_HCL, get_satellite_info, interpolate_positions, calculate_acceleration
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from ..tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00

import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_nc_accs_from_POD_accs(sat_name, interp_ephemeris_df, force_model_config, density_freq='15S'):
    #subtract the conservative accelerations from the accelerometer data to get the non-conservative accelerations

    sat_info = get_satellite_info(sat_name)
    settings = {'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
                'density_freq': density_freq}

    # Convert UTC column to datetime and set it as index
    interp_ephemeris_df['UTC'] = pd.to_datetime(interp_ephemeris_df['UTC'])
    interp_ephemeris_df.set_index('UTC', inplace=True)
    interp_ephemeris_df = interp_ephemeris_df.asfreq(settings['density_freq'])

    columns = ['UTC','accx', 'accy', 'accz', 'acc_h', 'acc_c', 'acc_l']

    non_conservative_accelerations_df = pd.DataFrame(columns=columns)

    for i in tqdm(range(1, len(interp_ephemeris_df)), desc='Computing Non-Conservative Accelerations'):
        epoch = interp_ephemeris_df.index[i]
        vel = np.array([interp_ephemeris_df['xv'].iloc[i], interp_ephemeris_df['yv'].iloc[i], interp_ephemeris_df['zv'].iloc[i]])
        state_vector = np.array([interp_ephemeris_df['x'].iloc[i], interp_ephemeris_df['y'].iloc[i], interp_ephemeris_df['z'].iloc[i], vel[0], vel[1], vel[2]])

        conservative_accelerations = state2acceleration(state_vector, epoch, settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], **force_model_config)
        computed_accelerations_sum = np.sum(list(conservative_accelerations.values()), axis=0)
        print(f"Computed Acc: {computed_accelerations_sum}")
        observed_acc = np.array([interp_ephemeris_df['accx'].iloc[i], interp_ephemeris_df['accy'].iloc[i], interp_ephemeris_df['accz'].iloc[i]])
        print(f"Observed Acc: {observed_acc}")
        diff = computed_accelerations_sum - observed_acc
        print(f"Diff: {diff}")

        diff_x, diff_y, diff_z = diff[0], diff[1], diff[2]
        diff_h, diff_c, diff_l = project_acc_into_HCL(diff_x, diff_y, diff_z, 
                                                      interp_ephemeris_df['x'].iloc[i], interp_ephemeris_df['y'].iloc[i], interp_ephemeris_df['z'].iloc[i], 
                                                      interp_ephemeris_df['xv'].iloc[i], interp_ephemeris_df['yv'].iloc[i], interp_ephemeris_df['zv'].iloc[i])

        non_conservative_accelerations_df.loc[i] = [epoch, diff_x, diff_y, diff_z, diff_h, diff_c, diff_l]
    return non_conservative_accelerations_df

# Use EKF and dymamics model to improve the acceleration estimates
def main():
    sat_names_to_test = ["GRACE-FO-A"]
    # force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
    conservative_force_model = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'relativity': True}
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        #slice the ephemeris to just contain the first 180 rows
        ephemeris_df = ephemeris_df.iloc[:180]
        interp_ephemeris_df = interpolate_positions(ephemeris_df, '0.01S') # Interpolate the ephemeris
        interp_ephemeris_df = calculate_acceleration(interp_ephemeris_df, '0.01S', 21, 7) # Get the inertial accelerations
        print(interp_ephemeris_df)
        #If already computed the accelerations, load them from the file
        # interp_ephemeris_df = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Data/GRACE-FO-A/2024-04-26_01-22-32_GRACE-FO-A_fm12597_density_inversion.csv")
        non_conservative_accelerations_df = compute_nc_accs_from_POD_accs(sat_name, interp_ephemeris_df, conservative_force_model, density_freq='15S')
        print(non_conservative_accelerations_df)

        xest, Pest, tspan = EKF(xest0, Pest0, dt, z, Q, R, f, time, h = [], F=[], H=[], method = 'RK4', const=[])

if __name__ == "__main__":
    main()
#  python -m source.DensityInversion.EKFDensity to run