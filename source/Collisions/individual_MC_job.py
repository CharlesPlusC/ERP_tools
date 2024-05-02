import numpy as np
import pandas as pd
import sys
import os
from scipy.interpolate import interp1d
from tools.orekit_tools import propagate_state
import datetime
from tools.utilities import get_satellite_info, load_force_model_configs

def interpolate_ephemeris(df, start_time, end_time, freq='0.0001S', stitch=False):
    df = df.drop_duplicates(subset='UTC').set_index('UTC')
    df = df.sort_index()
    df_resampled = df.reindex(pd.date_range(start=start_time, end=end_time, freq=freq), method='nearest').asfreq(freq)
    interp_funcs = {col: interp1d(df.index.astype(int), df[col], fill_value='extrapolate') for col in ['x', 'y', 'z']}
    for col in ['x', 'y', 'z']:
        df_resampled[col] = interp_funcs[col](df_resampled.index.astype(int))
    df_filtered = df_resampled.loc[start_time:end_time].reset_index().rename(columns={'index': 'UTC'})
    if stitch:
        df_stitched = pd.concat([df.loc[:start_time - pd.Timedelta(freq), :], df_filtered.set_index('UTC'), df.loc[end_time + pd.Timedelta(freq):, :]]).reset_index()
        return df_stitched
    return df_filtered

def propagate_and_calculate(sat_name, perturbed_state_id, force_model_num, outpath=None):
    force_model_configs = load_force_model_configs('misc/fm_configs.json')
    user_home_dir = os.path.expanduser("~")
    res_folder = f'{user_home_dir}/mc_collisions/ERP_tools/output/Collisions/MC/interpolated_MC_ephems/'
    sc_res_folder = f"{res_folder}/{sat_name}"
    nominal_collision_df = pd.read_csv(f"{sc_res_folder}/{sat_name}_nominal_collision.csv")
    perturbed_states = np.loadtxt(f"{sc_res_folder}/{sat_name}_fm{force_model_num}_perturbed_states.csv", delimiter=",")
    # Load the nominal collision trajectory
    perturbed_state = perturbed_states[perturbed_state_id]
    force_model_config = force_model_configs[force_model_num]
    print(f"Propagating perturbed state {perturbed_state_id} for {sat_name}")
    print(f"Perturbed state: {perturbed_state}")

    start_date = nominal_collision_df['UTC'].iloc[0]
    end_date = nominal_collision_df['UTC'].iloc[-1]
    #convert to datetime
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S.%f')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S.%f')
    sat_info = get_satellite_info(sat_name)
    cd = sat_info['cd']
    cr = sat_info['cr']
    cross_section = sat_info['cross_section']
    mass = sat_info['mass']
    print(f"propagating from {start_date} to {end_date}")
    print(f'cr: {cr}, cd: {cd}, cross_section: {cross_section}, mass: {mass}')
    print(f'force model config: {force_model_config}')
    propagated_state_df = propagate_state(start_date, end_date, perturbed_state, cr, cd, cross_section, mass,boxwing=False,ephem=True,dt=5, **force_model_config)
    print(f'propagated state: {propagated_state_df}')
    print(f"Propagated state first row: {propagated_state_df.iloc[0]}")

    # Interpolate the propagated state to match the nominal collision data points
    t_col = start_date + datetime.timedelta(hours=12)
    propagated_state_df_interp = interpolate_ephemeris(propagated_state_df, t_col - datetime.timedelta(seconds=7), t_col + datetime.timedelta(seconds=7), stitch=True) #interpolate the trajectory finely around the TCA

    # Calculate distances - this is an example using a simple subtraction of vectors
    distances = np.linalg.norm(propagated_state_df_interp[['x', 'y', 'z']].values - nominal_collision_df[['x', 'y', 'z']].values, axis=1)
    min_distance = np.min(distances)
    min_distance_time = propagated_state_df_interp['UTC'][np.argmin(distances)]
    #make sure the time stamps of propagated_state_df_interp and nominal_collision_df are the same
    assert np.all(propagated_state_df_interp['UTC'] == nominal_collision_df['UTC'])

    print(f"Closest Recorded Approach: {np.min(distances)}")

    # Save the results
    pd.DataFrame({'UTC': [min_distance_time], 'Distance': [min_distance]}).to_csv(outpath, index=False)

if __name__ == "__main__":
    sat_name = sys.argv[1]
    force_model_num = int(sys.argv[2])
    perturbed_state_id = int(sys.argv[3])
    outpath = str(sys.argv[4])
    print(f'running with sat_name: {sat_name}, force_model_num: {force_model_num}, perturbed_state_id: {perturbed_state_id}, outpath: {outpath}')
    propagate_and_calculate(sat_name, perturbed_state_id, force_model_num, outpath)

