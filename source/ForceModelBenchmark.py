import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir("misc/orekit-data.zip")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.utils import PVCoordinates
from org.orekit.orbits import CartesianOrbit
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from orekit import JArray_double
from org.orekit.orbits import OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants

import os
from tools.utilities import HCL_diff, get_satellite_info, pos_vel_from_orekit_ephem
from tools.plotting import combined_residuals_plot
from tools.orekit_tools import configure_force_models
from tools.BatchLeastSquares import OD_BLS
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import uuid
import sys

INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 60.0
POSITION_TOLERANCE = 1e-2 # 1 cm

def generate_config_name(config_dict, arc_number):
    config_keys = '+'.join(key for key, value in config_dict.items() if value)
    return f"arc{arc_number}_{config_keys}"

def benchmark(folder_path, sat_name, OD_points, OP_reference_trajectory, prop_length, arc_number, force_model_config):
    #arc_number included for naming purposes
    estimate_drag = False
    boxwing = False
    
    sat_info = get_satellite_info(sat_name)
    cd = sat_info['cd']
    cr = sat_info['cr']
    cross_section = sat_info['cross_section']
    mass = sat_info['mass']
    time_step = (OD_points['UTC'].iloc[1] - OD_points['UTC'].iloc[0]).total_seconds() / 60.0  # in minutes. Assuming equal time steps
    time_step_seconds = time_step * 60.0

    diffs_3d_abs_results = {}
    hcl_differences = {'H': {}, 'C': {}, 'L': {}}

    initial_values = OD_points.iloc[0][['x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
    initial_t = OD_points.iloc[0]['UTC']
    final_prop_t = initial_t + datetime.timedelta(seconds=prop_length)
    prop_observations_df = OP_reference_trajectory[(OP_reference_trajectory['UTC'] >= initial_t) & (OP_reference_trajectory['UTC'] <= final_prop_t)]
    initial_vals = np.array(initial_values.tolist() + [cd, cr, cross_section, mass], dtype=float)

    a_priori_estimate = np.concatenate(([initial_t.timestamp()], initial_vals))
    observations_df = OD_points[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
    optimized_states, cov_mats, residuals, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag, max_patience=1, boxwing=boxwing)
    initial_t_str = initial_t.strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"{folder_path}/{sat_name}/arc{arc_number+1}#pts{len(observations_df)}estdrag{estimate_drag}_{initial_t_str}_fm{force_model_config}" # Add uuid to avoid overwriting
    os.makedirs(output_folder, exist_ok=True)
    np.save(f"{output_folder}/optimized_states.npy", optimized_states)
    np.save(f"{output_folder}/cov_mats.npy", cov_mats)
    np.save(f"{output_folder}/ODresiduals.npy", residuals)
    np.save(f"{output_folder}/RMSs.npy", RMSs)

    #save the force model, arc length, and prop length in a file
    min_RMS_index = np.argmin(RMSs)
    optimized_state = optimized_states[min_RMS_index]
    residuals_final = residuals[min_RMS_index]

    combined_residuals_plot(observations_df, residuals_final, a_priori_estimate, optimized_state, force_model_config, RMSs[min_RMS_index], sat_name, i, arc_number, estimate_drag)
    
    optimized_state_orbit = CartesianOrbit(PVCoordinates(Vector3D(float(optimized_state[0]), float(optimized_state[1]), float(optimized_state[2])),
                                                        Vector3D(float(optimized_state[3]), float(optimized_state[4]), float(optimized_state[5]))),
                                            FramesFactory.getEME2000(),
                                            datetime_to_absolutedate(initial_t),
                                            Constants.WGS84_EARTH_MU)
    
    tolerances = NumericalPropagator.tolerances(POSITION_TOLERANCE, optimized_state_orbit, optimized_state_orbit.getType())
    integrator = DormandPrince853Integrator(INTEGRATOR_MIN_STEP, INTEGRATOR_MAX_STEP, JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(INTEGRATOR_INIT_STEP)
    initialState = SpacecraftState(optimized_state_orbit, mass)
    optimized_state_propagator = NumericalPropagator(integrator)
    optimized_state_propagator.setOrbitType(OrbitType.CARTESIAN)
    optimized_state_propagator.setInitialState(initialState)

    print(f"propagating with force model config: {force_model_config}")
    optimized_state_propagator = configure_force_models(optimized_state_propagator, cr, cross_section, cd, boxwing, **force_model_config)
    ephemGen_optimized = optimized_state_propagator.getEphemerisGenerator()
    end_state_optimized = optimized_state_propagator.propagate(datetime_to_absolutedate(initial_t), datetime_to_absolutedate(final_prop_t))
    ephemeris = ephemGen_optimized.getGeneratedEphemeris()

    times, state_vectors = pos_vel_from_orekit_ephem(ephemeris, datetime_to_absolutedate(initial_t), datetime_to_absolutedate(final_prop_t), time_step_seconds)
    state_vector_data = (times, state_vectors)

    observation_state_vectors = prop_observations_df[['x', 'y', 'z', 'xv', 'yv', 'zv']].values
    observation_times = pd.to_datetime(prop_observations_df['UTC'].values)

    # Compare the propagated state vectors with the observations
    config_name = generate_config_name(force_model_config, arc_number + 1)
    diffs_3d_abs = []
    for state_vector, observation_state_vector in zip(state_vectors, observation_state_vectors):
        diff_3d_abs = np.sqrt(np.mean(np.square(state_vector[:3] - observation_state_vector[:3])))
        diffs_3d_abs.append(diff_3d_abs)
    diffs_3d_abs_results[config_name] = diffs_3d_abs_results.get(config_name, []) + [diffs_3d_abs]
    h_diffs, c_diffs, l_diffs = HCL_diff(state_vectors, observation_state_vectors)
    hcl_differences['H'][config_name] = hcl_differences['H'].get(config_name, []) + [h_diffs]
    hcl_differences['C'][config_name] = hcl_differences['C'].get(config_name, []) + [c_diffs]
    hcl_differences['L'][config_name] = hcl_differences['L'].get(config_name, []) + [l_diffs]
    arc_length = len(observations_df)
    with open(f"{output_folder}/run_config.txt", "w") as f:
        f.write(f"intial_t: {initial_t}\n")
        f.write(f"final_prop_t: {final_prop_t}\n")
        f.write(f"sat_name: {sat_name}\n")
        f.write(f"force_model_config: {force_model_config}\n")
        f.write(f"arc_length: {arc_length}\n")
        f.write(f"prop_length: {prop_length}\n")
        f.write(f"estimate_drag: {estimate_drag}\n")
        f.write(f"boxwing: {boxwing}\n")

    #save arc-specific results
    np.save(f"{output_folder}_hcl_diffs.npy", hcl_differences)
    np.save(f"{output_folder}_prop_residuals.npy", diffs_3d_abs_results) #These are not the residuals from the OD fitting process, but from the propagation
    np.savez(f"{output_folder}_state_vector_data.npz", times=state_vector_data[0], state_vectors=state_vector_data[1])

    # Output directory for each arc
    output_dir = f"output/OD_BLS/Tapley/prop_estim_states/{sat_name}/arc{arc_number + 1}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for diff_type in ['H', 'C', 'L']:
        fig, ax = plt.subplots(figsize=(8, 4))
        vertical_offset = 0

        for config_name, diffs in hcl_differences[diff_type].items():
            flat_diffs = np.array(diffs).flatten()
            line, = ax.plot(observation_times, flat_diffs, label=config_name)
            ax.text(0.02, 0.98 - vertical_offset, f'{config_name}: {flat_diffs[-1]:.2f}', 
                    transform=ax.transAxes, color=line.get_color(), verticalalignment='top', fontsize=10)
            vertical_offset += 0.07

        ax.set_title(f'{diff_type} Differences for {sat_name}')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{diff_type} Difference')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{diff_type}_diff_{arc_length}obs_{prop_length}prop_{initial_t}.png")
        print(f"saved {diff_type} differences plot for {sat_name} to {output_dir}")
        plt.close()

#     output_dir = f"output/OD_BLS/Tapley/prop_estim_states/{sat_name}/arc{arc_number + 1}"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     plt.figure(figsize=(10, 6))
#     sns.set_palette(sns.color_palette("bright", len(diffs_3d_abs_results)))

#     for i, (config_name, rms_values_list) in enumerate(diffs_3d_abs_results.items()):
#         if len(rms_values_list) > arc_number:
#             rms_values = rms_values_list[arc_number]
#             plt.scatter(observation_times, rms_values, label=config_name, s=3, alpha=0.7)

#     plt.xlabel('Time')
#     plt.ylabel('RMS (m)')
#     plt.title(f'RMS Differences for {sat_name} - Arc {arc_number + 1}')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/RMS_diff_{arc_length}obs_{prop_length}prop_arc{arc_number + 1}.png", bbox_inches='tight')
#     plt.close()

# def main():
#     # sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
#     sat_names_to_test = ["Sentinel-1A", "Sentinel-3B"]
#     # dates_to_test = ["2019-01-01", "2023-05-04"]
#     dates_to_test = ["2023-05-04"]
#     for sat_name in sat_names_to_test:
#         for date in dates_to_test:
#             benchmark(sat_name, date)

if __name__ == "__main__":
    sat_name = sys.argv[0]
    #ensure sat name is a string
    assert isinstance(sat_name, str), "sat_name must be a string"
    OD_points = sys.argv[1]
    assert isinstance(OD_points, pd.DataFrame), "OD_points must be a pandas dataframe"
    OP_reference_trajectory = sys.argv[2]
    assert isinstance(OP_reference_trajectory, pd.DataFrame), "OP_reference_trajectory must be a pandas dataframe"
    prop_length = sys.argv[3]
    assert isinstance(prop_length, int), "prop_length must be an integer"
    arc_number  = sys.argv[4]
    assert isinstance(arc_number, int), "arc_number must be an integer"
    force_model_config = sys.argv[5:]
    assert isinstance(force_model_config, dict), "force_model_config must be a dictionary"