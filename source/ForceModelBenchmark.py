import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
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
from tools.utilities import build_boxwing, HCL_diff,build_boxwing, get_satellite_info, pos_vel_from_orekit_ephem
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.plotting import combined_residuals_plot
from tools.orekit_tools import configure_force_models
from tools.BatchLeastSquares import OD_BLS
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 60.0
POSITION_TOLERANCE = 1e-2 # 1 cm

def generate_config_name(config_dict, arc_number):
    config_keys = '+'.join(key for key, value in config_dict.items() if value)
    return f"arc{arc_number}_{config_keys}"

def main():
    sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    num_arcs = 6
    arc_length = 45 #mins
    prop_length = 60 * 60 * 6 #seconds
    estimate_drag = False
    boxwing = False
    force_model_configs = [
        # {'gravity': True},
        {'36x36gravity': True, '3BP': True},
        {'120x120gravity': True, '3BP': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'jb08drag': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'dtm2000drag': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'nrlmsise00drag': True}
    ]

    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        sat_info = get_satellite_info(sat_name)
        if boxwing:
            boxwing_model = build_boxwing(sat_name)
        else:
            boxwing_model = None
        cd = sat_info['cd']
        cr = sat_info['cr']
        cross_section = sat_info['cross_section']
        mass = sat_info['mass']
        ephemeris_df = ephemeris_df.iloc[::2, :]
        #slice the ephemeris to start 6 arcs past the beginning
        ephemeris_df = ephemeris_df.iloc[6*arc_length:]
        time_step = (ephemeris_df['UTC'].iloc[1] - ephemeris_df['UTC'].iloc[0]).total_seconds() / 60.0  # in minutes
        time_step_seconds = time_step * 60.0
        arc_step = int(arc_length / time_step)

        diffs_3d_abs_results = {}  
        cd_estimates = {}

        for arc in range(num_arcs):
            hcl_differences = {'H': {}, 'C': {}, 'L': {}}
            start_index = arc * arc_step
            end_index = start_index + arc_step
            arc_df = ephemeris_df.iloc[start_index:end_index]

            initial_values = arc_df.iloc[0][['x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
            initial_t = arc_df.iloc[0]['UTC']
            final_prop_t = initial_t + datetime.timedelta(seconds=prop_length)
            prop_observations_df = ephemeris_df[(ephemeris_df['UTC'] >= initial_t) & (ephemeris_df['UTC'] <= final_prop_t)]
            initial_vals = np.array(initial_values.tolist() + [cd, cr, cross_section, mass], dtype=float)

            a_priori_estimate = np.concatenate(([initial_t.timestamp()], initial_vals))
            observations_df = arc_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]

            for i, force_model_config in enumerate(force_model_configs):
                if not force_model_config.get('jb08drag', False) and not force_model_config.get('dtm2000drag', False) and not force_model_config.get('nrlmsise00drag', False):
                    estimate_drag = False

                optimized_states, cov_mats, residuals, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag, max_patience=1, boxwing=boxwing_model)
                date_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                folder_path = "output/OD_BLS/Tapley/saved_runs"
                initial_t_str = initial_t.strftime("%Y-%m-%d_%H-%M-%S")  # Format datetime
                output_folder = f"{folder_path}/{sat_name}/fm{i+1}arc{arc+1}#pts{len(observations_df)}estdrag{estimate_drag}_{initial_t_str}"
                os.makedirs(output_folder, exist_ok=True)
                np.save(f"{output_folder}/optimized_states.npy", optimized_states)
                np.save(f"{output_folder}/cov_mats.npy", cov_mats)
                np.save(f"{output_folder}/ODresiduals.npy", residuals)
                np.save(f"{output_folder}/RMSs.npy", RMSs)
                #save the force model, arc length, and prop length in a file
                min_RMS_index = np.argmin(RMSs)
                optimized_state = optimized_states[min_RMS_index]
                residuals_final = residuals[min_RMS_index]
                # Assume combined_residuals_plot and other necessary functions are defined here
                if estimate_drag:
                    # C_D value is the 7th element in optimized_state
                    cd_value = optimized_state[6]
                    cd = cd_value
                    config_name = generate_config_name(force_model_config, arc + 1)
                    cd_estimates[config_name] = cd_estimates.get(config_name, []) + [cd_value]

                combined_residuals_plot(observations_df, residuals_final, a_priori_estimate, optimized_state, force_model_config, RMSs[min_RMS_index], sat_name, i, arc, estimate_drag)
                # Propagation and RMS calculation logic
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

                # Add all the force models
                print(f"propagating with force model config: {force_model_config}")
                optimized_state_propagator = configure_force_models(optimized_state_propagator, cr, cross_section, cd,boxwing, **force_model_config)
                ephemGen_optimized = optimized_state_propagator.getEphemerisGenerator()
                end_state_optimized = optimized_state_propagator.propagate(datetime_to_absolutedate(initial_t), datetime_to_absolutedate(final_prop_t))
                ephemeris = ephemGen_optimized.getGeneratedEphemeris()

                times, state_vectors = pos_vel_from_orekit_ephem(ephemeris, datetime_to_absolutedate(initial_t), datetime_to_absolutedate(final_prop_t), time_step_seconds)
                state_vector_data = (times, state_vectors)

                observation_state_vectors = prop_observations_df[['x', 'y', 'z', 'xv', 'yv', 'zv']].values
                observation_times = pd.to_datetime(prop_observations_df['UTC'].values)

                # Compare the propagated state vectors with the observations
                config_name = generate_config_name(force_model_config, arc + 1)
                diffs_3d_abs = []
                for state_vector, observation_state_vector in zip(state_vectors, observation_state_vectors):
                    diff_3d_abs = np.sqrt(np.mean(np.square(state_vector[:3] - observation_state_vector[:3])))
                    diffs_3d_abs.append(diff_3d_abs)
                diffs_3d_abs_results[config_name] = diffs_3d_abs_results.get(config_name, []) + [diffs_3d_abs]
                h_diffs, c_diffs, l_diffs = HCL_diff(state_vectors, observation_state_vectors)
                hcl_differences['H'][config_name] = hcl_differences['H'].get(config_name, []) + [h_diffs]
                hcl_differences['C'][config_name] = hcl_differences['C'].get(config_name, []) + [c_diffs]
                hcl_differences['L'][config_name] = hcl_differences['L'].get(config_name, []) + [l_diffs]
                
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

            if estimate_drag:
                np.save(f"{output_folder}/cd_estimates.npy", cd_estimates)

            # Output directory for each arc
            output_dir = f"output/OD_BLS/Tapley/prop_estim_states/{sat_name}/arc{arc + 1}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)


            for diff_type in ['H', 'C', 'L']:
                fig, ax = plt.subplots(figsize=(8, 4))  # Adjusted figure size for better layout
                vertical_offset = 0  # Starting offset for text annotations

                for config_name, diffs in hcl_differences[diff_type].items():
                    flat_diffs = np.array(diffs).flatten()
                    line, = ax.plot(observation_times, flat_diffs, label=config_name)  # Keep a reference to the line object

                    # Annotate the last point in the top left corner
                    ax.text(0.02, 0.98 - vertical_offset, f'{config_name}: {flat_diffs[-1]:.2f}', 
                            transform=ax.transAxes, color=line.get_color(), verticalalignment='top', fontsize=10)
                    vertical_offset += 0.07  # Increment offset for the next line

                ax.set_title(f'{diff_type} Differences for {sat_name}')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'{diff_type} Difference')

                # Place legend below the graph
                # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3, fontsize='small')

                plt.tight_layout()
                plt.savefig(f"{output_dir}/{diff_type}_diff_{arc_length}obs_{prop_length}prop_{initial_t}.png")
                print(f"saved {diff_type} differences plot for {sat_name} to {output_dir}")
                plt.close()

    # Separate RMS plot for each arc
    for arc in range(num_arcs):
        output_dir = f"output/OD_BLS/Tapley/prop_estim_states/{sat_name}/arc{arc + 1}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(10, 6))
        sns.set_palette(sns.color_palette("bright", len(diffs_3d_abs_results)))  # Change to a brighter color palette

        for i, (config_name, rms_values_list) in enumerate(diffs_3d_abs_results.items()):
            if len(rms_values_list) > arc:
                rms_values = rms_values_list[arc]
                plt.scatter(observation_times, rms_values, label=config_name, s=3, alpha=0.7)

        plt.xlabel('Time')
        plt.ylabel('RMS (m)')
        plt.title(f'RMS Differences for {sat_name} - Arc {arc + 1}')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/RMS_diff_{arc_length}obs_{prop_length}prop_arc{arc + 1}.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()