# Run OD at low and high solar activity
# Use force full fidelity and swap out the CERES and Knocke ERP models
# Estimate C_d as part of every arc 

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
import uuid

INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 60.0
INTEGRATOR_INIT_STEP = 15.0
POSITION_TOLERANCE = 1e-2 # 1 cm

def generate_config_name(config_dict, arc_number):
    config_keys = '+'.join(key for key, value in config_dict.items() if value)
    return f"arc{arc_number}_{config_keys}"

def main():
    sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    num_arcs = 1
    arc_length = 60 #mins
    estimate_drag = True
    boxwing = False
    force_model_configs = [

        {'120x120gravity': True, 'knocke_erp': True, 'SRP': True, 'nrlmsise00drag': True},
        {'120x120gravity': True, 'ceres_erp': True, 'SRP': True, 'nrlmsise00drag': True},
        {'120x120gravity': True, 'knocke_erp': True,'SRP': True, 'jb08drag': True},
        {'120x120gravity': True, 'ceres_erp': True, 'SRP': True, 'jb08drag': True},
        {'120x120gravity': True, 'SRP': True, 'jb08drag': True},
        {'120x120gravity': True, 'SRP': True, 'jb08drag': True}]

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
        ephemeris_df = ephemeris_df.iloc[::4, :]
        #slice the ephemeris to start 12 arcs past the beginning
        # ephemeris_df = ephemeris_df.iloc[12*arc_length:]
        time_step = (ephemeris_df['UTC'].iloc[1] - ephemeris_df['UTC'].iloc[0]).total_seconds() / 60.0  # in minutes
        arc_step = int(arc_length / time_step)

        cd_estimates = {}

        for arc in range(num_arcs):
            start_index = arc * arc_step
            end_index = start_index + arc_step
            arc_df = ephemeris_df.iloc[start_index:end_index]

            initial_values = arc_df.iloc[0][['x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
            initial_t = arc_df.iloc[0]['UTC']
            initial_vals = np.array(initial_values.tolist() + [cd, cr, cross_section, mass], dtype=float)

            a_priori_estimate = np.concatenate(([initial_t.timestamp()], initial_vals))
            observations_df = arc_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]

            for i, force_model_config in enumerate(force_model_configs):
                print(f"starting BLS with initial values: {a_priori_estimate} and force model: {force_model_config} for arc {arc + 1} of {sat_name}...")
                optimized_states, _, _, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag, max_patience=1, boxwing=boxwing_model)
                #save the force model, arc length, and prop length in a file
                min_RMS_index = np.argmin(RMSs)
                optimized_state = optimized_states[min_RMS_index]
                # Assume combined_residuals_plot and other necessary functions are defined here
                cd_value = optimized_state[6]
                print(f"Estimated C_D: {cd_value}")
                config_name = generate_config_name(force_model_config, arc + 1)
                cd_estimates[config_name] = cd_estimates.get(config_name, []) + [cd_value]
                cd = sat_info['cd'] #reset the drag coefficient to the true value
                print(f"True C_D: {cd}")
                # combined_residuals_plot(observations_df, residuals_final, a_priori_estimate, optimized_state, force_model_config, RMSs[min_RMS_index], sat_name, i, arc, estimate_drag)

            print(f"all CD estimates: {cd_estimates}")
            all_cd_values = [cd for sublist in cd_estimates.values() for cd in sublist]

            plt.figure(figsize=(10, 6))
            plt.hist(all_cd_values, bins=20, color='blue', alpha=0.7)
            plt.title('Histogram of Estimated CD Values')
            plt.xlabel('CD Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
            # Output directory for each arc
            output_dir = f"output/OD_BLS/Tapley/prop_estim_states/{sat_name}/arc{arc + 1}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

#TODO: gravity, drag and SRP and ERP models
#TODO: use percentage forcing difference from Ray paper as basis for comparison
#TODO: see if the relative importance of ERP is reflected in the estimated drag coefficient

if __name__ == "__main__":
    main()