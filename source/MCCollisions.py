import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit, PositionAngleType
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from orekit import JArray_double
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants
from org.orekit.orbits import PositionAngleType, OrbitType
from org.orekit.utils import IERSConventions, PVCoordinates
from org.hipparchus.linear import MatrixUtils
from org.orekit.propagation import StateCovariance
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.ssa.collision.shorttermencounter.probability.twod import Patera2005

import os
from tools.utilities import get_satellite_info, pos_vel_from_orekit_ephem
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import configure_force_models, propagate_state, propagate_STM
from tools.BatchLeastSquares import OD_BLS
from tools.collision_tools import generate_collision_trajectory
import numpy as np
import numpy.random as npr
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 60.0
POSITION_TOLERANCE = 1e-2 # 1 cm

def generate_random_vectors(eigenvalues):
    random_vectors = []
    for lambda_val in eigenvalues:
        vector = [npr.normal(0, np.sqrt(lambda_val)) for _ in range(6)]
        random_vectors.append(vector)
    return random_vectors

def apply_perturbations(states, vectors, rotation_matrices):
    perturbed_states = []
    for state, vector_set, rotation_matrix in zip(states, vectors, rotation_matrices):
        for vector in vector_set:
            perturbation = np.dot(rotation_matrix, vector)
            perturbed_states.append(state + perturbation)
    return perturbed_states

#covariance associated with the collision trajectory (assuming good covariance)
secondary_covariance = [
    [4.7440894789163000000000, -1.2583279067770000000000, -1.2583279067770000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [-1.2583279067770000000000,6.1279552605419000000000, 2.1279552605419000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [-1.2583279067770000000000, 2.1279552605419000000000, 6.1279552605419000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000010000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000001, 0.0000010000000000000000, -0.0000000000000000000001],
    [0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000001, -0.0000000000000000000001, 0.0000010000000000000000]
]

def main():
    p_o_c_list = []
    sat_names_to_test = ["GRACE-FO-A"]
    arc_length = 5  # mins
    num_arcs = 1
    prop_length = 60 * 60 * 1.5  # around 2 orbital periods
    prop_length_days = prop_length / (60 * 60 * 24)
    force_model_configs = [
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'jb08drag': True}]

    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        sat_info = get_satellite_info(sat_name)
        cd = sat_info['cd']
        cr = sat_info['cr']
        cross_section = sat_info['cross_section']
        mass = sat_info['mass']
        ephemeris_df = ephemeris_df.iloc[::2, :] # downsample to 60 second intervals
        time_step = (ephemeris_df['UTC'].iloc[1] - ephemeris_df['UTC'].iloc[0]).total_seconds() / 60.0  # in minutes
        # find the time of the first time step
        t0 = ephemeris_df['UTC'].iloc[0]
        print(f't0: {t0}')
        t0_plus_24 = t0 + datetime.timedelta(days=prop_length_days) #this is the time at which the collision will occur
        print(f"t0_plus_24: {t0_plus_24}")
        collision_df = generate_collision_trajectory(ephemeris_df, t0_plus_24)

        arc_step = int(arc_length / time_step)
        for arc in range(num_arcs):
            
            start_index = arc * arc_step
            end_index = start_index + arc_step
            arc_df = ephemeris_df.iloc[start_index:end_index]

            observations_df = arc_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
            initial_values = arc_df.iloc[0][['x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
            initial_t = arc_df.iloc[0]['UTC']
            initial_vals = np.array(initial_values.tolist() + [cd, cr, cross_section, mass], dtype=float)

            a_priori_estimate = np.concatenate(([initial_t.timestamp()], initial_vals))

            for i, force_model_config in enumerate(force_model_configs):
                optimized_states, cov_mats, residuals, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag=False, max_patience=1)
                min_RMS_index = np.argmin(RMSs)
                optimized_state = optimized_states[min_RMS_index]
                optimized_state_cov = cov_mats[min_RMS_index]

            print("optimized_state_cov: ", optimized_state_cov)

            ##### Perturb the estimated state
            # Diagonalizing covariance matrices and generating perturbations
            eig_vals_primary, rotation_matrix_primary = np.linalg.eigh(optimized_state_cov)
            random_vectors_primary = generate_random_vectors(eig_vals_primary)
            print(f"random_vectors_primary: {random_vectors_primary}")
            # Perturbing the states and converting them back to the ECI frame
            perturbed_states_primary = apply_perturbations([optimized_state], [random_vectors_primary], [rotation_matrix_primary])
            print(f"perturbed_states_primary: {perturbed_states_primary}")

            #### Perturb the secondary state
            eig_vals_secondary, rotation_matrix_secondary = np.linalg.eigh(secondary_covariance)
            random_vectors_secondary = generate_random_vectors(eig_vals_secondary)
            print(f"random_vectors_secondary: {random_vectors_secondary}")
            #first state in collision_df is columns "x_col", "y_col", "z_col", "xv_col", "yv_col", "zv_col
            secondary_state = collision_df.iloc[0][["x_col", "y_col", "z_col", "xv_col", "yv_col", "zv_col"]].values
            perturbed_states_secondary = apply_perturbations([secondary_state], [random_vectors_secondary], [rotation_matrix_secondary])

            print(f"perturbed_states_secondary: {perturbed_states_secondary}")

            # propagate all the perturbed states
            # calculate the DCA between the collision_df and the perturbed states
            # plot the DCA for each perturbed state

        # Replace the usage of optimized_state with each of perturbed_states_primary in further computations
        # such as propagating states, computing distances, and calculating the probability of collision.

    print(p_o_c_list)

if __name__ == "__main__":
    main()