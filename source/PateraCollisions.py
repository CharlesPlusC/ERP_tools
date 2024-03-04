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
from tools.utilities import build_boxwing, HCL_diff,build_boxwing, get_satellite_info, pos_vel_from_orekit_ephem
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import configure_force_models, propagate_state, propagate_STM
from tools.BatchLeastSquares import OD_BLS
from tools.collision_tools import generate_collision_trajectory, patera05_poc
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


INTEGRATOR_MIN_STEP = 0.001
INTEGRATOR_MAX_STEP = 15.0
INTEGRATOR_INIT_STEP = 60.0
POSITION_TOLERANCE = 1e-2 # 1 cm

def plot_tca_dca(merged_df, sp3_to_opt_distances, sp3_to_kep_distances, opt_to_kep_distances, sat_name, force_model_config, min_sp3_to_kep_distance, time_of_closest_approach_sp3_to_kep, min_opt_to_kep_distance, time_of_closest_approach_opt_to_kep):
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    # First subplot
    ax[0].plot(merged_df['UTC'], sp3_to_opt_distances, label='Distance of optimized orbit to true trajectory')
    ax[0].set_xlabel('UTC')
    ax[0].set_ylabel('Distance (m)')
    ax[0].set_title(f"{sat_name} - {force_model_config}")
    ax[0].legend()
    ax[0].grid(True)

    # Second subplot
    ax[1].plot(merged_df['UTC'], sp3_to_kep_distances, label='Distance of true trajectory to secondary trajectory')
    ax[1].axhline(y=min_sp3_to_kep_distance, color='r', linestyle='--', label=f'Closest Approach: {min_sp3_to_kep_distance} m')
    ax[1].axvline(x=time_of_closest_approach_sp3_to_kep, color='g', linestyle='--', label=f'Time of Closest Approach: {time_of_closest_approach_sp3_to_kep}')
    ax[1].set_yscale('log')
    ax[1].grid(which='both')
    ax[1].set_xlabel('UTC')
    ax[1].set_ylabel('Distance (m)')
    ax[1].legend()

    # Third subplot
    ax[2].plot(merged_df['UTC'], opt_to_kep_distances, label='Optimized to secondary trajectory distance')
    ax[2].axhline(y=min_opt_to_kep_distance, color='r', linestyle='--', label=f'Closest Approach: {min_opt_to_kep_distance} m')
    ax[2].axvline(x=time_of_closest_approach_opt_to_kep, color='g', linestyle='--', label=f'Time of Closest Approach: {time_of_closest_approach_opt_to_kep}')
    ax[2].set_yscale('log')
    ax[2].grid(which='both')
    ax[2].set_xlabel('UTC')
    ax[2].set_ylabel('Distance (m)')
    ax[2].legend()

    plt.tight_layout()
    file_name = f"{sat_name}_{force_model_config}_tca_dca.png"
    folder = "output/Collisions"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, file_name))
    # plt.show()

good_covariance = [
    [4.7440894789163000000000, -1.2583279067770000000000, -1.2583279067770000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [-1.2583279067770000000000,6.1279552605419000000000, 2.1279552605419000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [-1.2583279067770000000000, 2.1279552605419000000000, 6.1279552605419000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000010000000000000000, 0.0000000000000000000000, 0.0000000000000000000000],
    [0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000001, 0.0000010000000000000000, -0.0000000000000000000001],
    [0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000000, 0.0000000000000000000001, -0.0000000000000000000001, 0.0000010000000000000000]
]

import numpy as np
import numpy.random as npr

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

def main():
    p_o_c_list = []
    sat_names_to_test = ["GRACE-FO-A"]
    num_arcs = 1
    arc_length = 5  # mins
    prop_length = 60 * 60 * 1.5  # around 2 orbital periods
    force_model_configs = [
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},
        {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'jb08drag': True},
    ]

    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        sat_info = get_satellite_info(sat_name)

        for arc in range(num_arcs):
            # Similar to the original code until we get optimized_state and optimized_state_cov

            # Diagonalizing covariance matrices and generating perturbations
            eig_vals_primary, rotation_matrix_primary = np.linalg.eigh(optimized_state_cov)
            random_vectors_primary = generate_random_vectors(eig_vals_primary)

            # Perturbing the states and converting them back to the ECI frame
            perturbed_states_primary = apply_perturbations([optimized_state], [random_vectors_primary], [rotation_matrix_primary])

            # Replace the usage of optimized_state with each of perturbed_states_primary in further computations
            # such as propagating states, computing distances, and calculating the probability of collision.

    print(p_o_c_list)

if __name__ == "__main__":
    main()