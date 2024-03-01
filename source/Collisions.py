import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.utils import PVCoordinates
from org.orekit.orbits import CartesianOrbit, KeplerianOrbit, PositionAngleType
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from orekit import JArray_double
from org.orekit.orbits import OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants

import os
from tools.utilities import build_boxwing, HCL_diff,build_boxwing, get_satellite_info, pos_vel_from_orekit_ephem
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import configure_force_models, propagate_state, propagate_STM
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

def main():
    sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    num_arcs = 6
    arc_length = 10 #mins
    prop_length = 60 * 60 * 6 #seconds
    prop_length_days = prop_length / (60 * 60 * 24)
    force_model_configs = [
        # {'gravity': True},
        # {'36x36gravity': True, '3BP': True},
        # {'120x120gravity': True, '3BP': True},
        # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True},
        # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True},
        # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True},
        # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'jb08drag': True},
        # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'dtm2000drag': True},
        # {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'nrlmsise00drag': True}
    ]

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
        t0_plus_24_plus_t = t0_plus_24 + datetime.timedelta(minutes=45) #this is the time at which we want the propagation to end
        #slice ephemeris_df so that it stops at t0_plus_24_plus_t
        ephemeris_df = ephemeris_df[(ephemeris_df['UTC'] >= t0) & (ephemeris_df['UTC'] <= t0_plus_24_plus_t)]
        print(f"t0_plus_24_plus_t: {t0_plus_24_plus_t}")
        
        pvcoords_t24 = PVCoordinates(Vector3D(float(ephemeris_df[ephemeris_df['UTC'] == t0_plus_24]['x']),
                                                float(ephemeris_df[ephemeris_df['UTC'] == t0_plus_24]['y']),
                                                float(ephemeris_df[ephemeris_df['UTC'] == t0_plus_24]['z'])),
                                        Vector3D(float(ephemeris_df[ephemeris_df['UTC'] == t0_plus_24]['xv']),
                                                float(ephemeris_df[ephemeris_df['UTC'] == t0_plus_24]['yv']),
                                                float(ephemeris_df[ephemeris_df['UTC'] == t0_plus_24]['zv'])))
        

        inverted_velocities = Vector3D(-float(ephemeris_df[ephemeris_df['UTC'] == t0_plus_24]['xv']),
                                        -float(ephemeris_df[ephemeris_df['UTC'] == t0_plus_24]['yv']),
                                        -float(ephemeris_df[ephemeris_df['UTC'] == t0_plus_24]['zv']))

        inverted_pvcoords_t24 = PVCoordinates(pvcoords_t24.getPosition(), inverted_velocities)
        inverted_kepler24 = KeplerianOrbit(inverted_pvcoords_t24,
                                FramesFactory.getEME2000(),
                                datetime_to_absolutedate(t0_plus_24),
                                Constants.WGS84_EARTH_MU)
                                           
        propagator24 = KeplerianPropagator(inverted_kepler24)
        ephemeris_generator24 = propagator24.getEphemerisGenerator()
        propagator24.propagate(datetime_to_absolutedate(t0_plus_24_plus_t), datetime_to_absolutedate(t0))
        ephemeris24 = ephemeris_generator24.getGeneratedEphemeris()
        kepler_prop_times24, kepler_prop_state_vectors24 = pos_vel_from_orekit_ephem(ephemeris24, datetime_to_absolutedate(t0_plus_24_plus_t), datetime_to_absolutedate(t0), 60.0)
        #use kepler_prop_times0 and t0 to construct the UTC times for the ephemeris_df
        kepler_prop_times24_dt = [t0_plus_24_plus_t + datetime.timedelta(seconds=sec) for sec in kepler_prop_times24]
        kepler_times_df = pd.DataFrame({'UTC': pd.to_datetime(kepler_prop_times24_dt)})
        kepler_states_df = pd.DataFrame(kepler_prop_state_vectors24, columns=['x_kep', 'y_kep', 'z_kep', 'xv_kep', 'yv_kep', 'zv_kep'])
        ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
        collision_df = pd.merge_asof(ephemeris_df.sort_values('UTC'), kepler_times_df.sort_values('UTC').join(kepler_states_df), on='UTC')
    
        # Calculating the 3D cartesian distance between the two orbits for the last 50 points
        sp3_to_kep_distances = np.sqrt((collision_df['x'] - collision_df['x_kep'])**2 + (collision_df['y'] - collision_df['y_kep'])**2 + (collision_df['z'] - collision_df['z_kep'])**2)

        arc_step = int(arc_length / time_step)
        for arc in range(num_arcs):
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

                optimized_states, cov_mats, residuals, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag=False, max_patience=1)
                min_RMS_index = np.argmin(RMSs)
                optimized_state = optimized_states[min_RMS_index]
                optimized_state_cov = cov_mats[min_RMS_index]
                print(f"optimized_state: {optimized_state}")
                print(f"optimized_state_cov: {optimized_state_cov}")

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
                optimized_state_propagator = configure_force_models(optimized_state_propagator, cr, cross_section, cd, False, **force_model_config)
                ephemGen_optimized = optimized_state_propagator.getEphemerisGenerator()
                optimized_state_propagator.propagate(datetime_to_absolutedate(initial_t), datetime_to_absolutedate(t0_plus_24_plus_t))
                ephemeris = ephemGen_optimized.getGeneratedEphemeris()

                times, state_vectors = pos_vel_from_orekit_ephem(ephemeris, datetime_to_absolutedate(initial_t), datetime_to_absolutedate(t0_plus_24_plus_t), 60.0)
                optimized_times_df = pd.DataFrame({'UTC': pd.to_datetime([initial_t + datetime.timedelta(seconds=sec) for sec in times])})
                #add the newly calculated state vectors to the collision_df using UTC as the thing to merge on (x_new, y_new, z_new, xv_new, yv_new, zv_new)
                optimized_states_df = pd.DataFrame(state_vectors, columns=['x_opt', 'y_opt', 'z_opt', 'xv_opt', 'yv_opt', 'zv_opt'])
                merged_df = pd.merge_asof(collision_df.sort_values('UTC'), optimized_times_df.sort_values('UTC').join(optimized_states_df), on='UTC')
                # Calculating the 3D cartesian distance between the two orbits for the last 50 points
                sp3_to_opt_distances = np.sqrt((merged_df['x'] - merged_df['x_opt'])**2 + (merged_df['y'] - merged_df['y_opt'])**2 + (merged_df['z'] - merged_df['z_opt'])**2)
                #now calculate the opt to kep distances
                opt_to_kep_distances = np.sqrt((merged_df['x_opt'] - merged_df['x_kep'])**2 + (merged_df['y_opt'] - merged_df['y_kep'])**2 + (merged_df['z_opt'] - merged_df['z_kep'])**2)
                print(f"min opt_to_kep_distances distance: {min(opt_to_kep_distances)}")
                print(f"min sp3_to_opt_distances distance: {min(sp3_to_opt_distances)}")

                # Calculate the minimum distance and the corresponding time for both cases
                min_sp3_to_kep_distance = min(sp3_to_kep_distances)
                time_of_closest_approach_sp3_to_kep = merged_df['UTC'][sp3_to_kep_distances.idxmin()]
                print(f"min sp3_to_kep_distances distance: {min_sp3_to_kep_distance}")
                print(f"time_of_closest_approach_sp3_to_kep: {time_of_closest_approach_sp3_to_kep}")

                min_opt_to_kep_distance = min(opt_to_kep_distances)
                time_of_closest_approach_opt_to_kep = merged_df['UTC'][opt_to_kep_distances.idxmin()]
                print(f"min opt_to_kep_distances distance: {min_opt_to_kep_distance}")
                print(f"time_of_closest_approach_opt_to_kep: {time_of_closest_approach_opt_to_kep}")

                # now propagate the covariance matrix
                # Initialize the STM at t0 as an identity matrix
                phi_i = np.identity(len(optimized_state_cov))
                # Propagate the STM to the final time
                phi_t1 = propagate_STM(optimized_state, initial_t, final_prop_t - initial_t, phi_i, cr, cd, cross_section, mass, estimate_drag=False, **force_model_config)
                print(f"phi_t1: {phi_t1}")
                # Propagate the covariance matrix
                optimized_state_cov_t1 = phi_t1 @ optimized_state_cov @ phi_t1.T
                print(f"optimized_state_cov_t1: {optimized_state_cov_t1}")
                print(f"initial_state_cov: {np.diag(optimized_state_cov)}")
                
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
                # Add horizontal line for Distance of Closest Approach
                ax[1].axhline(y=min_sp3_to_kep_distance, color='r', linestyle='--', label=f'Closest Approach: {min_sp3_to_kep_distance} m')
                # Add vertical line for Time of Closest Approach
                ax[1].axvline(x=time_of_closest_approach_sp3_to_kep, color='g', linestyle='--', label=f'Time of Closest Approach: {time_of_closest_approach_sp3_to_kep}')
                ax[1].set_yscale('log')
                ax[1].grid(which='both')
                ax[1].set_xlabel('UTC')
                ax[1].set_ylabel('Distance (m)')
                ax[1].legend()

                # Third subplot
                ax[2].plot(merged_df['UTC'], opt_to_kep_distances, label='Optimized to secondary trajectory distance')
                # Add horizontal line for Distance of Closest Approach
                ax[2].axhline(y=min_opt_to_kep_distance, color='r', linestyle='--', label=f'Closest Approach: {min_opt_to_kep_distance} m')
                # Add vertical line for Time of Closest Approach
                ax[2].axvline(x=time_of_closest_approach_opt_to_kep, color='g', linestyle='--', label=f'Time of Closest Approach: {time_of_closest_approach_opt_to_kep}')
                ax[2].set_yscale('log')
                ax[2].grid(which='both')
                ax[2].set_xlabel('UTC')
                ax[2].set_ylabel('Distance (m)')
                ax[2].legend()

                plt.tight_layout()
                plt.show()

                #select the state and covariance matrix that corresponds to the iteration with the lowest RMS
                #check the optimized state is in the format the initial_state_vector wants it
                #set start_date to t0, then set end_date to t0_plus_24 (+10 mins so we have a bit of a buffer for plotting)
                # set phi0 to the identity matrix?

                # state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass,boxwing=None, **force_model_config)
                # phi_ti = propagate_STM(state_ti_minus1, ti, dt, phi_ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass,boxwing=None, **force_model_config)

if __name__ == "__main__":
    main()