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
    arc_length = 45 #mins
    prop_length = 60 * 60 * 6 #seconds
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
        cd = sat_info['cd']
        cr = sat_info['cr']
        cross_section = sat_info['cross_section']
        mass = sat_info['mass']
        ephemeris_df = ephemeris_df.iloc[::2, :]
        time_step = (ephemeris_df['UTC'].iloc[1] - ephemeris_df['UTC'].iloc[0]).total_seconds() / 60.0  # in minutes
        # find the time of the first time step
        t0 = ephemeris_df['UTC'].iloc[0]
        t0_plus_24 = t0 + datetime.timedelta(days=1)
        
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

        kepler24 = KeplerianOrbit(pvcoords_t24,
                                FramesFactory.getEME2000(),
                                datetime_to_absolutedate(t0_plus_24),
                                Constants.WGS84_EARTH_MU)
        

        inverted_kepler24 = KeplerianOrbit(inverted_pvcoords_t24,
                                FramesFactory.getEME2000(),
                                datetime_to_absolutedate(t0_plus_24),
                                Constants.WGS84_EARTH_MU)
                                           

        # new_inclination = np.pi - kepler24.getI()  # Inverting inclination
        # new_raan = (kepler24.getRightAscensionOfAscendingNode() + np.pi) # Adjusting RAAN
        # new_arg_perigee = (kepler24.getPerigeeArgument() + np.pi)  # Adjusting argument of perigee

        # # Create the modified orbit
        # keplerMod = KeplerianOrbit(kepler24.getA(), kepler24.getE(), new_inclination, new_arg_perigee, 
        #                         new_raan, kepler24.getTrueAnomaly(), 
        #                         PositionAngleType.MEAN, FramesFactory.getEME2000(), 
        #                         datetime_to_absolutedate(t0_plus_24), Constants.WGS84_EARTH_MU)

        propagator24 = KeplerianPropagator(inverted_kepler24)
        ephemeris_generator24 = propagator24.getEphemerisGenerator()
        propagator24.propagate(datetime_to_absolutedate(t0_plus_24), datetime_to_absolutedate(t0))
        ephemeris24 = ephemeris_generator24.getGeneratedEphemeris()

        kepler_prop_times24, kepler_prop_state_vectors24 = pos_vel_from_orekit_ephem(ephemeris24, datetime_to_absolutedate(t0_plus_24), datetime_to_absolutedate(t0), 60.0)
        print(f"initial state vector in prop24: {kepler_prop_state_vectors24[0]}")
        print(f"state vector in ephemeris at t0_plus_24: {ephemeris_df[ephemeris_df['UTC'] == t0_plus_24][['x', 'y', 'z', 'xv', 'yv', 'zv']]}")
        # print(f"final state vector in prop0: {kepler_prop_state_vectors24[-1]}")

        #use kepler_prop_times0 and t0 to construct the UTC times for the ephemeris_df
        kepler_prop_times24_dt = [t0_plus_24 + datetime.timedelta(seconds=sec) for sec in kepler_prop_times24]
        kepler_times_df = pd.DataFrame({'UTC': pd.to_datetime(kepler_prop_times24_dt)})

        kepler_states_df = pd.DataFrame(kepler_prop_state_vectors24, columns=['x_kep', 'y_kep', 'z_kep', 'xv_kep', 'yv_kep', 'zv_kep'])

        # Merge based on the closest UTC match
        ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
        merged_df = pd.merge_asof(ephemeris_df.sort_values('UTC'), kepler_times_df.sort_values('UTC').join(kepler_states_df), on='UTC')

        last_50 = merged_df.iloc[:1441]

        # Generating a color map based on the index (time progression)
        color_map = np.linspace(0, 1, len(last_50))

        # Calculating the 3D cartesian distance between the two orbits for the last 50 points
        distances = np.sqrt((last_50['x'] - last_50['x_kep'])**2 + (last_50['y'] - last_50['y_kep'])**2 + (last_50['z'] - last_50['z_kep'])**2)

        print(f"min distance: {min(distances)}")
        print(f"max distance: {max(distances)}")
        #calculate the ifference in height at the point of closest approach
        min_dist_index = np.argmin(distances)
        print(f"min_dist_index: {min_dist_index}")

        print(f"distance at first time step: {distances[0]}")


        # Now plot the 3D cartesian distance over time in a new subplot
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        # First subplot for SP3 and Keplerian scatter plot
        ax1 = fig.add_subplot(121, projection='3d')
        sc1= ax1.scatter(last_50['x'], last_50['y'], last_50['z'], c=color_map, cmap='Greys', label='SP3')
        sc2= ax1.scatter(last_50['x_kep'], last_50['y_kep'], last_50['z_kep'], c=color_map, cmap='Purples', label='Keplerian')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_box_aspect([1,1,1])  # Equal aspect ratio
        ax1.legend()
        plt.colorbar(sc1, ax=ax1, label='Time Progression')
        ax1.set_title('SP3 and Keplerian Propagation')

        # Second subplot for distance over time
        ax2 = fig.add_subplot(122)
        time_indices = np.arange(len(distances))
        sc3 = ax2.scatter(time_indices, distances, c=color_map, cmap='viridis')
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Distance')
        ax2.set_title('3D Cartesian Distance Over Time')
        plt.colorbar(sc3, ax=ax2, label='Time Progression')

        plt.tight_layout()
        plt.show()


        # As verification, plot the difference between SP3 ephemeris and the two-body propagation. Should collide at t24 

        # arc_step = int(arc_length / time_step)

        # diffs_3d_abs_results = {}  
        # cd_estimates = {}

        # for arc in range(num_arcs):
        #     hcl_differences = {'H': {}, 'C': {}, 'L': {}}
        #     start_index = arc * arc_step
        #     end_index = start_index + arc_step
        #     arc_df = ephemeris_df.iloc[start_index:end_index]

        #     initial_values = arc_df.iloc[0][['x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]
        #     initial_t = arc_df.iloc[0]['UTC']
        #     final_prop_t = initial_t + datetime.timedelta(seconds=prop_length)
        #     prop_observations_df = ephemeris_df[(ephemeris_df['UTC'] >= initial_t) & (ephemeris_df['UTC'] <= final_prop_t)]
        #     initial_vals = np.array(initial_values.tolist() + [cd, cr, cross_section, mass], dtype=float)

        #     a_priori_estimate = np.concatenate(([initial_t.timestamp()], initial_vals))
        #     observations_df = arc_df[['UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]

        #     for i, force_model_config in enumerate(force_model_configs):
        #         if not force_model_config.get('jb08drag', False) and not force_model_config.get('dtm2000drag', False) and not force_model_config.get('nrlmsise00drag', False):
        #             estimate_drag = False

        #         optimized_states, cov_mats, residuals, RMSs = OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag, max_patience=1)

        #         state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass,boxwing=None, **force_model_config)
        #         phi_ti = propagate_STM(state_ti_minus1, ti, dt, phi_ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass,boxwing=None, **force_model_config)

if __name__ == "__main__":
    main()



            # keplerMod = KeplerianOrbit(keplerOG.getA(), keplerOG.getE(), keplerOG.getI(), keplerOG.getPerigeeArgument(), 
        #                          keplerOG.getRightAscensionOfAscendingNode(), keplerOG.getTrueAnomaly(), 
        #                          PositionAngleType.MEAN, FramesFactory.getEME2000(), datetime_to_absolutedate(t24), 
        #                          Constants.WGS84_EARTH_MU)
