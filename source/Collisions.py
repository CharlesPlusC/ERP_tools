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
        #slice the ephemeris to start 6 arcs past the beginning
        time_step = (ephemeris_df['UTC'].iloc[1] - ephemeris_df['UTC'].iloc[0]).total_seconds() / 60.0  # in minutes
        time_step_seconds = time_step * 60.0

        #now find the value of the first time step
        t0 = ephemeris_df['UTC'].iloc[0]
        print(f"t1: {t0}")
        print(f"ephemeris at t1: {ephemeris_df.iloc[0]}")
        #now find 24 hours later
        t24 = t0 + datetime.timedelta(days=1)
        print(f"ti: {t24}")
        #now find the index of the ephemeris at t24
        t24_index = ephemeris_df[ephemeris_df['UTC'] == t24].index[0]
        print(f"ephemeris at ti: {ephemeris_df.iloc[t24_index]}")
        
        # make a KeplerianOrbit object from the ephemeris at t24
            # get 'x', 'y', 'z', 'xv', 'yv', 'zv' from ephemeris_df.iloc[t24_index]

        x, y, z, xv, yv, zv = ephemeris_df.iloc[t24_index][['x', 'y', 'z', 'xv', 'yv', 'zv']]
        print(f"x: {x}, y: {y}, z: {z}, xv: {xv}, yv: {yv}, zv: {zv}")
        pv_coords_t24 = PVCoordinates(Vector3D(float(x), float(y), float(z)),
                                    Vector3D(float(xv), float(yv), float(zv)))

        keplerOG = KeplerianOrbit(pv_coords_t24, FramesFactory.getEME2000(), datetime_to_absolutedate(t24), Constants.WGS84_EARTH_MU)
        # Precess the raan by 180 degrees
        keplerMod = KeplerianOrbit(keplerOG.getA(), keplerOG.getE(), keplerOG.getI(), keplerOG.getPerigeeArgument(), 
                                 keplerOG.getRightAscensionOfAscendingNode() + float(np.pi/2), keplerOG.getAnomaly(PositionAngleType.TRUE), 
                                 PositionAngleType.TRUE, FramesFactory.getEME2000(), datetime_to_absolutedate(t24), 
                                 Constants.WGS84_EARTH_MU)
        # Propagate this new orbit backwards to t1 (Keplerian propagation)
        propagator = KeplerianPropagator(keplerMod) 
        ephemeris_generator = propagator.getEphemerisGenerator()
        kepler_state_at_t1 = propagator.propagate(datetime_to_absolutedate(t24), datetime_to_absolutedate(t0)) 

        ephemeris = ephemeris_generator.getGeneratedEphemeris()
        kepler_prop_times, kepler_prop_state_vectors = pos_vel_from_orekit_ephem(ephemeris, datetime_to_absolutedate(t0), 
                                                        datetime_to_absolutedate(t24), 60.0)
        
        #now put the kepler_prop_state_vectors into the dataframe with the SP3 ephemeris but as [x_kep, y_kep, z_kep, xv_kep, yv_kep, zv_kep]
        #kepler_prop_state_vectors is of the shape array([-5.09081592e+06, -2.86173423e+06, -3.61381978e+06,  3.54737132e+03,
        # 1.80505709e+03, -6.49082528e+03]), array([-4.86686204e+06, -2.74717620e+06, -3.99498267e+06,  3.91494921e+03,
        # 2.01210444e+03, -6.20995679e+03]), array([-4.62137472e+06, -2.62046329e+06, -4.35847011e+06,  4.26489444e+03,
        # 2.21007370e+03, -5.90186773e+03]),
        #and then plot the difference between the SP3 ephemeris and the two-body propagation.
        print(f"kepler prop state vecs:", kepler_prop_state_vectors)
        print(f"kepler times:", kepler_prop_times)
        print(f"epehem df at t24:", ephemeris_df.iloc[t24_index])

        kepler_prop_state_vectors = np.array(kepler_prop_state_vectors)
        ephemeris_df = ephemeris_df.iloc[:len(kepler_prop_state_vectors)]

        ephemeris_df['x_kep'] = kepler_prop_state_vectors[:, 0]
        ephemeris_df['y_kep'] = kepler_prop_state_vectors[:, 1]
        ephemeris_df['z_kep'] = kepler_prop_state_vectors[:, 2]
        ephemeris_df['xv_kep'] = kepler_prop_state_vectors[:, 3]
        ephemeris_df['yv_kep'] = kepler_prop_state_vectors[:, 4]
        ephemeris_df['zv_kep'] = kepler_prop_state_vectors[:, 5]

        ephemeris_df_truncated = ephemeris_df.iloc[:len(kepler_prop_state_vectors)]

        # Assign Keplerian propagated state vectors to the truncated DataFrame
        ephemeris_df_truncated['x_kep'] = kepler_prop_state_vectors[:, 0]
        ephemeris_df_truncated['y_kep'] = kepler_prop_state_vectors[:, 1]
        ephemeris_df_truncated['z_kep'] = kepler_prop_state_vectors[:, 2]
        ephemeris_df_truncated['xv_kep'] = kepler_prop_state_vectors[:, 3]
        ephemeris_df_truncated['yv_kep'] = kepler_prop_state_vectors[:, 4]
        ephemeris_df_truncated['zv_kep'] = kepler_prop_state_vectors[:, 5]

        # Now plot both sets of x, y, z in 3D for the truncated DataFrame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ephemeris_df_truncated['x'], ephemeris_df_truncated['y'], ephemeris_df_truncated['z'], label='SP3')
        ax.scatter(ephemeris_df_truncated['x_kep'], ephemeris_df_truncated['y_kep'], ephemeris_df_truncated['z_kep'], label='Keplerian')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #make sure the aspect ratio is equal
        ax.set_box_aspect([1,1,1])
        ax.legend()
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