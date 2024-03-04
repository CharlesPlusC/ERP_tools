import orekit
from orekit.pyhelpers import setup_orekit_curdir

vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

import datetime
import pandas as pd
from org.orekit.utils import IERSConventions, PVCoordinates
from org.hipparchus.linear import MatrixUtils
from org.orekit.propagation import StateCovariance
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.ssa.collision.shorttermencounter.probability.twod import Patera2005
from org.orekit.orbits import CartesianOrbit, KeplerianOrbit
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.utils import Constants
from org.orekit.orbits import PositionAngleType, OrbitType

from tools.utilities import pos_vel_from_orekit_ephem


def generate_collision_trajectory(ephemeris_df, t_col):
    """Generate a trajectory of a satellite that will collide with another satellite at a given time.

    Parameters
    ----------
    ephemeris_df : pandas.DataFrame
        A DataFrame containing the ephemeris data for the satellite to be collided with. 
        Must contain columns 'UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv'.
    t_col : datetime.datetime
        The time of the collision.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the ephemeris data for the satellite to be collided with, 
        including the trajectory of the satellite up to 45 minutes after the collision.
    """
    #start of sim is the UTC value of the first row of the ephemeris
    start_of_sim = ephemeris_df['UTC'].iloc[0]
    end_of_sim = t_col + datetime.timedelta(minutes=45) #continue the ephemeris for 45 minutes after the collision

    #slice ephemeris_df so that it stops at t0_plus_24_plus_t
    ephemeris_df = ephemeris_df[ephemeris_df['UTC'] <= end_of_sim]
    
    #coordinates of the satellite when the collision will occur
    pvcoords_col = PVCoordinates(
        Vector3D(float(ephemeris_df[ephemeris_df['UTC'] == t_col]['x'].iloc[0]),
                float(ephemeris_df[ephemeris_df['UTC'] == t_col]['y'].iloc[0]),
                float(ephemeris_df[ephemeris_df['UTC'] == t_col]['z'].iloc[0])),
        Vector3D(float(ephemeris_df[ephemeris_df['UTC'] == t_col]['xv'].iloc[0]),
                float(ephemeris_df[ephemeris_df['UTC'] == t_col]['yv'].iloc[0]),
                float(ephemeris_df[ephemeris_df['UTC'] == t_col]['zv'].iloc[0]))
    )

    #invert the velocities
    inverted_velocities = Vector3D(-float(ephemeris_df[ephemeris_df['UTC'] == t_col]['xv'].iloc[0]),
                                    -float(ephemeris_df[ephemeris_df['UTC'] == t_col]['yv'].iloc[0]),
                                    -float(ephemeris_df[ephemeris_df['UTC'] == t_col]['zv'].iloc[0]))

    #use the inverted velocities to generate the collision trajectory
    inverted_pvcoords_tcol = PVCoordinates(pvcoords_col.getPosition(), inverted_velocities)
    inverted_kepler_col = KeplerianOrbit(inverted_pvcoords_tcol,
                            FramesFactory.getEME2000(),
                            datetime_to_absolutedate(t_col),
                            Constants.WGS84_EARTH_MU)
                                        
    propagator_col = KeplerianPropagator(inverted_kepler_col)
    ephemeris_generator_col = propagator_col.getEphemerisGenerator()
    #propagate backwards from time of collision to t0 to get state vectors over the entire time window of interest
    propagator_col.propagate(datetime_to_absolutedate(end_of_sim), datetime_to_absolutedate(start_of_sim))
    ephemeris_col = ephemeris_generator_col.getGeneratedEphemeris()
    prop_times_col, prop_state_vectors_col = pos_vel_from_orekit_ephem(ephemeris_col, datetime_to_absolutedate(t_col), datetime_to_absolutedate(start_of_sim), 60.0)
    prop_times_col_dt = [t_col + datetime.timedelta(seconds=sec) for sec in prop_times_col]
    col_times_df = pd.DataFrame({'UTC': pd.to_datetime(prop_times_col_dt)})
    col_states_df = pd.DataFrame(prop_state_vectors_col, columns=['x_col', 'y_col', 'z_col', 'xv_col', 'yv_col', 'zv_col'])
    ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    collision_df = pd.merge_asof(ephemeris_df.sort_values('UTC'), col_times_df.sort_values('UTC').join(col_states_df), on='UTC')
    return collision_df

def patera05_poc(merged_df, object1_cov, object2_cov, tca, hbr_1, hbr_2):
    """Calculate the probability of collision between two objects using the Patera2005 algorithm.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        A DataFrame containing the ephemeris data for the two objects.
    object1_cov : np.ndarray
        The covariance matrix for the first object.
    object2_cov : np.ndarray
        The covariance matrix for the second object.
    tca : datetime.datetime
        The time of closest approach between the two objects.
    hbr_1 : float
        The hard body radius of the first object
    hbr_2 : float
        The hard body radius of the second object

    Returns
    -------
    float
        The probability of collision between the two objects at the time of closest approach.
    """
    try:
        # Convert Python arrays to Java RealMatrix objects for Orekit
        jarray_cov1 = MatrixUtils.createRealMatrix(len(object1_cov), len(object1_cov[0]))
        jarray_cov2 = MatrixUtils.createRealMatrix(len(object1_cov), len(object1_cov[0]))
        for i in range(len(object1_cov)):
            for j in range(len(object1_cov[i])):
                jarray_cov1.setEntry(i, j, float(object1_cov[i][j]))
                jarray_cov2.setEntry(i, j, float(object2_cov[i][j]))
        
        # Create StateCovariance objects for the two covariances
        cov_1 = StateCovariance(jarray_cov1, datetime_to_absolutedate(tca), 
                                  FramesFactory.getITRF(IERSConventions.IERS_2010, False), OrbitType.CARTESIAN, PositionAngleType.TRUE)
        cov_2 = StateCovariance(jarray_cov2, datetime_to_absolutedate(tca), 
                                  FramesFactory.getITRF(IERSConventions.IERS_2010, False), OrbitType.CARTESIAN, PositionAngleType.TRUE)

        # Extract state vectors at the time of closest approach from merged_df
        closest_approach_index = merged_df.index[merged_df['UTC'] == tca].tolist()[0]
        opt_state_TCA = merged_df.iloc[closest_approach_index][['x_opt', 'y_opt', 'z_opt', 'xv_opt', 'yv_opt', 'zv_opt']].values.astype(float)
        kep_state_TCA = merged_df.iloc[closest_approach_index][['x_col', 'y_col', 'z_col', 'xv_col', 'yv_col', 'zv_col']].values.astype(float)

        # Convert state vectors to PVCoordinates and then to CartesianOrbit objects
        # pv1_TCA = PVCoordinates(Vector3D(*opt_state_TCA[:3]), Vector3D(*opt_state_TCA[3:]))
        pv1_TCA = PVCoordinates(Vector3D(float(opt_state_TCA[0]), float(opt_state_TCA[1]), float(opt_state_TCA[2])), Vector3D(float(opt_state_TCA[3]), float(opt_state_TCA[4]), float(opt_state_TCA[5])))
        pv2_TCA = PVCoordinates(Vector3D(float(kep_state_TCA[0]), float(kep_state_TCA[1]), float(kep_state_TCA[2])), Vector3D(float(kep_state_TCA[3]), float(kep_state_TCA[4]), float(kep_state_TCA[5])))
        
        opt_TCA_orbit = CartesianOrbit(pv1_TCA, FramesFactory.getEME2000(), datetime_to_absolutedate(tca), Constants.WGS84_EARTH_MU)
        kep_tca_orbit = CartesianOrbit(pv2_TCA, FramesFactory.getEME2000(), datetime_to_absolutedate(tca), Constants.WGS84_EARTH_MU)

        # Compute the probability of collision
        combined_hbr = hbr_1 + hbr_2
        patera2005 = Patera2005() 
        poc_result = patera2005.compute(opt_TCA_orbit, cov_1, kep_tca_orbit, cov_2, combined_hbr, 1e-10)
        return poc_result.getValue()
    except Exception as e:
        print(f"Failed to calculate probability of collision: {str(e)}")
        return None