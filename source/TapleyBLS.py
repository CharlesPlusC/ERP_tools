from tools.utilities import yyyy_mm_dd_hh_mm_ss_to_jd, jd_to_utc
from tools.spaceX_ephem_tools import  parse_spacex_datetime_stamps

import pandas as pd
import numpy as np

def std_dev_from_lower_triangular(lower_triangular_data):
    cov_matrix = np.zeros((6, 6))
    row, col = np.tril_indices(6)
    cov_matrix[row, col] = lower_triangular_data
    cov_matrix = cov_matrix + cov_matrix.T - np.diag(cov_matrix.diagonal())
    std_dev = np.sqrt(np.diag(cov_matrix))
    return std_dev

def spacex_ephem_to_df_w_cov(ephem_path: str) -> pd.DataFrame:
    """
    Convert SpaceX ephemeris data, including covariance terms, into a pandas DataFrame.

    Parameters
    ----------
    ephem_path : str
        Path to the ephemeris file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing parsed SpaceX ephemeris data, including covariance terms.
    """
    with open(ephem_path) as f:
        lines = f.readlines()

    # Remove header lines and select every 4th line starting from the first data line
    t_xyz_uvw = lines[4::4]

    # Extract t, x, y, z, u, v, w
    t = [float(i.split()[0]) for i in t_xyz_uvw]
    x = [float(i.split()[1]) for i in t_xyz_uvw]
    y = [float(i.split()[2]) for i in t_xyz_uvw]
    z = [float(i.split()[3]) for i in t_xyz_uvw]
    u = [float(i.split()[4]) for i in t_xyz_uvw]
    v = [float(i.split()[5]) for i in t_xyz_uvw]
    w = [float(i.split()[6]) for i in t_xyz_uvw]

    # Extract the 21 covariance terms (3 lines after each primary data line)
    covariance_data = {f'cov_{i+1}': [] for i in range(21)}
    for i in range(5, len(lines), 4):  # Start from the first covariance line
        cov_lines = lines[i:i+3]  # Get the three lines of covariance terms
        cov_values = ' '.join(cov_lines).split()
        for j, value in enumerate(cov_values):
            covariance_data[f'cov_{j+1}'].append(float(value))

    # Convert timestamps to strings and call parse_spacex_datetime_stamps
    t_str = [str(int(i)) for i in t]  # Convert timestamps to string
    parsed_timestamps = parse_spacex_datetime_stamps(t_str)

    # Calculate Julian Dates for each timestamp
    jd_stamps = np.zeros(len(parsed_timestamps))
    for i in range(len(parsed_timestamps)):
        jd_stamps[i] = yyyy_mm_dd_hh_mm_ss_to_jd(int(parsed_timestamps[i][0]), int(parsed_timestamps[i][1]), 
                                                 int(parsed_timestamps[i][2]), int(parsed_timestamps[i][3]), 
                                                 int(parsed_timestamps[i][4]), int(parsed_timestamps[i][5]), 
                                                 int(parsed_timestamps[i][6]))

    # Initialize lists for averaged position and velocity standard deviations
    sigma_xs = []
    sigma_ys = []
    sigma_zs = []
    sigma_us = []
    sigma_vs = []
    sigma_ws = []

    # Calculate averaged standard deviations for each row
    for _, row in pd.DataFrame(covariance_data).iterrows():
        std_devs = std_dev_from_lower_triangular(row.values)
        sigma_xs.append(std_devs[0])
        sigma_ys.append(std_devs[1])
        sigma_zs.append(std_devs[2])
        sigma_us.append(std_devs[3])
        sigma_vs.append(std_devs[4])
        sigma_ws.append(std_devs[5])

    # Construct the DataFrame with all data
    spacex_ephem_df = pd.DataFrame({
        't': t,
        'x': x,
        'y': y,
        'z': z,
        'u': u,
        'v': v,
        'w': w,
        'JD': jd_stamps,
        'sigma_xs': sigma_xs,
        'sigma_ys': sigma_ys,
        'sigma_zs': sigma_zs,
        'sigma_us': sigma_us,
        'sigma_vs': sigma_vs,
        'sigma_ws': sigma_ws,
        **covariance_data
    })

    spacex_ephem_df['hours'] = (spacex_ephem_df['JD'] - spacex_ephem_df['JD'][0]) * 24.0 # hours since first timestamp
    # calculate UTC time by applying jd_to_utc() to each JD value
    spacex_ephem_df['UTC'] = spacex_ephem_df['JD'].apply(jd_to_utc)
    # TODO: I am gaining 3 milisecond per minute in the UTC time. Why?

    return spacex_ephem_df


spacex_ephem_df = spacex_ephem_to_df_w_cov('external/ephems/starlink/MEME_57632_STARLINK-30309_3530645_Operational_1387262760_UNCLASSIFIED.txt')

# Initialize state vector from the first point in the SpaceX ephemeris
# TODO: Can perturb this initial state vector to test convergence later
initial_X = spacex_ephem_df['x'][0]*1000
initial_Y = spacex_ephem_df['y'][0]*1000
initial_Z = spacex_ephem_df['z'][0]*1000
initial_VX = spacex_ephem_df['u'][0]*1000
initial_VY = spacex_ephem_df['v'][0]*1000
initial_VZ = spacex_ephem_df['w'][0]*1000
cd = 2.2
initial_t = spacex_ephem_df['UTC'][0]
a_priori_estimate = np.array([initial_t, initial_X, initial_Y, initial_Z, initial_VX, initial_VY, initial_VZ, 2.2])

observations_df = spacex_ephem_df[['UTC', 'x', 'y', 'z', 'u', 'v', 'w', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']]

def BLS_optimize(observations_df, force_model_config, a_priori_estimate=None):
    """
    Batch Least Squares orbit determination algorithm.

    Parameters
    ----------
    state_vector : np.array
        Initial state vector. Must be in the form [t, x, y, z, u, v, w, cd].
    observations_df : pd.DataFrame
        Observations dataframe. Must have columns: ['UTC', 'x', 'y', 'z', 'u', 'v', 'w', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv'].
    force_model_config : dict
        Dictionary containing force model configuration parameters.
    t0 : float
        Initial time of orbit determination.
    tfinal : float
        Final time of orbit determination.
    a_priori_estimate : np.array, optional
        A priori state estimate. The default is None.

    Returns
    -------
    None.

    """
    

    #observations must be in the form of a pandas dataframe with columns:
    #   t, x, y, z, u, v, w, sigma_x, sigma_y, sigma_z, sigma_u, sigma_v, sigma_w
    assert isinstance(observations_df, pd.DataFrame), "observations must be a pandas dataframe"
    assert len(observations_df.columns) == 13, "observations dataframe must have 13 columns"
    required_obs_cols = ['UTC', 'x', 'y', 'z', 'u', 'v', 'w', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']
    assert all([col in observations_df.columns for col in required_obs_cols]), f"observations dataframe must have columns: {required_obs_cols}"
    ### 1) initialize iterations

    it = 1

    ti_minus1 = t0
    state_ti_minus1 = state_t0
    phi_ti_minus1 = phi_t0 

    # initialize lamda and N
    lamda = 0
    N = 0
    if a_priori_estimate is not None:
        #a_priori_estimate is in the form ['UTC', 'x', 'y', 'z', 'u', 'v', 'w', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']
        x_bar_0 = a_priori_estimate[1:7]
        P_0 = np.diag(a_priori_estimate[7:13])
        N = np.linalg.inv(P_0) * x_bar_0
    else:
        lamda = 0
        N = 0

    # read observation number 'it'
    obs_time = observations_df['UTC'][it]
    obs_state = observations_df[['x', 'y', 'z', 'u', 'v', 'w']].iloc[it].values
    obs_covariance = observations_df[['sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']].iloc[it].values


    # assert that state_t0 is in the form [t, x, y, z, u, v, w, cd] and all are floats
    assert isinstance(state_t0, np.ndarray), "state_t0 must be a numpy array"
    assert len(state_t0) == 8, "state_t0 must have 8 elements"
    assert all([isinstance(i, float) for i in state_t0]), "all elements of state_t0 must be floats"



    # make an arrys of the time to iterate over
    ts = observations_df['UTC'].values
    for i in range(1, len(ts)+1):
        ti_minus1 = ts[0]
        state_ti_minus1 = state_t0

        phi_t0 = np.identity(len(state_t0)) #STM
        assert phi_t0.shape == (len(state_t0), len(state_t0)), f"phi_t0 must be a square matrix with dimensions {len(state_t0)} x {len(state_t0)}"
        phi_ti_minus1 = phi_t0 #STM at time ti_minus1

        if a_priori_estimate is not None: #a priori state estimate (e.g. from TLE)
            lamda = np.linalg.inv(P_0)
            N = np.linalg.inv(P_0) * x_0 

            if observations at t0:
            lamda = np.linalg.inv(P_0) + H_i.T * np.linalg.inv(ri) * H_i
            N = H_i.T * np.linalg.inv(ri) * y_i + np.linalg.inv(P_0) * observations

        elif a_priori_estimateis None:
            lamda = 0
            N = 0

    ### 2) read next observation
        ti = obs_time
        yi = obs_state
        ri = obs_covariance

    ### 3) propagate state and STM from ti_minus1 to ti
        state_ti = propagate_state(state_ti_minus1, ti) # along reference trajectory
        phi_ti = propagate_STM(phi_ti_minus1, ti)

    ### 4) compute H-matrix
        # H-tilde is the observation-state mapping matrix
        H_tilde_i = dg/d_state #where g is the observation-state relationship along the reference trajectory
        d_rho/d_state * phi_t1 = one row in the H matrix 
        y_i = yi - rho_i(state_ti) #residual (except for perfect state measurements this is just yi)
        H_i = H_tilde_i * phi_ti
        lamda = lamda + H_i.T * np.linalg.inv(ri) * H_i
        N = N + H_i.T * np.linalg.inv(ri) * y_i

    ### 5) Time check
        if ti < tfinal:
            i = i + 1
            ti_minus1 = ti
            state_ti_minus1 = state_ti
            phi_ti_minus1 = phi_ti
            # go to step 2
        if ti>=tfinal:
            # solve normal equations
            # lamda_xhat = N
            lamda_0 = H_i.T * np.linalg.inv(ri) * H_i + np.linalg.inv(P_0)
            xhat_0 = np.linalg.inv(lamda_0) * N
            P_0 = np.linalg.inv(lamda)

    ### 6) convergence check
        residuals = yi - H_i * xhat_0
        if np.linalg.norm(lamda_xhat - lamda_xhat_old) < 0.01:
            break
        else:
            #update nominal trajectory
            state_t0 = state_t0 + lamda_xhat
            x_bar_0 = x_bar_0 - lamda_xhat

            # use original value of P_0

            # go to step 1

    pass

def propagate_STM():

    df_dy = np.zeros((len(state_t),len(state_t))) #initialize matrix of partial derivatives (partials at time t0)
    # numerical estimation of partial derivatives
    # get the state at t0 and the accelerations at t0
    # perturb each state variable by a small amount and get the new accelerations
    # new accelerations - old accelerations / small amount = partial derivative

    # check -> the first 3*3 should just be 0s
    # check -> the last 3*3 should just be 0s
    # check -> the top right hand 3*3 should be the identity matrix
    # check -> the bottom left hand 3*3 should is where the partial derivatives are changing

    phi_t1 = phi_i + df_dy * phi_i * dt #STM at time t1

    # get d_rho/d_state -> for perfect state measurements this is just ones (include for compelteness)
    # check -> dimensions of d_rho/d_state are 1 x len(state_t)

    # d_rho/d_state * phi_t1 = one row in the H matrix