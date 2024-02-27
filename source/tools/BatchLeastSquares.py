import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from tools.orekit_tools import  propagate_state, propagate_STM, rho_i, make_STM
import numpy as np

def OD_BLS(observations_df, force_model_config, a_priori_estimate, estimate_drag, max_patience=1, boxwing=None):

    t0 = observations_df['UTC'].iloc[0]
    x_bar_0 = np.array(a_priori_estimate[1:7])  # x, y, z, u, v, w
    state_covs = a_priori_estimate[7:13]
    cd = a_priori_estimate[-4]
    cr = a_priori_estimate[-3]
    cross_section = a_priori_estimate[-2]
    mass = a_priori_estimate[-1]
    
    if estimate_drag:
        print(f"Estimating drag coefficient. Initial value: {cd}")
        x_bar_0 = np.append(x_bar_0, cd)

    state_covs = np.diag(state_covs)
    phi_ti_minus1 = np.identity(len(x_bar_0))  # n*n identity matrix for n state variables
    P_0 = np.array(state_covs, dtype=float)  # Covariance matrix from a priori estimate

    if estimate_drag:
        P_0 = np.pad(P_0, ((0, 1), (0, 1)), 'constant', constant_values=0)
        # Assign a non-zero variance to the drag coefficient to avoid non-invertible matrix
        initial_cd_variance = 1  # Setting an arbitrary value for now (but still high)
        P_0[-1, -1] = initial_cd_variance

    d_rho_d_state = np.eye(len(x_bar_0))
    H_matrix = np.empty((0, len(x_bar_0)))
    converged = False
    iteration = 1
    max_iterations = 5
    weighted_rms_last = np.inf 
    convergence_threshold = 0.001 
    no_times_diff_increased = 0

    all_residuals = []
    all_rms = []
    all_xbar_0s = []
    all_covs = []

    while not converged and iteration < max_iterations:
        print(f"Iteration: {iteration}")
        N = np.zeros(len(x_bar_0))
        lamda = np.linalg.inv(P_0)
        y_all = np.empty((0, len(x_bar_0)))
        ti_minus1 = t0
        state_ti_minus1 = x_bar_0 
        RMSs = []
        for _, row in observations_df.iterrows():
            ti = row['UTC']
            observed_state = row[['x', 'y', 'z', 'xv', 'yv', 'zv']].values
            if estimate_drag:
                observed_state = np.append(observed_state, x_bar_0[-1]) #add drag coefficient to the observed state
                obs_covariance = np.diag(row[['sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']])
                cd_covariance = 1  # TODO: is this even allowed? just setting a high value for now
                                     # TODO: do i want to reduce the covariance of Cd after each iteration?
                obs_covariance = np.pad(obs_covariance, ((0, 1), (0, 1)), 'constant', constant_values=0)
                obs_covariance[-1, -1] = cd_covariance
            else:
                obs_covariance = np.diag(row[['sigma_x', 'sigma_y', 'sigma_z', 'sigma_xv', 'sigma_yv', 'sigma_zv']])
            obs_covariance = np.array(obs_covariance, dtype=float)
            W_i = np.linalg.inv(obs_covariance)

            # Propagate state and STM
            dt = ti - ti_minus1
            if estimate_drag:
                state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1[:6], cr=cr, cd=state_ti_minus1[-1], cross_section=cross_section,mass=mass,boxwing=boxwing, **force_model_config)
                #replace with make_STM(state_ti,t0, cr, cd, cross_section,mass,estimate_drag=False, **force_model_config) and propagate_STM(df_dy, phi_i, dt)
                stm = make_STM(state_ti_minus1[:6], ti_minus1, cr=cr, cd=state_ti_minus1[-1], cross_section=cross_section,mass=mass,estimate_drag=True,boxwing=boxwing, **force_model_config)
                phi_ti = propagate_STM(df_dy=stm, phi_i=phi_ti_minus1, dt=dt)
            else:
                state_ti = propagate_state(start_date=ti_minus1, end_date=ti, initial_state_vector=state_ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass,boxwing=boxwing, **force_model_config)
                stm = make_STM(state_ti_minus1, ti_minus1, cr=cr, cd=cd, cross_section=cross_section,mass=mass,estimate_drag=False,boxwing=boxwing, **force_model_config)
                phi_ti = propagate_STM(df_dy=stm, phi_i=phi_ti_minus1, dt=dt)

            # Compute H matrix and residuals for this observation
            H_matrix_row = d_rho_d_state @ phi_ti
            H_matrix = np.vstack([H_matrix, H_matrix_row])
            if estimate_drag:
                state_ti = np.append(state_ti, state_ti_minus1[-1])
            y_i = observed_state - rho_i(state_ti, 'state')
            y_i = np.array(y_i, dtype=float)
            y_all = np.vstack([y_all, y_i])

            # Update lambda and N matrices
            lamda += H_matrix_row.T @ W_i @ H_matrix_row
            N += H_matrix_row.T @ W_i @ y_i

            # Update for next iteration
            ti_minus1 = ti
            state_ti_minus1 = np.array(state_ti)
            phi_ti_minus1 = phi_ti
            RMSs.append(y_i.T @ W_i @ y_i)

        RMSs = np.array(RMSs)
        weighted_rms = np.sqrt(np.sum(RMSs) / (len(x_bar_0) * len (y_all)))
        print(f"RMS: {weighted_rms}")
        # Solve normal equations
        xhat = np.linalg.inv(lamda) @ N
        # Check for convergence
        if abs(weighted_rms - weighted_rms_last) < convergence_threshold:
            print("Converged!")
            converged = True
        else:
            if weighted_rms > weighted_rms_last:
                no_times_diff_increased += 1
                print(f"RMS increased {no_times_diff_increased} times in a row.")
                if no_times_diff_increased >= max_patience:
                    print("Stopping iteration.")
                    break
            else:
                no_times_diff_increased = 0
            weighted_rms_last = weighted_rms
            x_bar_0 += xhat  # Update nominal trajectory
            print("updated x_bar_0: ", x_bar_0)

        all_residuals.append(y_all)
        all_rms.append(weighted_rms)
        all_xbar_0s.append(x_bar_0)
        all_covs.append(np.linalg.inv(lamda))
        iteration += 1

    return all_xbar_0s, all_covs, all_residuals, all_rms