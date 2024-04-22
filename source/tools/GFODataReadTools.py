import pandas as pd
import numpy as np
from tools.utilities import gps_time_to_utc

def read_accelerometer(file_path):
    column_names = [
        'gps_time', 'GRACEFO_id', 'lin_accl_x', 'lin_accl_y', 'lin_accl_z',
        'ang_accl_x', 'ang_accl_y', 'ang_accl_z',
        'acl_x_res', 'acl_y_res', 'acl_z_res', 'qualflg'
    ]
    
    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file):
                if line.strip() == '# End of YAML header':
                    header_end = line_number + 1
                    break
            else:
                raise ValueError("End of YAML header not found.")
        
        df = pd.read_csv(
            file_path,
            delim_whitespace=True,  
            names=column_names,  
            skiprows=header_end  
        )
    except Exception as e:
        print("An error occurred:", e)
        return None
    
    return df

def read_quaternions(file_path):
    column_names = [
        'gps_time', 'GRACEFO_id', 'sca_id', 'quatangle', 
        'quaticoeff', 'quatjcoeff', 'quatkcoeff', 
        'qual_rss', 'qualflg'
    ]
    
    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file):
                if line.strip() == '# End of YAML header':
                    header_end = line_number + 1
                    break
            else:
                raise ValueError("End of YAML header not found.")
        
        df = pd.read_csv(
            file_path,
            delim_whitespace=True, 
            names=column_names, 
            skiprows=header_end  
        )
    except Exception as e:
        print("An error occurred:", e)
        return None
    
    return df

def rotate_vector(quaternion, vector):
    # Quaternion and vector should be numpy arrays
    q = quaternion
    v = np.array([0, *vector])  # Extend vector to quaternion form with 0 as the scalar component

    # Quaternion multiplication (q * v * q^-1)
    q_conj = q * np.array([1, -1, -1, -1])  # Conjugate of quaternion
    v_rot = quaternion_multiply(quaternion_multiply(q, v), q_conj)

    return v_rot[1:]  # Return the vector part of the resulting quaternion

def quaternion_multiply(q, r):
    # Multiplication of two quaternions q and r
    return np.array([
        q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],  # Scalar component
        q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],  # i-component
        q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],  # j-component
        q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]   # k-component
    ])

def accelerations_to_inertial(acc_df):
    # Assumes columns 'quaticoeff', 'quatjcoeff', 'quatkcoeff', 'quatangle', 'lin_accl_x', 'lin_accl_y', 'lin_accl_z', 'gps_time' exist
    # Quaternion components: [quatangle, quaticoeff, quatjcoeff, quatkcoeff]
    # Acceleration components: [lin_accl_x, lin_accl_y, lin_accl_z]

    # Prepare the DataFrame to store the inertial frame accelerations and gps_time
    transformed = pd.DataFrame(index=acc_df.index, columns=['gps_time', 'inertial_x_acc', 'inertial_y_acc', 'inertial_z_acc'])

    for idx, row in acc_df.iterrows():
        quaternion = np.array([row['quatangle'], row['quaticoeff'], row['quatjcoeff'], row['quatkcoeff']])
        acceleration = np.array([row['lin_accl_x'], row['lin_accl_y'], row['lin_accl_z']])

        # Rotate acceleration vector
        inertial_acc = rotate_vector(quaternion, acceleration)

        # Store transformed accelerations and the gps_time
        transformed.loc[idx, 'gps_time'] = row['gps_time']
        transformed.loc[idx, 'inertial_x_acc'] = inertial_acc[0]
        transformed.loc[idx, 'inertial_y_acc'] = inertial_acc[1]
        transformed.loc[idx, 'inertial_z_acc'] = inertial_acc[2]

    return transformed

def get_gfo_inertial_accelerations(acc_data_path, quat_data_path):
    acc_df = read_accelerometer(acc_data_path)
    quat_df = read_quaternions(quat_data_path)
    acc_and_quat_df = pd.merge(acc_df, quat_df, on='gps_time')
    inertial_df = accelerations_to_inertial(acc_and_quat_df)
    inertial_df['utc_time'] = inertial_df['gps_time'].apply(gps_time_to_utc)
    return inertial_df
