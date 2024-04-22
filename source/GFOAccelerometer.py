import pandas as pd
import matplotlib.pyplot as plt
from tools.GFODataReadTools import read_accelerometer, read_quaternions, accelerations_to_inertial
from tools.utilities import gps_time_to_utc

def plot_full_hour_accelerations(df):
    plt.figure(figsize=(10, 6))

    # Make sure utc_time is a datetime object and format it for the plot
    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df['utc_time_str'] = df['utc_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Generate labels conditionally for full hours
    df['is_full_hour'] = df['utc_time'].dt.minute == 0
    full_hours = df[df['is_full_hour']]
    
    # Plotting
    plt.plot(df['utc_time'], df['inertial_x_acc'], label='X Acceleration', linewidth=1)
    plt.plot(df['utc_time'], df['inertial_y_acc'], label='Y Acceleration', linewidth=1)
    plt.plot(df['utc_time'], df['inertial_z_acc'], label='Z Acceleration', linewidth=1)

    # Set custom ticks only at full hours
    plt.xticks(full_hours['utc_time'], full_hours['utc_time_str'], rotation=45)

    plt.xlabel('Time (UTC)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Inertial Frame Accelerations at Full Hours')
    plt.legend()
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.grid(True)
    plt.show()

def example_plot():
    acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-05_C_04.txt"
    quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-05_C_04.txt"

    #TODO: use podaacpy to download the data

    acc_df = read_accelerometer(acc_data_path)
    quat_df = read_quaternions(quat_data_path)

    print(acc_df.head())
    print(quat_df.head())

    #merge the dataframes on the gps_time column
    acc_and_quat_df = pd.merge(acc_df, quat_df, on='gps_time')

    inertial_df = accelerations_to_inertial(acc_and_quat_df)
    print(inertial_df.head())

    inertial_df['utc_time'] = inertial_df['gps_time'].apply(gps_time_to_utc)

    inertial_df['utc_time'] = pd.to_datetime(inertial_df['utc_time'])  # Ensuring datetime type

    # Now you can format utc_time to string for plotting purposes
    inertial_df['utc_time_str'] = inertial_df['utc_time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # Assuming 'inertial_df' is ready and the function is defined correctly, call it
    plot_full_hour_accelerations(inertial_df)

if __name__ == '__main__':
    example_plot()