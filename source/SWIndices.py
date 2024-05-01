# Kp, Ap, SN, F10.7 reader
# Dst Reader
# Merge
# Identify G indices time series and plot
# Fetch SP3 data at the transition into and out of each of the G indices
# Perform POD accelerometery density inversion
import pandas as pd
import re
import matplotlib.pyplot as plt

def read_dst(filepath = "external/SWIndices/Dst_2000_2024.txt"):
    """
        From: https://wdc.kugi.kyoto-u.ac.jp/dstae/format/dstformat.html
        COLUMN	FORMAT	SHORT DESCRIPTION
        1-3	A3	Index name 'DST'
        4-5	I2	The last two digits of the year
        6-7	I2	Month
        8	A1	'*' for index
        9-10	I2	Date
        11-12	A2	All spaces or may be "RR" for quick look
        13	A1	'X' (for index)
        14	A1	Version (0: quicklook, 1: provisional, 2: final, 3 and up: corrected final or may be space)
        15-16	I2	Top two digits of the year (19 or space for 19XX, 20 from 2000)
        17-20	I4	Base value, unit 100 nT
        21-116	24I4	24 hourly values, 4 digit number, unit 1 nT, value 9999 for the missing data.
        First data is for the first hour of the day, and Last data is for the last hour of the day.
        117-120	I4	Daily mean value, unit 1 nT. Value 9999 for the missing data.
        """

    import pandas as pd

    col_specs = [
        (0, 3), (3, 5), (5, 7), (7, 8), (8, 10), (10, 12), (12, 13), (13, 14), (14, 20), (20, 116), (116, 120)
    ]
    names = [
        'Index', 'Year', 'Month', 'Mark', 'Day', 'Flag', 'X', 'Version', 'BaseValue', 'HourlyValues', 'DailyMean'
    ]
    
    # Read data with adjusted settings
    dst_df = pd.read_fwf(filepath, colspecs=col_specs, names=names, index_col=False)

    # Adjust the 'Year' column by adding 2000 to handle years properly
    dst_df['Year'] = dst_df['Year'].astype(int) + 2000

    # Create the 'Date' column from 'Year', 'Month', and 'Day'
    dst_df['Date'] = pd.to_datetime(dst_df[['Year', 'Month', 'Day']])

    return dst_df

def parse_hourly_values(hourly_str):
    # Replace missing values with NaN and handle concatenated negative numbers
    hourly_str = hourly_str.replace('9999', 'NaN')  # Handling missing data
    hourly_list = hourly_str.replace('-', ' -').split()  # Splitting properly with space before negative sign
    return [float(val) if val != 'NaN' else None for val in hourly_list]

def process_kp_ap_f107_sn(filepath='external/SWIndices/Kp_ap_Ap_SN_F107_since_1932.txt'):
    # Read the data, skipping the header lines
    kp_data = pd.read_csv(filepath, delim_whitespace=True, skiprows=40, header=None,
                          names=[
                              "Year", "Month", "Day", "Days", "Days_m", "BSR", "dB",
                              "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8",
                              "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8",
                              "Ap", "SN", "F10.7obs", "F10.7adj", "D"
                          ])

    # Convert date columns to datetime format
    kp_data['Date'] = pd.to_datetime(kp_data[['Year', 'Month', 'Day']])

    # Calculating the daily average Kp value
    kp_columns = ['Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8']
    kp_data['Kp_avg'] = kp_data[kp_columns].mean(axis=1)

    kp_details = pd.DataFrame()

    for i in range(1, 9):
        temp_df = kp_data[['Date', f'Kp{i}', 'Ap', 'SN', 'F10.7obs']].copy()
        temp_df.rename(columns={f'Kp{i}': 'Kp'}, inplace=True)
        temp_df['DateTime'] = temp_df['Date'] + pd.to_timedelta((i-1)*3, unit='h')
        kp_details = pd.concat([kp_details, temp_df], ignore_index=True)

    return kp_data, kp_details

def get_sw_indices():

    daily_dst_df = read_dst()

    daily_dst_df['HourlyValuesParsed'] = daily_dst_df['HourlyValues'].apply(parse_hourly_values)

    # Expand the hourly data into a DataFrame for plotting
    hour_cols = [f'Hour_{i}' for i in range(1, 25)]
    hourly_dst = pd.DataFrame(daily_dst_df['HourlyValuesParsed'].tolist(), columns=hour_cols, index=daily_dst_df['Date'])

    # Melt the DataFrame for easier plotting
    hourly_dst = hourly_dst.reset_index().melt(id_vars='Date', value_vars=hour_cols, var_name='Hour', value_name='Value')

    # Convert 'Hour' to a numeric type for plotting
    hourly_dst['Hour'] = hourly_dst['Hour'].str.extract('(\d+)').astype(int)
    hourly_dst['DateTime'] = hourly_dst.apply(lambda x: x['Date'] + pd.Timedelta(hours=x['Hour']-1), axis=1)

    kp_data, kp_3hrly = process_kp_ap_f107_sn()

    # Merging the Dst data with the other indices data based on the 'Date' column
    daily_indices = pd.merge(kp_data, daily_dst_df[['Date', 'DailyMean']], on='Date', how='left')

    return daily_indices, kp_3hrly, hourly_dst

def plot_all_indices_in_one(merged_data, kp_details, hourly_long_df, daily_dst = False, daily_kp = False):

    # Plotting all indices over time with separate y-axes
    fig, ax1 = plt.subplots(figsize=(15, 9))

    if daily_dst:
        # Daily Dst
        color = 'black'
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Dst', color=color)
        ax1.plot(merged_data['Date'], merged_data['DailyMean'], label='Daily Dst', color=color, alpha=0.5)
        ax1.tick_params(axis='y', labelcolor=color)
    else:
        # Hourly Dst
        color = 'blue'
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Dst', color=color)
        ax1.scatter(hourly_long_df['DateTime'], hourly_long_df['Value'], label='Hourly Dst', color=color, alpha=0.5, s=1)
        ax1.tick_params(axis='y', labelcolor=color)

    # # Daily Kp
    if daily_kp:
        color = 'red'
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Kp', color=color)
        ax1.plot(merged_data['Date'], merged_data['Kp_avg'], label='Daily Kp', color=color, alpha=0.5)
        ax1.tick_params(axis='y', labelcolor=color)
    else:
        # Plotting 3-hour Kp
        ax2 = ax1.twinx()
        color = 'red'
        ax2.set_ylabel('Kp', color=color)
        ax2.scatter(kp_details['DateTime'], kp_details['Kp'], label='3-hour Kp', color=color, alpha=0.5, s=1)
        ax2.tick_params(axis='y', labelcolor=color)

    # Plotting Ap
    ax3 = ax1.twinx()
    color = 'green'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Ap', color=color)
    ax3.plot(merged_data['Date'], merged_data['Ap'], label='Daily Ap', color=color, alpha=0.5)
    ax3.tick_params(axis='y', labelcolor=color)

    # Plotting SN
    ax4 = ax1.twinx()
    color = 'purple'
    ax4.spines['right'].set_position(('outward', 120))
    ax4.set_ylabel('SN', color=color)
    ax4.plot(merged_data['Date'], merged_data['SN'], label='Daily SN', color=color, alpha=0.5)
    ax4.tick_params(axis='y', labelcolor=color)

    # Plotting F10.7obs
    ax5 = ax1.twinx()
    color = 'orange'
    ax5.spines['right'].set_position(('outward', 180))
    ax5.set_ylabel('F10.7obs', color=color)
    ax5.plot(merged_data['Date'], merged_data['F10.7obs'], label='Daily F10.7obs', color=color, alpha=0.5)
    ax5.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Geomagnetic and Solar Indices Over Time Including Dst')
    fig.legend(loc='upper left')
    plt.show()

def plot_all_indices_separate(merged_data, kp_details, hourly_long_df, daily_dst = False, daily_kp = False):
    # Plotting all indices over time with separate y-axes
    fig, ax1 = plt.subplots(figsize=(15, 9))

    if daily_dst:
        # Daily Dst
        color = 'black'
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Dst', color=color)
        ax1.plot(merged_data['Date'], merged_data['DailyMean'], label='Daily Dst', color=color, alpha=0.5)
        ax1.tick_params(axis='y', labelcolor=color)
    else:
        # Hourly Dst
        color = 'blue'
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Dst', color=color)
        ax1.scatter(hourly_long_df['DateTime'], hourly_long_df['Value'], label='Hourly Dst', color=color, alpha=0.5, s=1)
        ax1.tick_params(axis='y', labelcolor=color)

    if daily_kp:
        color = 'red'
        #Daily Kp values
        ax2 = ax1.twinx()
        ax2.set_xlabel('DateTime')
        ax2.set_ylabel('Kp', color=color)
        ax2.scatter(merged_data['Date'], merged_data['Kp_avg'], label='Daily Kp', color=color, alpha=0.5, s=1)
        ax2.tick_params(axis='y', labelcolor=color)
    else:
        # #3 hour Kp values
        ax2 = ax1.twinx()
        color = 'red'
        ax2.set_xlabel('DateTime')
        ax2.set_ylabel('Kp', color=color)
        ax2.scatter(kp_details['DateTime'], kp_details['Kp'], label='3-hour Kp', color=color, alpha=0.5, s=1)
        ax2.tick_params(axis='y', labelcolor=color)

    color = 'green'
    ax2.set_ylabel('Ap', color=color)
    ax2.plot(merged_data['Date'], merged_data['Ap'], label='Daily Ap', color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    color = 'purple'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('SN', color=color)
    ax3.plot(merged_data['Date'], merged_data['SN'], label='Daily SN', color=color, alpha=0.5)
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax1.twinx()
    color = 'orange'
    ax4.spines['right'].set_position(('outward', 120))
    ax4.set_ylabel('F10.7obs', color=color)
    ax4.plot(merged_data['Date'], merged_data['F10.7obs'], label='Daily F10.7obs', color=color, alpha=0.5)
    ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Geomagnetic and Solar Indices Over Time')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.show()