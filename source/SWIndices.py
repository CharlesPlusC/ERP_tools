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
print(f"Reading Dst data from file")
daily_dst_df = read_dst()



print(daily_dst_df.head())

def parse_hourly_values(hourly_str):
    # Replace missing values with NaN and handle concatenated negative numbers
    hourly_str = hourly_str.replace('9999', 'NaN')  # Handling missing data
    hourly_list = hourly_str.replace('-', ' -').split()  # Splitting properly with space before negative sign
    return [float(val) if val != 'NaN' else None for val in hourly_list]

daily_dst_df['HourlyValuesParsed'] = daily_dst_df['HourlyValues'].apply(parse_hourly_values)

# Expand the hourly data into a DataFrame for plotting
hour_cols = [f'Hour_{i}' for i in range(1, 25)]
hourly_values_df = pd.DataFrame(daily_dst_df['HourlyValuesParsed'].tolist(), columns=hour_cols, index=daily_dst_df['Date'])

# Melt the DataFrame for easier plotting
hourly_long_df = hourly_values_df.reset_index().melt(id_vars='Date', value_vars=hour_cols, var_name='Hour', value_name='Value')

# Convert 'Hour' to a numeric type for plotting
hourly_long_df['Hour'] = hourly_long_df['Hour'].str.extract('(\d+)').astype(int)
hourly_long_df['DateTime'] = hourly_long_df.apply(lambda x: x['Date'] + pd.Timedelta(hours=x['Hour']-1), axis=1)


# Plotting a subset of hourly data
plt.figure(figsize=(15, 7))
plt.scatter(hourly_long_df['DateTime'], hourly_long_df['Value'], marker='o', linestyle='-', color='blue', s=1, alpha=0.5, label='Hourly Dst Index')
plt.plot(daily_dst_df['Date'], daily_dst_df['DailyMean'], color='red', linewidth=0.5, label='Daily Mean Dst Index')
plt.title('Hourly and Daily Mean Dst Values Over Time')
plt.xlabel('DateTime')
plt.ylabel('Dst Index Value (nT)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()