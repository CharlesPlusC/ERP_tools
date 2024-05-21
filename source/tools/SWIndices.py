import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import gzip
import netCDF4 as nc

def read_ae(start_date, end_date, components=['AE']):
    """
    Read AE index files within a date range and expand each row to have one row per minute.
    
    Parameters:
    - start_date: str, start date in the format 'YYYY-MM-DD'
    - end_date: str, end date in the format 'YYYY-MM-DD'
    - components: list of str, components to read (default is ['AE'])
    
    Returns:
    - DataFrame with AE index data within the specified date range or None if no data is found
    """
    colspecs = [(0, 12), (12, 14), (14, 16), (16, 18), (18, 19), (19, 21), (21, 24), (24, 34)] + [(34 + i*6, 34 + (i+1)*6) for i in range(60)] + [(394, 400)]
    names = ['identifier', 'year', 'month', 'day', 'component', 'hour', 'index_name', 'edition'] + [f'minute_{i}' for i in range(1, 61)] + ['hourly_mean']
    
    def parse_minute_values(row):
        return [int(row[f'minute_{i}']) for i in range(1, 61)]

    def read_ae_file(filepath):
        ae_df = pd.read_fwf(filepath, colspecs=colspecs, names=names)
        ae_df['date'] = '20' + ae_df['year'].astype(str).str.zfill(2) + ae_df['month'].astype(str).str.zfill(2) + ae_df['day'].astype(str).str.zfill(2)
        ae_df['Datetime'] = pd.to_datetime(ae_df['date'], format='%Y%m%d') + pd.to_timedelta(ae_df['hour'].astype(int), unit='h')
        ae_df['minute_values'] = ae_df.apply(parse_minute_values, axis=1)
        return ae_df

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    all_data = []
    for year in range(start_year, end_year + 1):
        filepath = f"external/SWIndices/AE_{year}.txt"
        if os.path.exists(filepath):
            yearly_data = read_ae_file(filepath)
            all_data.append(yearly_data)
    
    if not all_data:
        return None

    ae_df = pd.concat(all_data)
    ae_df = ae_df[ae_df['index_name'].str.strip().isin(components)]
    ae_df = ae_df[(ae_df['Datetime'] >= start_date) & (ae_df['Datetime'] <= end_date)]

    if ae_df.empty:
        return None

    expanded_rows = []
    for _, row in ae_df.iterrows():
        base_time = row['Datetime']
        for i, minute_value in enumerate(row['minute_values']):
            expanded_rows.append({'Datetime': base_time + pd.Timedelta(minutes=i), 'minute_value': minute_value, 'component': row['index_name'].strip()})

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df = expanded_df[(expanded_df['Datetime'] >= start_date) & (expanded_df['Datetime'] <= end_date)]
    
    if expanded_df.empty:
        return None
    
    return expanded_df

def read_sym(start_date, end_date, components=['H']):
    """
    Read SYM index files within a date range and expand each row to have one row per minute.

    Parameters:
    - start_date: str, start date in the format 'YYYY-MM-DD'
    - end_date: str, end date in the format 'YYYY-MM-DD'
    - components: list of str, list of components to include ('D' for D-comp. and/or 'H' for H-comp.), default is ['H']

    Returns:
    - DataFrame with SYM index data within the specified date range or None if no data is found
    """
    colspecs = [(0, 12), (12, 14), (14, 16), (16, 18), (18, 19), (19, 21), (21, 24), (24, 34)] + [(34 + i*6, 34 + (i+1)*6) for i in range(60)] + [(394, 400)]
    names = ['identifier', 'year', 'month', 'day', 'component', 'hour', 'index_name', 'edition'] + [f'minute_{i}' for i in range(1, 61)] + ['hourly_mean']

    def parse_minute_values(row):
        return [int(row[f'minute_{i}']) for i in range(1, 61)]

    def read_sym_file(filepath):
        sym_df = pd.read_fwf(filepath, colspecs=colspecs, names=names)
        sym_df['date'] = '20' + sym_df['year'].astype(str).str.zfill(2) + sym_df['month'].astype(str).str.zfill(2) + sym_df['day'].astype(str).str.zfill(2)
        sym_df['Datetime'] = pd.to_datetime(sym_df['date'], format='%Y%m%d') + pd.to_timedelta(sym_df['hour'].astype(int), unit='h')
        sym_df['minute_values'] = sym_df.apply(parse_minute_values, axis=1)
        return sym_df[['Datetime', 'minute_values', 'hourly_mean', 'component']]

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    all_data = []
    for year in range(start_year, end_year + 1):
        filepath = f"external/SWIndices/SYM_ASY_{year}.txt"
        if os.path.exists(filepath):
            yearly_data = read_sym_file(filepath)
            all_data.append(yearly_data)

    if not all_data:
        return None

    sym_df = pd.concat(all_data)
    sym_df = sym_df[(sym_df['Datetime'] >= start_date) & (sym_df['Datetime'] <= end_date)]
    sym_df = sym_df[sym_df['component'].isin(components)]

    if sym_df.empty:
        return None

    expanded_rows = []
    for _, row in sym_df.iterrows():
        base_time = row['Datetime']
        for i, minute_value in enumerate(row['minute_values']):
            expanded_rows.append({'Datetime': base_time + pd.Timedelta(minutes=i), 'minute_value': minute_value, 'component': row['component']})

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df = expanded_df[(expanded_df['Datetime'] >= start_date) & (expanded_df['Datetime'] <= end_date)]
    
    if expanded_df.empty:
        return None
    
    return expanded_df

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

def classify_storm(kp_val):
    NOAA_storm_classification = {
        5: 'G1',
        6: 'G2',
        7: 'G3',
        8: 'G4',
        9: 'G5'
    }
    kp_int = round(float(kp_val))
    if kp_int <= 5:
        return 'G1'
    else:
        return NOAA_storm_classification.get(kp_int, 'G1')

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

    kp_details = pd.DataFrame()

    for i in range(1, 9):
        # Extract each Kp value and its datetime
        temp_df = kp_data[['Date', f'Kp{i}', 'Ap', 'SN', 'F10.7obs']].copy()
        temp_df.rename(columns={f'Kp{i}': 'Kp'}, inplace=True)
        temp_df['DateTime'] = temp_df['Date'] + pd.to_timedelta((i-1)*3, unit='h')
        
        # Apply classification to each individual Kp
        temp_df['storm_scale'] = temp_df['Kp'].apply(classify_storm)
        
        kp_details = pd.concat([kp_details, temp_df], ignore_index=True)

    return kp_data, kp_details

def get_kp_ap_dst_f107():

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

def plot_all_indices_in_one(merged_data, kp_details, hourly_long_df, daily_dst=False, daily_kp=False):
    cmap = plt.cm.get_cmap('tab20')
    g_index_colors = {f'G{i}': cmap(i / 5) for i in range(1, 6)}  # Dynamically create colors for G1 to G5

    fig, ax1 = plt.subplots(figsize=(15, 9))

    # Set background color blocks based on G-index
    legend_handles = []  # List to hold the legend handles
    used_labels = set()  # Set to track which labels have been used
    for idx, row in kp_details.iterrows():
        g_index = row['storm_scale']
        if g_index in g_index_colors:
            if g_index not in used_labels:
                # Create a patch for the legend if it's the first time this label is used
                patch = plt.Rectangle((0, 0), 1, 1, color=g_index_colors[g_index], label=g_index)
                legend_handles.append(patch)
                used_labels.add(g_index)
            ax1.axvspan(row['DateTime'], row['DateTime'] + pd.Timedelta(hours=3), color=g_index_colors[g_index], alpha=0.3)

    if daily_dst:
        color = 'black'
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Dst', color=color)
        ax1.plot(merged_data['Date'], merged_data['DailyMean'], color=color, alpha=0.5)
        ax1.tick_params(axis='y', labelcolor=color)
    else:
        color = 'blue'
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Dst', color=color)
        ax1.scatter(hourly_long_df['DateTime'], hourly_long_df['Value'], color=color, alpha=0.5, s=1)
        ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'red'
    if daily_kp:
        ax2.set_ylabel('Kp', color=color)
        ax2.plot(merged_data['Date'], merged_data['Kp_avg'], color=color, alpha=0.5)
    else:
        ax2.set_ylabel('Kp', color=color)
        ax2.scatter(kp_details['DateTime'], kp_details['Kp'], color=color, alpha=0.5, s=1)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color = 'green'
    ax3.set_ylabel('Ap', color=color)
    ax3.plot(merged_data['Date'], merged_data['Ap'], color=color, alpha=0.5)
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    color = 'purple'
    ax4.set_ylabel('SN', color=color)
    ax4.plot(merged_data['Date'], merged_data['SN'], color=color, alpha=0.5)
    ax4.tick_params(axis='y', labelcolor=color)

    ax5 = ax1.twinx()
    ax5.spines['right'].set_position(('outward', 180))
    color = 'orange'
    ax5.set_ylabel('F10.7obs', color=color)
    ax5.plot(merged_data['Date'], merged_data['F10.7obs'], color=color, alpha=0.5)
    ax5.tick_params(axis='y', labelcolor=color)

    # Adjust layout to make room for the legend
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    # Add the legend with unique entries
    ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), title="G-storm Level")
    plt.title('Geomagnetic and Solar Indices Over Time Including Dst')

    plt.show()

import plotly.graph_objects as go

def plot_all_indices_in_one_plotly(merged_data, kp_details, hourly_long_df, daily_dst=False, daily_kp=False):
    cmap = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)']
    g_index_colors = {f'G{i}': cmap[i - 1] for i in range(1, 6)}  # Define colors for G1 to G5

    fig = go.Figure()

    # Set background color blocks based on G-index, merging consecutive periods
    current_start = None
    current_g_index = None
    for idx, row in kp_details.iterrows():
        if current_g_index != row['storm_scale']:
            if current_start is not None:
                fig.add_vrect(
                    x0=current_start, x1=row['DateTime'],
                    fillcolor=g_index_colors[current_g_index], opacity=0.3,
                    layer="below", line_width=0
                )
            current_start = row['DateTime']
            current_g_index = row['storm_scale']

    # Add the last vrect if needed
    if current_start is not None:
        fig.add_vrect(
            x0=current_start, x1=kp_details['DateTime'].iloc[-1] + pd.Timedelta(hours=3),
            fillcolor=g_index_colors[current_g_index], opacity=0.3,
            layer="below", line_width=0
        )

    # Plot Dst or hourly data
    if daily_dst:
        fig.add_trace(go.Scattergl(x=merged_data['Date'], y=merged_data['DailyMean'],
                                   mode='lines', name='Daily Mean Dst',
                                   line=dict(color='black')))
    else:
        fig.add_trace(go.Scattergl(x=hourly_long_df['DateTime'], y=hourly_long_df['Value'],
                                   mode='markers', name='Hourly Dst',
                                   marker=dict(color='blue', size=3)))

    # Plot Kp or average Kp
    if daily_kp:
        fig.add_trace(go.Scattergl(x=merged_data['Date'], y=merged_data['Kp_avg'],
                                   mode='lines', name='Daily Avg Kp',
                                   line=dict(color='red'), yaxis='y2'))
    else:
        fig.add_trace(go.Scattergl(x=kp_details['DateTime'], y=kp_details['Kp'],
                                   mode='markers', name='Hourly Kp',
                                   marker=dict(color='red', size=3), yaxis='y2'))

    # Additional indices with axis specifications
    fig.add_trace(go.Scattergl(x=merged_data['Date'], y=merged_data['Ap'], name='Ap',
                               line=dict(color='green'), yaxis='y3'))
    fig.add_trace(go.Scattergl(x=merged_data['Date'], y=merged_data['SN'], name='SN',
                               line=dict(color='purple'), yaxis='y4'))
    fig.add_trace(go.Scattergl(x=merged_data['Date'], y=merged_data['F10.7obs'], name='F10.7obs',
                               line=dict(color='orange'), yaxis='y5'))

    # Configure layout and axes
    fig.update_layout(
        xaxis=dict(title='DateTime'),
        yaxis=dict(title='Dst', color='blue', side='left', position=0.05),
        yaxis2=dict(title='Kp', color='red', overlaying='y', side='right', position=0.95),
        yaxis3=dict(title='Ap', color='green', overlaying='y', side='right', position=0.90),
        yaxis4=dict(title='SN', color='purple', overlaying='y', side='right', position=0.85),
        yaxis5=dict(title='F10.7obs', color='orange', overlaying='y', side='right', position=0.80),
        title='Geomagnetic and Solars Indices Over Time Including Dst'
    )

    fig.show()

def plot_all_indices_separate(merged_data, kp_details, hourly_long_df, daily_dst=False, daily_kp=False):
    fig, ax1 = plt.subplots(figsize=(15, 9))

    if daily_dst:
        color = 'black'
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Dst', color=color)
        ax1.plot(merged_data['Date'], merged_data['DailyMean'], color=color, alpha=0.5)
        ax1.tick_params(axis='y', labelcolor=color)
    else:
        color = 'blue'
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Dst', color=color)
        ax1.scatter(hourly_long_df['DateTime'], hourly_long_df['Value'], color=color, alpha=0.5, s=1)
        ax1.tick_params(axis='y', labelcolor=color)

    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    color = 'red'
    ax2.set_ylabel('Kp', color=color)
    if daily_kp:
        ax2.scatter(merged_data['Date'], merged_data['Kp_avg'], color=color, alpha=0.5, s=1)
    else:
        ax2.scatter(kp_details['DateTime'], kp_details['Kp'], color=color, alpha=0.5, s=1)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    color = 'green'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Ap', color=color)
    ax3.plot(merged_data['Date'], merged_data['Ap'], color=color, alpha=0.5)
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax1.twinx()
    color = 'purple'
    ax4.spines['right'].set_position(('outward', 120))
    ax4.set_ylabel('SN', color=color)
    ax4.plot(merged_data['Date'], merged_data['SN'], color=color, alpha=0.5)
    ax4.tick_params(axis='y', labelcolor=color)

    ax5 = ax1.twinx()
    color = 'orange'
    ax5.spines['right'].set_position(('outward', 180))
    ax5.set_ylabel('F10.7obs', color=color)
    ax5.plot(merged_data['Date'], merged_data['F10.7obs'], color=color, alpha=0.5)
    ax5.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Geomagnetic and Solar Indices Over Time')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.show()


def filter_by_date_range(df, start, end):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    return df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)]

def distribute_selection(storm_dates, min_days_apart, max_count, already_selected):
    selected_dates = []
    for date in storm_dates:
        if not any((date - sel_date).days < min_days_apart and (date - sel_date).days >= 0 for sel_date in already_selected):
            if not selected_dates or (date - selected_dates[-1]).days >= min_days_apart:
                selected_dates.append(date)
                if len(selected_dates) >= max_count:
                    break
    return selected_dates

def select_storms(kp_3hrly):
    satellite_periods = {
        'CHAMP': ('2001-01-01', '2010-12-31'),
        'GRACE-FO-A': ('2019-01-01', '2024-06-01'),
        'TerraSAR-X': ('2010-01-01', '2024-06-01')
    }

    storm_levels = ['G5']  # Process from highest to lowest
    storm_selections = {sat: {level: [] for level in storm_levels} for sat in satellite_periods}

    # Collect storm dates per satellite and storm level
    for satellite, (start, end) in satellite_periods.items():
        filtered_storms = filter_by_date_range(kp_3hrly, start, end)
        already_selected = []

        for level in storm_levels:
            storm_dates = filtered_storms[filtered_storms['storm_scale'] == level]['DateTime'].dt.date.unique()
            storm_dates.sort()

            selected_dates = distribute_selection(storm_dates, 5, 10, already_selected)
            storm_selections[satellite][level] = selected_dates
            already_selected.extend(selected_dates)
            already_selected.sort()  # Keep the list sorted to maintain the order

    # Write selected storm periods to a file and print in the specified format
    with open("output/DensityInversion/PODBasedAccelerometry/selected_storms.txt", "w") as file:
        for satellite, levels in storm_selections.items():
            file.write(f"{satellite} Satellite:\n")
            for level, dates in levels.items():
                formatted_dates = ' '.join(f"datetime.date({d.year}, {d.month}, {d.day})" for d in dates)
                file.write(f"  {level}: [{formatted_dates}]\n")
            file.write("\n")

def convert_timestamps_to_utc(timestamps):
    timestamps_seconds = timestamps / 1000
    utc_datetimes = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) for ts in timestamps_seconds]
    return utc_datetimes

def read_imf(start_date, end_date):
    def read_ace_data(year, base_path="external/SWIndices/InterPlanMagFieldData"):
        ace_base_path = os.path.join(base_path, f"ACE_data/magswe_{year}.txt")
        ace_cols = ['year', 'day_of_year', 'hour', 'minute', 'seconds', 'Bx', 'By', 'Bz']
        ace_data = pd.read_csv(ace_base_path, delim_whitespace=True, names=ace_cols)

        for index, row in ace_data.head(10).iterrows():
            print(row.to_dict())

        def construct_datetime(row):
            year = int(row['year'])
            day_of_year = int(row['day_of_year'])
            hour = int(row['hour'])
            minute = int(row['minute'])
            second = int(row['seconds'])
            date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day_of_year - 1)
            return pd.Timestamp(year=year, month=date.month, day=date.day, hour=hour, minute=minute, second=second)

        ace_data['DateTime'] = ace_data.apply(construct_datetime, axis=1)
        return ace_data

    def read_discvr_data(year, base_path="external/SWIndices/InterPlanMagFieldData"):
        discvr_dir = os.path.join(base_path, "DISCVR_data")
        discvr_files = [f for f in os.listdir(discvr_dir) if f.startswith(f'oe_m1m_dscovr_s{year}') and f.endswith('.nc.gz')]
        
        if not discvr_files:
            return None

        all_discvr_data = []

        for file_name in discvr_files:
            file_path = os.path.join(discvr_dir, file_name)

            with gzip.open(file_path, 'rb') as gz_file:
                temp_file_path = file_path.replace('.gz', '')
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(gz_file.read())

            dataset = nc.Dataset(temp_file_path, 'r')
            time = convert_timestamps_to_utc(dataset.variables['time'][:])
            bx_gsm = dataset.variables['bx_gsm'][:]
            by_gsm = dataset.variables['by_gsm'][:]
            bz_gsm = dataset.variables['bz_gsm'][:]

            discvr_data = pd.DataFrame({
                'DateTime': time,
                'Bx': bx_gsm,
                'By': by_gsm,
                'Bz': bz_gsm
            })

            all_discvr_data.append(discvr_data)
            dataset.close()
            os.remove(temp_file_path)

        return pd.concat(all_discvr_data)

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    all_data = []

    for year in range(start_year, end_year + 1):
        discvr_data = read_discvr_data(year)
        if discvr_data is not None:
            print(f"DISCVR data for {year} found.")
            all_data.append(discvr_data)
        else:
            print(f"DISCVR data for {year} not found.\n Fetching ACE data.")
            ace_data = read_ace_data(year)
            all_data.append(ace_data)

    imf_df = pd.concat(all_data)
    imf_df = imf_df[(imf_df['DateTime'] >= start_date) & (imf_df['DateTime'] <= end_date)]
    return imf_df

if __name__ == "__main__":
    ####Plot Space Weather Indices
    # daily_indices, kp_3hrly, hourly_dst = get_kp_ap_dst_f107()

    # kp_3hrly = kp_3hrly[kp_3hrly['DateTime'] > '2000-01-01']
    # daily_indices = daily_indices[daily_indices['Date'] > '2000-01-01']  
    # hourly_dst = hourly_dst[hourly_dst['DateTime'] > '2000-01-01'] 

    # select_storms(kp_3hrly)

    # print("Storm selection completed and written to 'selected_storms.txt'.")

    # plot_all_indices_separate(daily_indices, kp_3hrly, hourly_dst, daily_dst=False, daily_kp=False)
    # plot_all_indices_in_one(daily_indices, kp_3hrly, hourly_dst, daily_dst=False, daily_kp=False)
    # plot_all_indices_in_one_plotly(daily_indices, kp_3hrly, hourly_dst, daily_dst=False, daily_kp=False)

    ####Test reading AE index
    # start_date = '2018-01-01'
    # end_date = '2020-02-01'
    # sym = read_ae(start_date, end_date)
    # print(sym)
    start_date = '2022-05-16'
    end_date = '2022-05-18'
    imf_df = read_imf(start_date, end_date)
    print(imf_df)