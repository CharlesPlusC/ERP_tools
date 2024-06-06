import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def find_file(directory, keyword):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if keyword in file:
                return os.path.join(root, file)
    return None

def calculate_rms(diff_series):
    values = next(iter(diff_series.values()))
    return np.sqrt(np.mean(np.square(values)))

def extract_data(root_folder):
    data = []

    periods = ["FM_Bench_2019", "FM_Bench_2023"]
    satellites = ["GRACE-FO-A", "GRACE-FO-B", "Sentinel-2A", "Sentinel-2B",
                  "Sentinel-3A", "Sentinel-3B", "TerraSAR-X", "TanDEM-X"]
    arcs = [f"arc_{i}" for i in range(21)]
    force_models = [f"output_fm{i}" for i in range(9)]
    
    for period in periods:
        period_path = os.path.join(root_folder, period)
        for satellite in satellites:
            satellite_path = os.path.join(period_path, satellite)
            for arc in arcs:
                arc_number = arc.split('_')[1]
                arc_path = os.path.join(satellite_path, arc)
                for force_model in force_models:
                    fm_path = os.path.join(arc_path, f"{force_model}_arc{arc_number}", satellite)
                    if not os.path.exists(fm_path):
                        continue
                    
                    try:
                        diffs_file = find_file(fm_path, 'hcl_diffs.npy')
                        
                        epoch = diffs_file.split('False_')[1].split('_fm')[0]
                        epoch = pd.to_datetime(epoch, format='%Y-%m-%d_%H-%M-%S')
                        
                        diffs = np.load(diffs_file, allow_pickle=True).item()
                        
                        cov_file = find_file(fm_path, 'cov_mats.npy')
                        od_rms_file = find_file(fm_path, 'RMSs.npy')
                        prop_3d_residuals_file = find_file(fm_path, 'prop_residuals.npy')
                        
                        cov_matrix = np.load(cov_file, allow_pickle=True)
                        od_rmss = np.load(od_rms_file, allow_pickle=True)
                        prop_3d_residuals= np.load(prop_3d_residuals_file, allow_pickle=True)
                        prop_3d_residuals = prop_3d_residuals.item()  # Convert the array element to a dictionary
                        first_key = next(iter(prop_3d_residuals))  # Get the first key from the dictionary
                        prop_3d_residuals = prop_3d_residuals[first_key][0]
                        
                        # select the smallest OD RMS value since this is the one associated with the best fit (which the state vector is based on)
                        od_rms = np.min(od_rmss)

                        force_model_number = int(force_model.split('_')[1][2:])

                        H_timeseries = diffs['H']
                        H_timeseries = next(iter(diffs['H'].values()))[0]
                        C_timeseries = diffs['C']
                        C_timeseries = next(iter(diffs['C'].values()))[0]
                        L_timeseries = diffs['L']
                        L_timeseries = next(iter(diffs['L'].values()))[0]

                        rms_H = calculate_rms(diffs['H'])
                        rms_C = calculate_rms(diffs['C'])
                        rms_L = calculate_rms(diffs['L'])
                        rms_3D = np.sqrt(np.mean(np.square(prop_3d_residuals)))

                        data.append([
                            satellite, arc_number, period.split('_')[-1], epoch, od_rms, 
                            prop_3d_residuals, H_timeseries, C_timeseries, L_timeseries,
                            rms_3D, rms_H, rms_C, rms_L, cov_matrix, force_model_number
                        ])

                    except Exception as e:
                        print(f"Error processing {fm_path}: {e}")

    columns = ["Satellite Name", "Arc Number", "Year", "arc epoch", "OD Fit RMS", 
               "Prop 3D Residuals", "H_diffs", "C_diffs", "L_diffs",
               "3D RMS", "H RMS", "C RMS", "L RMS", "cov matrix", "Force Model Number"]

    df = pd.DataFrame(data, columns=columns)
    return df

def calculate_percentage_improvement(df, diff_type):
    df = df.sort_values(by=['Satellite Name', 'Arc Number', 'arc epoch', 'Force Model Number'])
    df[f'{diff_type} RMS Improvement %'] = np.nan
    
    for satellite in df['Satellite Name'].unique():
        for arc in df['Arc Number'].unique():
            for epoch in df['arc epoch'].unique():
                for model in range(1, 9):
                    current_rms = df.loc[(df['Satellite Name'] == satellite) & 
                                         (df['Arc Number'] == arc) & 
                                         (df['arc epoch'] == epoch) & 
                                         (df['Force Model Number'] == model), f'{diff_type} RMS']
                    
                    if model in [6, 7, 8]:
                        previous_rms = df.loc[(df['Satellite Name'] == satellite) & 
                                              (df['Arc Number'] == arc) & 
                                              (df['arc epoch'] == epoch) & 
                                              (df['Force Model Number'] == 5), f'{diff_type} RMS']
                    else:
                        previous_rms = df.loc[(df['Satellite Name'] == satellite) & 
                                              (df['Arc Number'] == arc) & 
                                              (df['arc epoch'] == epoch) & 
                                              (df['Force Model Number'] == model-1), f'{diff_type} RMS']
                    
                    if not current_rms.empty and not previous_rms.empty:
                        improvement = 100 * (previous_rms.values[0] - current_rms.values[0]) / previous_rms.values[0]
                        df.loc[(df['Satellite Name'] == satellite) & 
                               (df['Arc Number'] == arc) & 
                               (df['arc epoch'] == epoch) & 
                               (df['Force Model Number'] == model), f'{diff_type} RMS Improvement %'] = improvement

    df = df.dropna(subset=[f'{diff_type} RMS Improvement %'])
    df = df[(df[f'{diff_type} RMS Improvement %'] <= 110) & (df[f'{diff_type} RMS Improvement %'] >= -110)]
    df = df[df[f'{diff_type} RMS Improvement %'] != 0]
    
    return df

def plot_stripplot(df, diff_type):
    colors = px.colors.qualitative.Plotly
    satellite_colors = {satellite: colors[i % len(colors)] for i, satellite in enumerate(df['Satellite Name'].unique())}
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['2019', '2023'], shared_yaxes=True)

    for col, year in enumerate(['2019', '2023'], start=1):
        df_year = df[df['Year'] == year].copy()
        df_year.loc[:, 'Force Model Number'] += np.random.uniform(-0.1, 0.1, size=len(df_year))

        for satellite in df_year['Satellite Name'].unique():
            satellite_data = df_year[df_year['Satellite Name'] == satellite]
            y_values = satellite_data['Force Model Number']
            x_values = satellite_data[f'{diff_type} RMS Improvement %'] if diff_type != 'OD Fit RMS' else satellite_data['OD Fit RMS']

            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(
                    size=5,
                    color=satellite_colors[satellite],
                    showscale=False
                ),
                name=satellite,
                legendgroup=satellite
            ), row=1, col=col)

    fig.update_layout(
        title=f'{diff_type} by Force Model Number and Satellite for 2019 and 2023',
        xaxis_title=f'{diff_type}',
        yaxis_title='Force Model Number',
        legend_title='Satellite Name',
        template='plotly_white',
        showlegend=True
    )

    fig.show()

def calculate_running_statistics(data, window_size=10):
    """Calculate running median, IQR1, and IQR3 for a given time series."""
    running_median = data.rolling(window=window_size, min_periods=1).median()
    iqr1 = data.rolling(window=window_size, min_periods=1).quantile(0.25)
    iqr3 = data.rolling(window=window_size, min_periods=1).quantile(0.75)
    return np.abs(running_median), np.abs(iqr1), np.abs(iqr3)

def plot_timeseries_raw(df, metrics, window_size=10):
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=3, 
        cols=2, 
        subplot_titles=[
            'H (2019)', 'H (2023)',
            'C (2019)', 'C (2023)',
            'L (2019)', 'L (2023)'
        ], 
        shared_yaxes=True
    )

    for metric_idx, metric in enumerate(metrics):
        for col, year in enumerate(['2019', '2023'], start=1):
            df_year = df[df['Year'] == year].copy()

            for force_model in df_year['Force Model Number'].unique():
                fm_data = df_year[df_year['Force Model Number'] == force_model]

                combined_data = pd.DataFrame(fm_data[f'{metric}_diffs'].tolist()).T

                running_median, iqr1, iqr3 = calculate_running_statistics(combined_data, window_size)

                fig.add_trace(go.Scatter(
                    x=combined_data.index,
                    y=np.log10(running_median.median(axis=1)),
                    mode='lines',
                    line=dict(color=colors[force_model % len(colors)], width=2),
                    name=f'FM {force_model} Median',
                    legendgroup=f'FM {force_model}',
                    showlegend=(metric_idx == 0 and col == 1)
                ), row=metric_idx + 1, col=col)

                fig.add_trace(go.Scatter(
                    x=combined_data.index,
                    y=np.log10(iqr1.median(axis=1)),
                    fill=None,
                    mode='lines',
                    line=dict(color=colors[force_model % len(colors)], dash='dash', width=1),
                    name=f'FM {force_model} IQR1',
                    legendgroup=f'FM {force_model}',
                    showlegend=False
                ), row=metric_idx + 1, col=col)

                fig.add_trace(go.Scatter(
                    x=combined_data.index,
                    y=np.log10(iqr3.median(axis=1)),
                    fill='tonexty',
                    mode='lines',
                    line=dict(color=colors[force_model % len(colors)], dash='dash', width=1),
                    name=f'FM {force_model} IQR3',
                    legendgroup=f'FM {force_model}',
                    showlegend=False
                ), row=metric_idx + 1, col=col)

    fig.update_layout(
        title='Running Median and IQR of Time Series Data by Metric, Year, and Force Model (Log Scale)',
        xaxis_title='Index Number',
        yaxis_title='Log(Value)',
        template='plotly_white',
        showlegend=True,
        height=1500,
        width=1500,
    )

    fig.update_xaxes(showgrid=True, zeroline=True)
    fig.update_yaxes(showgrid=True, zeroline=True)

    fig.show()


def plot_force_model_mean(df, metric, year, window_size=10):
    colors = px.colors.qualitative.Plotly
    satellites = df['Satellite Name'].unique()

    fig = make_subplots(
        rows=len(satellites), 
        cols=1, 
        subplot_titles=[f'{sat} ({year})' for sat in satellites],
        shared_yaxes=True
    )

    for row, satellite in enumerate(satellites, start=1):
        df_satellite = df[(df['Year'] == year) & (df['Satellite Name'] == satellite)].copy()

        for force_model in df_satellite['Force Model Number'].unique():
            fm_data = df_satellite[df_satellite['Force Model Number'] == force_model]

            combined_data = pd.DataFrame(fm_data[f'{metric}_diffs'].tolist()).T

            running_median, iqr1, iqr3 = calculate_running_statistics(combined_data, window_size)

            fig.add_trace(go.Scatter(
                x=combined_data.index,
                y=np.log10(running_median.median(axis=1)),
                mode='lines',
                line=dict(color=colors[force_model % len(colors)], width=2),
                name=f'FM {force_model} Median',
                legendgroup=f'FM {force_model}',
                showlegend=(row == 1)
            ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=combined_data.index,
                y=np.log10(iqr1.median(axis=1)),
                fill=None,
                mode='lines',
                line=dict(color=colors[force_model % len(colors)], dash='dash', width=1),
                name=f'FM {force_model} IQR1',
                legendgroup=f'FM {force_model}',
                showlegend=False
            ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=combined_data.index,
                y=np.log10(iqr3.median(axis=1)),
                fill='tonexty',
                mode='lines',
                line=dict(color=colors[force_model % len(colors)], dash='dash', width=1),
                name=f'FM {force_model} IQR3',
                legendgroup=f'FM {force_model}',
                showlegend=False
            ), row=row, col=1)

    fig.update_layout(
        title=f'Running Median and IQR of {metric} Time Series Data by Satellite for {year} (Log Scale)',
        xaxis_title='Index Number',
        yaxis_title='Log(Value)',
        template='plotly_white',
        showlegend=True,
        height=1500,
        width=1500,
    )

    fig.update_xaxes(showgrid=True, zeroline=True)
    fig.update_yaxes(showgrid=True, zeroline=True)

    fig.show()

def plot_all_metrics_all_years(df, metrics, window_size=10):
    colors = px.colors.qualitative.Plotly
    satellites = df['Satellite Name'].unique()

    fig = make_subplots(
        rows=len(satellites), 
        cols=len(metrics) * 2, 
        subplot_titles=[
            f'{sat} {metric} (2019)' for sat in satellites for metric in metrics
        ] + [
            f'{sat} {metric} (2023)' for sat in satellites for metric in metrics
        ],
        shared_yaxes=True
    )

    col_idx = {f'{metric}_{year}': idx + 1 for idx, (metric, year) in enumerate([(m, '2019') for m in metrics] + [(m, '2023') for m in metrics])}

    for row, satellite in enumerate(satellites, start=1):
        for metric in metrics:
            for year in ['2019', '2023']:
                col = col_idx[f'{metric}_{year}']
                df_satellite = df[(df['Year'] == year) & (df['Satellite Name'] == satellite)].copy()

                for force_model in df_satellite['Force Model Number'].unique():
                    fm_data = df_satellite[df_satellite['Force Model Number'] == force_model]

                    combined_data = pd.DataFrame(fm_data[f'{metric}_diffs'].tolist()).T

                    running_median, iqr1, iqr3 = calculate_running_statistics(combined_data, window_size)

                    fig.add_trace(go.Scatter(
                        x=combined_data.index,
                        y=np.log10(running_median.median(axis=1)),
                        mode='lines',
                        line=dict(color=colors[force_model % len(colors)], width=2),
                        name=f'FM {force_model} Median',
                        legendgroup=f'FM {force_model}',
                        showlegend=(row == 1 and col == 1)
                    ), row=row, col=col)

                    fig.add_trace(go.Scatter(
                        x=combined_data.index,
                        y=np.log10(iqr1.median(axis=1)),
                        fill=None,
                        mode='lines',
                        line=dict(color=colors[force_model % len(colors)], dash='dash', width=1),
                        name=f'FM {force_model} IQR1',
                        legendgroup=f'FM {force_model}',
                        showlegend=False
                    ), row=row, col=col)

                    fig.add_trace(go.Scatter(
                        x=combined_data.index,
                        y=np.log10(iqr3.median(axis=1)),
                        fill='tonexty',
                        mode='lines',
                        line=dict(color=colors[force_model % len(colors)], dash='dash', width=1),
                        name=f'FM {force_model} IQR3',
                        legendgroup=f'FM {force_model}',
                        showlegend=False
                    ), row=row, col=col)

    fig.update_layout(
        title='Running Median and IQR of Time Series Data by Satellite, Metric, and Year (Log Scale)',
        xaxis_title='Index Number',
        yaxis_title='Log(Value)',
        template='plotly_white',
        showlegend=True,
        height=2000,
        width=3000,
        title_font=dict(size=14)
    )

    fig.update_xaxes(showgrid=True, zeroline=True)
    fig.update_yaxes(showgrid=True, zeroline=True)
    fig.update_annotations(font_size=10)

    fig.show()
    
if __name__ == "__main__":
    df = extract_data('output/Myriad_FM_Bench')

    print(f"length of H_diffs: {len(df['H_diffs'])}")
    print(f"total number of values in the first H_diffs: {len(df['H_diffs'][0])}")

    # print(f"head of OD Fit RMS: {df['OD Fit RMS'].head()}")
    # print(f"head of H RMS: {df['H RMS'].head()}")
    # print(f"head of h timeseries: {df['H_diffs'].head()}")
    print(f"head of 3D residuals: {df['Prop 3D Residuals'].head()}")

    # Calculate RMS improvement for each diff type ('H', 'C', 'L', '3D', 'OD Fit')
    # for diff_type in ['3D', 'OD Fit']:
    #     df_improvement = calculate_percentage_improvement(df, diff_type)
    #     plot_stripplot(df_improvement, diff_type)

    # plot_timeseries_raw(df, metrics=['H', 'C', 'L'])
    # plot_force_model_mean(df, metric='H', year='2023')
    plot_all_metrics_all_years(df, metrics=['H', 'C', 'L'])


    #TODO: strip plots for L_diffs, C_diffs, 3D_diffs, and OD Fit RMS
    #TODO: add median value to plot + box and whisker or violin plot
    #TODO: heatmap of RMS improvement %
    #TODO: map the force model number onto actual force model names
    #TODO: show not only the percentage improvement but the scale of the absolute value differences
    #TODO: whgat abouts some kind of aggreagte metric for overall improvement by: Conservative force field, Radiative and Drag