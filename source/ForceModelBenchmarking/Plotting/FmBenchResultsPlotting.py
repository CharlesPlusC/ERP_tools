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
                        
                        cov_matrix = np.load(cov_file, allow_pickle=True)
                        od_rmss = np.load(od_rms_file, allow_pickle=True)
                        # select the smallest OD RMS value since this is the one associated with the best fit (which the state vector is based on)
                        od_rms = np.min(od_rmss)

                        force_model_number = int(force_model.split('_')[1][2:])

                        rms_H = calculate_rms(diffs['H'])
                        rms_C = calculate_rms(diffs['C'])
                        rms_L = calculate_rms(diffs['L'])

                        data.append([
                            satellite, arc_number, period.split('_')[-1], epoch, od_rms,
                            rms_H, rms_C, rms_L, cov_matrix, force_model_number
                        ])

                    except Exception as e:
                        print(f"Error processing {fm_path}: {e}")

    columns = ["Satellite Name", "Arc Number", "Year", "arc epoch", "OD Fit RMS",
               "H RMS", "C RMS", "L RMS", "cov matrix", "Force Model Number"]

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


if __name__ == "__main__":
    df = extract_data('output/Myriad_FM_Bench')

    print(f"head of OD Fit RMS: {df['OD Fit RMS'].head()}")
    print(f"head of H RMS: {df['H RMS'].head()}")

    # Calculate RMS improvement for each diff type ('H', 'C', 'L')
    for diff_type in ['H', 'C', 'L', 'OD Fit RMS']:
        df_improvement = calculate_percentage_improvement(df, diff_type)
        plot_stripplot(df_improvement, diff_type)


    #TODO: strip plots for L_diffs, C_diffs, 3D_diffs, and OD Fit RMS
    #TODO: add median value to plot + box and whisker or violin plot
    #TODO: heatmap of RMS improvement %
    #TODO: map the force model number onto actual force model names
    #TODO: show not only the percentage improvement but the scale of the absolute value differences
    #TODO: whgat abouts some kind of aggreagte metric for overall improvement by: Conservative force field, Radiative and Drag