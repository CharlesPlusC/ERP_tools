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
                        residuals_file = find_file(fm_path, 'prop_residuals.npy')
                        if not diffs_file or not residuals_file:
                            continue
                        
                        epoch = diffs_file.split('_')[1]
                        
                        diffs = np.load(diffs_file, allow_pickle=True).item()
                        residuals = np.load(residuals_file, allow_pickle=True).item()
                        residuals_3d = next(iter(residuals.values()))
                        
                        cov_file = find_file(fm_path, 'cov_mats.npy')
                        rms_file = find_file(fm_path, 'RMSs.npy')

                        if not cov_file or not rms_file:
                            continue
                        
                        cov_matrix = np.load(cov_file, allow_pickle=True)
                        rms = np.load(rms_file, allow_pickle=True)

                        force_model_number = int(force_model.split('_')[1][2:])

                        for i, (cov, rms_value) in enumerate(zip(cov_matrix, rms)):
                            data.append([
                                satellite, arc_number, period.split('_')[-1], epoch, rms_value,
                                residuals_3d, diffs['H'], diffs['C'], diffs['L'], cov, force_model_number
                            ])
                    except Exception as e:
                        print(f"Error processing {fm_path}: {e}")

    columns = ["Satellite Name", "Arc Number", "Year", "arc epoch", "OD Fit RMS", 
               "3D diffs time series", "H diffs time series", "C diffs time series", 
               "L diffs time series", "cov matrix", "Force Model Number"]

    df = pd.DataFrame(data, columns=columns)
    return df


def calculate_rms_for_h_diffs(df):
    df['H diffs RMS'] = df['H diffs time series'].apply(
        lambda x: np.sqrt(np.mean(np.square(np.abs(list(x.values())[0])))) if isinstance(x, dict) else np.nan)
    return df

def calculate_percentage_improvement(df):
    df = df.sort_values(by=['Satellite Name', 'Force Model Number'])
    df['Previous RMS'] = df.groupby('Satellite Name')['H diffs RMS'].shift(1)

    for satellite in df['Satellite Name'].unique():
        for model in [6, 7, 8]:
            prev_rms = df[(df['Satellite Name'] == satellite) & (df['Force Model Number'] == 5)]['H diffs RMS'].values
            df.loc[(df['Satellite Name'] == satellite) & (df['Force Model Number'] == model), 'Previous RMS'] = prev_rms

    df['RMS Improvement %'] = ((df['Previous RMS'] - df['H diffs RMS']) / df['Previous RMS']) * 100
    df = df.dropna(subset=['RMS Improvement %'])
    df = df[(df['RMS Improvement %'] <= 110) & (df['RMS Improvement %'] >= -110)]
    
    # Drop 0% RMS improvement values
    df = df[df['RMS Improvement %'] != 0]

    return df

def plot_stripplot(df):
    colors = px.colors.qualitative.Plotly
    satellite_colors = {satellite: colors[i % len(colors)] for i, satellite in enumerate(df['Satellite Name'].unique())}
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['2019', '2023'], shared_yaxes=True)

    for col, year in enumerate(['2019', '2023'], start=1):
        df_year = df[df['Year'] == year]

        df_year['Force Model Number'] = df_year['Force Model Number'] + np.random.uniform(-0.1, 0.1, size=len(df_year))

        for satellite in df_year['Satellite Name'].unique():
            satellite_data = df_year[df_year['Satellite Name'] == satellite]
            fig.add_trace(go.Scatter(
                x=satellite_data['RMS Improvement %'],
                y=satellite_data['Force Model Number'],
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
        title='RMS Improvement % by Force Model Number and Satellite for 2019 and 2023',
        xaxis_title='% Improvement in RMS',
        yaxis_title='Force Model Number',
        legend_title='Satellite Name',
        template='plotly_white',
        showlegend=True
    )

    fig.show()

if __name__ == "__main__":
    df = pd.read_pickle('output/Myriad_FM_Bench/ProcessedResults/df.pkl')

    # df = calculate_rms_for_h_diffs(df)

    # df = calculate_percentage_improvement(df)

    # plot_stripplot(df)

    #TODO: strip plots for L_diffs, C_diffs, 3D_diffs, and OD Fit RMS
    #TODO: add median value to plot + box and whisker or violin plot
    #TODO: heatmap of RMS improvement %
    #TODO: map the force model number onto actual force model names
    #TODO: show not only the percentage improvement but the scale of the absolute value differences
    #TODO: whgat abouts some kind of aggreagte metric for overall improvement by: Conservative force field, Radiative and Drag