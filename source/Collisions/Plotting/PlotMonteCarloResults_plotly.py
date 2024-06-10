import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import os
from plotly.subplots import make_subplots

# Function to calculate collision probabilities for a given radius
def calculate_collision_probabilities(df, radius):
    return np.mean(df['DCA'] <= radius) * 100

# Function to plot TCA vs. DCA with a 2D density plot heatmap
def plot_tca_vs_dca(dataframes, filenames, save_path, sat_name):
    for df, filename in zip(dataframes, filenames):
        tca_seconds = (pd.to_datetime(df['TCA'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce') - pd.to_datetime('2023-05-05 09:59:42.000000', format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')).dt.total_seconds()

        # Apply log transformation to DCA to match the logarithmic scale
        dca_log = np.log10(df['DCA'].replace(0, np.nan).dropna())

        fig = go.Figure()

        # Create 2D density plot
        heatmap = go.Histogram2d(
            x=tca_seconds,
            y=dca_log,
            colorscale='Turbo',
            nbinsx=200,
            nbinsy=200,
            colorbar=dict(title='Density')
        )
        fig.add_trace(heatmap)

        # Add scatter plot overlay
        scatter = go.Scatter(
            x=tca_seconds,
            y=dca_log,
            mode='markers',
            marker=dict(size=1, color='white', opacity=0.8),
            name='Data Points'
        )
        fig.add_trace(scatter)

        fig.update_layout(
            title=f'{sat_name} - TCA vs. DCA ({filename})',
            xaxis_title='Δ Nominal TCA (seconds)',
            yaxis_title='Log10 Δ Nominal DCA (meters)',
            yaxis=dict(
                title='Δ Nominal DCA (meters)',
                tickmode='array',
                tickvals=[-3, -2, -1, 0, 1, 2, 3],
                ticktext=['0.001', '0.01', '0.1', '1', '10', '100', '1000']
            ),
            showlegend=False,
            autosize=False,
            width=1000,
            height=800
        )

        fig.show()
        fig.write_image(os.path.join(save_path, f'{sat_name}_{filename}_TCA_vs_DCA_plotly.jpg'), scale=2)

def plot_tca_distributions_facetgrid(dataframes, filenames, save_path, sat_name):
    combined_df = pd.DataFrame()
    for df, filename in zip(dataframes, filenames):
        unique_id = filename.split('_')[3]
        label = f"fm{unique_id}"
        tca_seconds = (pd.to_datetime(df['TCA'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce') - pd.to_datetime('2023-05-05 09:59:42.000000', format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')).dt.total_seconds()
        temp_df = pd.DataFrame({'TCA': tca_seconds, 'Label': label})
        combined_df = pd.concat([combined_df, temp_df])

    fig = px.histogram(combined_df, x="TCA", color="Label", marginal="box", title=f'{sat_name} - TCA Distributions', nbins=100)
    fig.show()
    fig.write_image(os.path.join(save_path, f'{sat_name}_TCA_Distributions_FacetGrid_plotly.jpg'), scale=2)

def plot_dca_distributions_facetgrid(dataframes, filenames, save_path, sat_name):
    combined_df = pd.DataFrame()
    for df, filename in zip(dataframes, filenames):
        unique_id = filename.split('_')[3]
        label = f"fm{unique_id}"
        dca_values = df['DCA']
        temp_df = pd.DataFrame({'DCA': dca_values, 'Label': label})
        combined_df = pd.concat([combined_df, temp_df])

    fig = px.histogram(combined_df, x="DCA", color="Label", marginal="box", title=f'{sat_name} - DCA Distributions', nbins=100)
    fig.update_xaxes(range=[-combined_df['DCA'].max(), combined_df['DCA'].max()])
    fig.show()
    fig.write_image(os.path.join(save_path, f'{sat_name}_DCA_Distributions_FacetGrid_plotly.jpg'), scale=2)

# Function to plot the probability of collision estimate and save the plot
def plot_collision_probability_estimate(probabilities, filenames, save_path, sat_name):
    fig = go.Figure(data=[go.Bar(x=filenames, y=probabilities, marker_color='orange')])
    fig.update_layout(
        title=f'{sat_name} - Probability of Collision Estimate',
        xaxis_title='Force Model',
        yaxis_title='Probability of Collision (%)',
        yaxis=dict(range=[0, 100]),
    )
    fig.show()
    fig.write_image(os.path.join(save_path, f'{sat_name}_Probability_of_Collision_Estimate_plotly.jpg'), scale=2)

# Function to plot cumulative distribution for DCA across all files for a satellite
def plot_cumulative_distribution_dca(dataframes, filenames, save_path, sat_name):
    fig = go.Figure()
    for df, filename in zip(dataframes, filenames):
        dca_sorted = np.sort(df['DCA'].values)
        fig.add_trace(go.Scatter(x=dca_sorted, y=np.linspace(0, 100, len(dca_sorted)), mode='lines', name=filename))

    fig.update_layout(
        title=f'{sat_name} - Cumulative Distribution of DCA',
        xaxis_title='DCA (m)',
        yaxis_title='Cumulative Percentage (%)'
    )
    fig.show()
    fig.write_image(os.path.join(save_path, f'{sat_name}_Cumulative_Distribution_DCA_plotly.jpg'), scale=2)

# Function to plot TCA vs. DCA in a 3x3 subplot matrix
def plot_tca_vs_dca_matrix(dataframes, filenames, save_path, sat_name):
    # Create a 3x3 subplot layout
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=filenames[:9],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    all_tca_seconds = []
    all_dca_pseudo_log = []

    # Collect all data for consistent axis ranges
    for df in dataframes[:9]:  # Limit to 9 for 3x3 matrix
        tca_seconds = (pd.to_datetime(df['TCA'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce') -
                       pd.to_datetime('2023-05-05 09:59:42.000000', format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')).dt.total_seconds()
        dca_pseudo_log = np.log10(df['DCA'] + 1e-6)  # Pseudo-log transformation
        all_tca_seconds.extend(tca_seconds)
        all_dca_pseudo_log.extend(dca_pseudo_log)

    tca_min, tca_max = min(all_tca_seconds), max(all_tca_seconds)
    dca_min, dca_max = min(all_dca_pseudo_log), max(all_dca_pseudo_log)

    for i, (df, filename) in enumerate(zip(dataframes, filenames)):
        if i >= 9:
            break
        
        tca_seconds = (pd.to_datetime(df['TCA'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce') -
                       pd.to_datetime('2023-05-05 09:59:42.000000', format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')).dt.total_seconds()

        dca_pseudo_log = np.log10(df['DCA'] + 1e-6)  # Pseudo-log transformation

        row, col = divmod(i, 3)
        row += 1  # Plotly rows are 1-indexed
        col += 1  # Plotly columns are 1-indexed
        
        # Create 2D density plot
        heatmap = go.Histogram2d(
            x=tca_seconds,
            y=dca_pseudo_log,
            colorscale='Turbo',
            nbinsx=100,
            nbinsy=100,
            showscale=False,
            zmax=4  # Adjust to keep the color range consistent
        )
        fig.add_trace(heatmap, row=row, col=col)

        # Add contour lines
        contour = go.Histogram2dContour(
            x=tca_seconds,
            y=dca_pseudo_log,
            colorscale='Turbo',
            nbinsx=100,
            nbinsy=100,
            line=dict(width=0.5, color='white'),
            showscale=False,
            zmax=4  # Adjust to keep the color range consistent
        )
        fig.add_trace(contour, row=row, col=col)

        fig.update_xaxes(range=[tca_min, tca_max], title_text='Δ Nominal TCA (seconds)', row=row, col=col)
        fig.update_yaxes(
            range=[dca_min, dca_max],
            title_text='Δ log Nominal DCA (meters)',
            tickmode='array',
            tickvals=[-6, -3, 0, 3, 6],
            ticktext=['1e-6', '1e-3', '1', '1e3', '1e6'],
            row=row, col=col
        )

    fig.update_layout(
        title=f'{sat_name} - TCA vs. DCA Comparison',
        height=1200,
        width=1700,
        showlegend=False
    )

    fig.show()
    fig.write_image(os.path.join(save_path, f'{sat_name}_TCA_vs_DCA_Matrix_plotly.jpg'), scale=2)

sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TanDEM-X", "TerraSAR-X"]

for sat_name in sat_names_to_test:
    base_path = f'output/Collisions/MC/TCA_DCA/{sat_name}/data'
    num_files = len([f for f in os.listdir(base_path) if f.endswith('_TCADCA.csv')])
    files = [f'sc_{sat_name}_fm_fm{i}_TCADCA.csv' for i in range(num_files)]
    file_paths = [os.path.join(base_path, file) for file in files]
    filenames = [f'File {i+1}' for i in range(len(file_paths))]

    collision_probabilities = []
    dataframes = []
    for path in file_paths:
        df = pd.read_csv(path)
        collision_probabilities.append(calculate_collision_probabilities(df, 6))
        dataframes.append(df)

    save_path = f'output/Collisions/MC/TCA_DCA/{sat_name}/plots'
    os.makedirs(save_path, exist_ok=True)

    plot_collision_probability_estimate(collision_probabilities, filenames, save_path, sat_name)
    plot_cumulative_distribution_dca(dataframes, filenames, save_path, sat_name)
    plot_tca_vs_dca(dataframes, filenames, save_path, sat_name)
    plot_tca_vs_dca_matrix(dataframes, files, save_path, sat_name)
    plot_tca_distributions_facetgrid(dataframes, files, save_path, sat_name)
    plot_dca_distributions_facetgrid(dataframes, files, save_path, sat_name)
