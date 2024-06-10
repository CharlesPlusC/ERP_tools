import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os

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
def plot_all_indices_in_one_plotly(merged_data, kp_details, hourly_long_df, daily_dst=False, daily_kp=False):
    cmap = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)']
    g_index_colors = {f'G{i}': cmap[i - 1] for i in range(1, 6)}  # Define colors for G1 to G5

    fig = go.Figure()

    # Removed the background color blocks based on G-index
    first_date = kp_details['DateTime'].min()
    last_date = kp_details['DateTime'].max()

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
        title='Geomagnetic and Solar Indices Over Time'
    )

    fig.show()
    output_path = os.path.join("output/SWIndices")
    os.makedirs(output_path, exist_ok=True)

    # Define the file name
    file_name = f"sw_indicesplot{first_date}_{last_date}.html"
    file_path = os.path.join(output_path, file_name)

    # Save the plot as an HTML file
    pio.write_html(fig, file=file_path)

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

def plot_imf_components(imf_df, start_date, end_date):
    # Filter the data for the specified date range
    imf_df = imf_df[(imf_df['DateTime'] >= start_date) & (imf_df['DateTime'] <= end_date)]

    # Create the figure
    fig = go.Figure()

    # Add traces for each component
    fig.add_trace(go.Scatter(x=imf_df['DateTime'], y=imf_df['Bx'], mode='lines', name='Bx'))
    fig.add_trace(go.Scatter(x=imf_df['DateTime'], y=imf_df['By'], mode='lines', name='By'))
    fig.add_trace(go.Scatter(x=imf_df['DateTime'], y=imf_df['Bz'], mode='lines', name='Bz'))

    # Update layout
    fig.update_layout(
        title='IMF Components (Bx, By, Bz) from 2001',
        xaxis_title='DateTime',
        yaxis_title='Magnetic Field (nT)',
        template='plotly_white'
    )

    # Show the plot
    fig.show()