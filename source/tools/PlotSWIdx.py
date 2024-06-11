import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os
from plotly.subplots import make_subplots

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

def plot_indices_dual(daily_indices, kp_3hrly, hourly_dst, time_period_1, time_period_2, daily_dst=False, daily_kp=False):
    def filter_data_by_period(data, period):
        start, end = pd.to_datetime(period[0]), pd.to_datetime(period[1])
        return data[(data['Date'] >= start) & (data['Date'] <= end)]
    
    # Filter data for both periods
    data_1 = filter_data_by_period(daily_indices, time_period_1)
    kp_details_1 = filter_data_by_period(kp_3hrly, time_period_1)
    hourly_data_1 = filter_data_by_period(hourly_dst, time_period_1)

    data_2 = filter_data_by_period(daily_indices, time_period_2)
    kp_details_2 = filter_data_by_period(kp_3hrly, time_period_2)
    hourly_data_2 = filter_data_by_period(hourly_dst, time_period_2)

    # Determine global min and max for y-axes
    dst_min = min(data_1['DailyMean'].min(), data_2['DailyMean'].min()) if daily_dst else min(hourly_data_1['Value'].min(), hourly_data_2['Value'].min())
    dst_max = max(data_1['DailyMean'].max(), data_2['DailyMean'].max()) if daily_dst else max(hourly_data_1['Value'].max(), hourly_data_2['Value'].max())

    kp_min = min(data_1['Kp_avg'].min(), data_2['Kp_avg'].min()) if daily_kp else min(kp_details_1['Kp'].min(), kp_details_2['Kp'].min())
    kp_max = max(data_1['Kp_avg'].max(), data_2['Kp_avg'].max()) if daily_kp else max(kp_details_1['Kp'].max(), kp_details_2['Kp'].max())

    ap_min = min(data_1['Ap'].min(), data_2['Ap'].min())
    ap_max = max(data_1['Ap'].max(), data_2['Ap'].max())

    sn_min = min(data_1['SN'].min(), data_2['SN'].min())
    sn_max = max(data_1['SN'].max(), data_2['SN'].max())

    f10_min = min(data_1['F10.7obs'].min(), data_2['F10.7obs'].min())
    f10_max = max(data_1['F10.7obs'].max(), data_2['F10.7obs'].max())

    # Create subplots layout
    fig = make_subplots(rows=5, cols=2, 
                        shared_yaxes=True,
                        subplot_titles=[f'Period 1: {time_period_1[0]} to {time_period_1[1]}',
                                        f'Period 2: {time_period_2[0]} to {time_period_2[1]}',
                                        "", "", "", "", "", "", "", ""],
                        vertical_spacing=0.08)

    # Plot Dst as scatter
    if daily_dst:
        fig.add_trace(go.Scatter(x=data_1['Date'], y=data_1['DailyMean'], mode='markers', name='Daily Mean Dst', marker=dict(color='black', size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_2['Date'], y=data_2['DailyMean'], mode='markers', name='Daily Mean Dst', marker=dict(color='black', size=4)), row=1, col=2)
    else:
        fig.add_trace(go.Scatter(x=hourly_data_1['DateTime'], y=hourly_data_1['Value'], mode='markers', name='Hourly Dst', marker=dict(color='blue', size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hourly_data_2['DateTime'], y=hourly_data_2['Value'], mode='markers', name='Hourly Dst', marker=dict(color='blue', size=4)), row=1, col=2)
    fig.update_yaxes(title_text="Dst", range=[dst_min, dst_max], row=1, col=1)
    fig.update_yaxes(title_text="Dst", range=[dst_min, dst_max], row=1, col=2)

    # Plot Kp as scatter
    if daily_kp:
        fig.add_trace(go.Scatter(x=data_1['Date'], y=data_1['Kp_avg'], mode='markers', name='Daily Avg Kp', marker=dict(color='red', size=4)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data_2['Date'], y=data_2['Kp_avg'], mode='markers', name='Daily Avg Kp', marker=dict(color='red', size=4)), row=2, col=2)
    else:
        fig.add_trace(go.Scatter(x=kp_details_1['DateTime'], y=kp_details_1['Kp'], mode='markers', name='Hourly Kp', marker=dict(color='red', size=4)), row=2, col=1)
        fig.add_trace(go.Scatter(x=kp_details_2['DateTime'], y=kp_details_2['Kp'], mode='markers', name='Hourly Kp', marker=dict(color='red', size=4)), row=2, col=2)
    fig.update_yaxes(title_text="Kp", range=[kp_min, kp_max], row=2, col=1)
    fig.update_yaxes(title_text="Kp", range=[kp_min, kp_max], row=2, col=2)

    # Plot Ap
    fig.add_trace(go.Scatter(x=data_1['Date'], y=data_1['Ap'], mode='lines', name='Ap', line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data_2['Date'], y=data_2['Ap'], mode='lines', name='Ap', line=dict(color='green')), row=3, col=2)
    fig.update_yaxes(title_text="Ap", range=[ap_min, ap_max], row=3, col=1)
    fig.update_yaxes(title_text="Ap", range=[ap_min, ap_max], row=3, col=2)

    # Plot SN
    fig.add_trace(go.Scatter(x=data_1['Date'], y=data_1['SN'], mode='lines', name='SN', line=dict(color='purple')), row=4, col=1)
    fig.add_trace(go.Scatter(x=data_2['Date'], y=data_2['SN'], mode='lines', name='SN', line=dict(color='purple')), row=4, col=2)
    fig.update_yaxes(title_text="SN", range=[sn_min, sn_max], row=4, col=1)
    fig.update_yaxes(title_text="SN", range=[sn_min, sn_max], row=4, col=2)

    # Plot F10.7obs
    fig.add_trace(go.Scatter(x=data_1['Date'], y=data_1['F10.7obs'], mode='lines', name='F10.7obs', line=dict(color='orange')), row=5, col=1)
    fig.add_trace(go.Scatter(x=data_2['Date'], y=data_2['F10.7obs'], mode='lines', name='F10.7obs', line=dict(color='orange')), row=5, col=2)
    fig.update_yaxes(title_text="F10.7obs", range=[f10_min, f10_max], row=5, col=1)
    fig.update_yaxes(title_text="F10.7obs", range=[f10_min, f10_max], row=5, col=2)

    # Update layout
    fig.update_layout(height=900, width=1200, title_text="Geomagnetic and Solar Indices Over Two Periods")
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)

    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_xaxes(title_text="Date", row=3, col=2)
    fig.update_xaxes(title_text="Date", row=4, col=2)
    fig.update_xaxes(title_text="Date", row=5, col=2)

    fig.show()
    #save
    file_name = f"output/SWIndices/sw_indicesplot{time_period_1[0]}_{time_period_2[1]}.html"
    pio.write_html(fig, file=file_name)

