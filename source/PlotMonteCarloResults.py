import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_theme(style="dark")

# Function to calculate collision probabilities for a given radius
def calculate_collision_probabilities(df, radius):
    return np.mean(df['DCA'] <= radius) * 100

# Function to plot TCA vs. DCA with scatter, hist, and KDE
def plot_tca_vs_dca(dataframes, filenames, save_path, sat_name):
    for df, filename in zip(dataframes, filenames):
        f, ax = plt.subplots(figsize=(6, 6))

        # Converting TCA to seconds since the first TCA
        tca_seconds = (pd.to_datetime(df['TCA']) - pd.to_datetime(df['TCA'].iloc[0])).dt.total_seconds()
        dca = df['DCA']

        # Draw a combo histogram and scatterplot with density contours
        sns.scatterplot(x=tca_seconds, y=dca, s=5, color=".15", ax=ax)
        sns.histplot(x=tca_seconds, y=dca, bins=50, pthresh=.1, cmap="mako", ax=ax)
        sns.kdeplot(x=tca_seconds, y=dca, levels=5, color="w", linewidths=1, ax=ax)

        ax.set_title(f'{sat_name} - TCA vs. DCA ({filename})')
        ax.set_xlabel('Time of Closest Approach (seconds)')
        ax.set_ylabel('Distance of Closest Approach (meters)')
        #add grid with black color
        ax.grid(color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{sat_name}_{filename}_TCA_vs_DCA.png'))
        plt.close()

def plot_tca_distributions_facetgrid(dataframes, filenames, save_path, sat_name):
    combined_df = pd.DataFrame()
    for df, filename in zip(dataframes, filenames):
        unique_id = filename.split('_')[3]  # Adjust based on your filename structure
        label = f"fm{unique_id}"
        tca_seconds = (pd.to_datetime(df['TCA']) - pd.to_datetime(df['TCA'].iloc[0])).dt.total_seconds()
        temp_df = pd.DataFrame({'TCA': tca_seconds, 'Label': label})
        combined_df = pd.concat([combined_df, temp_df])

    # Determine maximum TCA value range for x-axis limits
    max_tca = max(abs(combined_df['TCA'].min()), combined_df['TCA'].max())

    # Create the FacetGrid object
    pal = sns.cubehelix_palette(len(dataframes), rot=-.25, light=.7)
    g = sns.FacetGrid(combined_df, row="Label", hue="Label", aspect=15, height=.5, palette=pal)

    # Draw the densities and add the red line at TCA=0
    g.map(sns.kdeplot, "TCA", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(plt.axvline, x=0, color='red', linestyle='--')

    # Add vertical lines for mean, first, and third quartiles
    def add_lines(x, **kwargs):
        mean = np.mean(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        plt.axvline(mean, color='green', linestyle='--')
        plt.axvline(q1, color='blue', linestyle=':')
        plt.axvline(q3, color='blue', linestyle=':')

    g.map(add_lines, "TCA")

    # Function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=9)

    g.map(label, "TCA")

    # Adjust subplot parameters and remove axes details that don't play well with overlap
    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Add the overarching title
    plt.suptitle(f'{sat_name} - TCA Distributions', y=1.02, fontsize=16)

    plt.savefig(os.path.join(save_path, f'{sat_name}_TCA_Distributions_FacetGrid.png'))
    plt.close()

# Function to plot the probability of collision estimate and save the plot
def plot_collision_probability_estimate(probabilities, filenames, save_path, sat_name):
    fig, ax = plt.subplots()
    ax.bar(filenames, probabilities, color='skyblue')
    ax.set_xlabel('File')
    ax.set_ylabel('Probability of Collision (%)')
    ax.set_title(f'{sat_name} - Probability of Collision Estimate')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 100)
    ax.grid(True)

    for i, v in enumerate(probabilities):
        ax.text(i, max(v, 1e-4) * 1.1, f"{v:.4f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{sat_name}_Probability_of_Collision_Estimate.png'))
    plt.close()

# Function to plot cumulative distribution for DCA across all files for a satellite
def plot_cumulative_distribution_dca(dataframes, filenames, save_path, sat_name):
    fig, ax = plt.subplots()
    for df, filename in zip(dataframes, filenames):
        dca_sorted = np.sort(df['DCA'].values)
        ax.plot(dca_sorted, np.linspace(0, 100, len(dca_sorted)), label=filename)

    ax.set_title(f'{sat_name} - Cumulative Distribution of DCA')
    ax.set_xlabel('DCA (m)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{sat_name}_Cumulative_Distribution_DCA.png'))
    plt.close()

# Function to plot symmetrical histograms and line plots of TCA sample numbers for each file
def plot_tca_histograms_and_sample_lines(dataframes, filenames, save_path, sat_name):
    combined_fig, combined_ax = plt.subplots()  # For the combined plot of all TCA lines

    for df, filename in zip(dataframes, filenames):
        fig, ax = plt.subplots()
        tca_seconds = (pd.to_datetime(df['TCA']) - pd.to_datetime(df['TCA'].iloc[0])).dt.total_seconds()
        median_tca = np.median(tca_seconds)
        
        # Creating symmetric bins and plotting histogram
        max_range = max(median_tca - min(tca_seconds), max(tca_seconds) - median_tca)
        bins = np.arange(median_tca - max_range, median_tca + max_range + 0.001, 0.001)
        counts, edges = np.histogram(tca_seconds, bins=bins)

        ax.hist(tca_seconds, bins=bins, color='orange', edgecolor='black', alpha=0.5)
        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(bin_centers, counts, '-k')

        ax.set_xlim(median_tca - max_range, median_tca + max_range)

        ax.set_title(f'{sat_name} - TCA Histogram ({filename})')
        ax.set_xlabel('Time of Closest Approach (seconds)')
        ax.set_ylabel('Number of Samples')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{sat_name}_{filename}_TCA_Histogram_Sample_Numbers.png'))
        plt.close()

        # Adding the line plot to the combined plot
        combined_ax.plot(bin_centers, counts, label=filename)

    combined_ax.set_title(f'{sat_name} - Combined TCA Sample Number Lines')
    combined_ax.set_xlabel('Time of Closest Approach (seconds)')
    combined_ax.set_ylabel('Number of Samples')
    combined_ax.grid(True)
    combined_ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{sat_name}_Combined_TCA_Sample_Number_Lines.png'))
    plt.close()

# Function to plot TCA vs. DCA in a 3x3 subplot matrix
def plot_tca_vs_dca_matrix(dataframes, filenames, save_path, sat_name):
    # Setting up a 3x3 subplot matrix
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()  # Flatten to iterate easily

    for i, (df, filename) in enumerate(zip(dataframes, filenames)):
        if i >= 9:  # Only fill up to 9 subplots
            break

        # Converting TCA to seconds since the first TCA
        tca_seconds = (pd.to_datetime(df['TCA']) - pd.to_datetime(df['TCA'].iloc[0])).dt.total_seconds()
        dca = df['DCA']

        # Drawing the scatterplot, histogram, and KDE on the subplot
        sns.scatterplot(x=tca_seconds, y=dca, s=5, color=".15", ax=axs[i])
        sns.histplot(x=tca_seconds, y=dca, bins=50, pthresh=.1, cmap="mako", ax=axs[i])
        sns.kdeplot(x=tca_seconds, y=dca, levels=5, color="w", linewidths=1, ax=axs[i])

        axs[i].set_title(f'{filename}')
        axs[i].set_xlabel('TCA (seconds)')
        axs[i].set_ylabel('DCA (meters)')
        axs[i].grid(True)
        axs[i].legend()
        #make the y lim from 0-12
        axs[i].set_ylim(0, 12)
        #make the x lim from -0.008 to 0.008
        axs[i].set_xlim(-0.008, 0.008)
        #add a black grid
        axs[i].grid(color='black', linestyle='-', linewidth=0.5)

    # Hide unused subplots
    for j in range(i + 1, 9):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.suptitle(f'{sat_name} - TCA vs. DCA Comparison', y=1.02)
    plt.savefig(os.path.join(save_path, f'{sat_name}_TCA_vs_DCA_Matrix.png'))
    plt.close()


# Function to plot all TCA vs. DCA data on a single plot with KDE and hue differentiation
def plot_tca_vs_dca_jointplot(dataframes, filenames, save_path, sat_name):
    sns.set_theme(style="ticks")
    # Combine all dataframes with an identifying label
    combined_df = pd.DataFrame()
    for df, filename in zip(dataframes, filenames):
        # Extract the unique ID from the filename
        unique_id = filename.split('_')[3]  # Adjust based on your filename structure
        label = f"fm{unique_id}"

        tca_seconds = (pd.to_datetime(df['TCA']) - pd.to_datetime(df['TCA'].iloc[0])).dt.total_seconds()
        temp_df = pd.DataFrame({'TCA': tca_seconds, 'DCA': df['DCA'], 'Force Model': label})
        combined_df = pd.concat([combined_df, temp_df])

    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(
        data=combined_df,
        x="TCA", y="DCA", hue="Force Model",
        kind="kde",
        height=8,
        space=0,
    )

    plt.suptitle(f'{sat_name} - TCA vs. DCA KDE Comparison', y=1.02)
    plt.savefig(os.path.join(save_path, f'{sat_name}_TCA_vs_DCA_KDE.png'))
    plt.close()

# List of satellites to test
sat_names_to_test = ["GRACE-FO-A"]

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
    plot_tca_histograms_and_sample_lines(dataframes, filenames, save_path, sat_name)
    plot_tca_vs_dca(dataframes, filenames, save_path, sat_name)
    plot_tca_vs_dca_matrix(dataframes, files, save_path, sat_name)
    plot_tca_distributions_facetgrid(dataframes, files, save_path, sat_name)
    plot_tca_vs_dca_jointplot(dataframes, files, save_path, sat_name)




