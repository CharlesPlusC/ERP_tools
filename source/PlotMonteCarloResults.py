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

        #now look at TCA seconds since 2023-05-05 09:59:42.000000
        tca_seconds = (pd.to_datetime(df['TCA'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce') - pd.to_datetime('2023-05-05 09:59:42.000000', format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')).dt.total_seconds()

        dca = df['DCA']

        # Draw a combo histogram and scatterplot with density contours
        sns.scatterplot(x=tca_seconds, y=dca, s=5, color=".15", ax=ax)
        sns.histplot(x=tca_seconds, y=dca, bins=50, pthresh=.1, cmap="rocket", cbar=True, ax=ax)
        sns.kdeplot(x=tca_seconds, y=dca, levels=4, color="xkcd:white", linewidths=1, ax=ax)

        ax.set_title(f'{sat_name} - TCA vs. DCA ({filename})')
        ax.set_xlabel('Δ Nominal TCA (seconds)')
        ax.set_ylabel('Δ Nominal DCA (meters)')
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{sat_name}_{filename}_TCA_vs_DCA.png'))
        plt.close()

def plot_tca_distributions_facetgrid(dataframes, filenames, save_path, sat_name):
    combined_df = pd.DataFrame()
    for df, filename in zip(dataframes, filenames):
        unique_id = filename.split('_')[3]  # Adjust based on your filename structure
        label = f"fm{unique_id}"
        tca_seconds = (pd.to_datetime(df['TCA'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce') - pd.to_datetime('2023-05-05 09:59:42.000000', format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')).dt.total_seconds()
        temp_df = pd.DataFrame({'TCA': tca_seconds, 'Label': label})
        combined_df = pd.concat([combined_df, temp_df])

    # Determine maximum TCA value range for x-axis limits
    max_tca = max(abs(combined_df['TCA'].min()), combined_df['TCA'].max())

    # Create the FacetGrid object
    pal = sns.cubehelix_palette(len(dataframes), light=.7)
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
                ha="left", va="center", transform=ax.transAxes, fontsize=11)

    g.map(label, "TCA")

    g.set_axis_labels("ΔTCA (seconds)")
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Add the overarching title
    plt.suptitle(f'{sat_name} - TCA Distributions', fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f'{sat_name}_TCA_Distributions_FacetGrid.png'))
    plt.close()

def plot_dca_distributions_facetgrid(dataframes, filenames, save_path, sat_name):
    combined_df = pd.DataFrame()
    for df, filename in zip(dataframes, filenames):
        unique_id = filename.split('_')[3]  # Adjust based on your filename structure
        label = f"fm{unique_id}"
        dca_values = df['DCA']
        temp_df = pd.DataFrame({'DCA': dca_values, 'Label': label})
        combined_df = pd.concat([combined_df, temp_df])

    # Find the maximum DCA value to set symmetrical x-axis limits
    max_dca = max(abs(combined_df['DCA'].min()), combined_df['DCA'].max())

    # Create the FacetGrid object
    pal = sns.cubehelix_palette(len(dataframes), light=.7)
    g = sns.FacetGrid(combined_df, row="Label", hue="Label", aspect=15, height=.5, palette=pal)

    # Draw the densities
    g.map(sns.kdeplot, "DCA", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)

    # Add vertical lines for mean and quartiles and a reference line at DCA=0
    def add_lines(x, **kwargs):
        mean = np.mean(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        plt.axvline(mean, color='green', linestyle='--')
        plt.axvline(q1, color='blue', linestyle=':')
        plt.axvline(q3, color='blue', linestyle=':')
        plt.axvline(0, color='red', linestyle='--')

    g.map(add_lines, "DCA")

    # Function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=11)

    g.map(label, "DCA")

    # Set symmetrical x-axis limits based on the maximum DCA value
    g.set(xlim=(-max_dca, max_dca))

    g.set_axis_labels("DCA (meters)")
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    plt.suptitle(f'{sat_name} - DCA Distributions', fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f'{sat_name}_DCA_Distributions_FacetGrid.png'))
    plt.close()

# Function to plot the probability of collision estimate and save the plot
def plot_collision_probability_estimate(probabilities, filenames, save_path, sat_name):
    fig, ax = plt.subplots()
    ax.bar(filenames, probabilities, color='orange')
    ax.set_xlabel('Force Model')
    ax.set_ylabel('Probability of Collision (%)')
    #make the title be a little bit separated upwards
    ax.set_title(f'{sat_name} - Probability of Collision Estimate', y=1.1)
    # ax.set_yscale('log')
    ax.set_ylim(0, 100)
    ax.grid(color='black', linestyle='-', linewidth=0.3)

    for i, v in enumerate(probabilities):
        ax.text(i, max(v, 1e-4) * 1.05, f"{v:.2f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{sat_name}_Probability_of_Collision_Estimate.png'))
    plt.close()

# Function to plot cumulative distribution for DCA across all files for a satellite
def plot_cumulative_distribution_dca(dataframes, filenames, save_path, sat_name):
    fig, ax = plt.subplots()
    for df, filename in zip(dataframes, filenames):
        dca_sorted = np.sort(df['DCA'].values)
        ax.plot(dca_sorted, np.linspace(0, 100, len(dca_sorted)), label=filename)
        ax.grid(color='black', linestyle='-', linewidth=0.5)

    ax.set_title(f'{sat_name} - Cumulative Distribution of DCA')
    ax.set_xlabel('DCA (m)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{sat_name}_Cumulative_Distribution_DCA.png'))
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
        tca_seconds = (pd.to_datetime(df['TCA'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce') - pd.to_datetime('2023-05-05 09:59:42.000000', format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')).dt.total_seconds()
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
        axs[i].set_yscale('log')
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

        tca_seconds = (pd.to_datetime(df['TCA'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce') - pd.to_datetime('2023-05-05 09:59:42.000000', format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')).dt.total_seconds()
        temp_df = pd.DataFrame({'TCA': tca_seconds, 'DCA': df['DCA'], 'Force Model': label})
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(
        data=combined_df,
        x="TCA", y="DCA", hue="Force Model",
        kind="kde",
        height=8,
        space=0,
    )

    # Change x label to Time of Closest Approach (seconds)
    g.set_axis_labels("ΔTCA (seconds)", "ΔDCA (meters)")

    g.ax_joint.set_yscale('log')

    # Add grid with black color
    g.ax_joint.grid(color='black', linestyle='-', linewidth=0.3)

    # Set figure title
    g.figure.suptitle(f'{sat_name} - TCA vs. DCA KDE')

    # Tight layout
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f'{sat_name}_TCA_vs_DCA_KDE.png'))
    plt.close()
# List of satellites to test
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
    plot_tca_vs_dca_jointplot(dataframes, files, save_path, sat_name)