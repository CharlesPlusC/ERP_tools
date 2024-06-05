import pandas as pd
import matplotlib.pyplot as plt
import orekit
from orekit.pyhelpers import setup_orekit_curdir
# download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants
from ..tools.utilities import interpolate_positions,calculate_acceleration, get_satellite_info, project_acc_into_HCL
from ..tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from .PODDensity import density_inversion
import numpy as np
import datetime
from tqdm import tqdm
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt

from ..tools.GFODataReadTools import get_gfo_inertial_accelerations

# podaac-data-downloader -c GRACEFO_L1B_ASCII_GRAV_JPL_RL04 -d ./GRACE-FO_A_DATA -sd 2023-05-10T06:00:00Z -ed 2023-05-11T23:59:59Z -e ".*" --verbose
# podaac-data-downloader -c GRACEFO_L1B_ASCII_GRAV_JPL_RL04 -d ./GRACE-FO_A_DATA -sd 2024-05-10T06:00:00Z -ed 2024-05-12T23:59:59Z -e ".*" --verbose

def ACT_vs_EDR_vs_POD_plot(POD_and_ACT_data, EDR_data):
    
    POD_and_ACT_data['UTC'] = pd.to_datetime(POD_and_ACT_data['UTC'])
    EDR_data['UTC'] = pd.to_datetime(EDR_data['UTC'])
    merged_data = pd.merge(POD_and_ACT_data, EDR_data, on='UTC')
    merged_data['UTC'] = pd.to_datetime(merged_data['UTC'])
    merged_data['EDR_rolling'] = (merged_data['rho_eff']*10).rolling(window=180).mean()
    POD_and_ACT_data['POD_rolling'] = POD_and_ACT_data['Velocity_Computed Density'].rolling(window=90, center=True).mean()

    merged_data = merged_data.iloc[100:]

    median_ACT = merged_data['ACT_Computed Density'].median()
    merged_data['ACT_Computed Density'] = 2 * median_ACT - merged_data['ACT_Computed Density']

    date_of_data = merged_data['UTC'].iloc[0].strftime("%Y-%m-%d")
    merged_data = merged_data[(merged_data['UTC'].dt.hour >= 2) & (merged_data['UTC'].dt.hour <= 18)]
    POD_and_ACT_data = POD_and_ACT_data[(POD_and_ACT_data['UTC'].dt.hour >= 2) & (POD_and_ACT_data['UTC'].dt.hour <= 18)]
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    #multiply ACT_computed density by -1 
    merged_data['ACT_Computed Density'] = -1 * merged_data['ACT_Computed Density']

    ax[0].plot(mdates.date2num(merged_data['UTC']), merged_data['ACT_Computed Density'], label='ACT Density', color="xkcd:teal", linewidth=1)
    ax[0].plot(mdates.date2num(merged_data['UTC']), merged_data['EDR_rolling'], label='EDR Density', color="xkcd:goldenrod", linewidth=1)
    ax[0].plot(mdates.date2num(POD_and_ACT_data['UTC']), POD_and_ACT_data['POD_rolling'], label='POD Density', color="xkcd:purple", linewidth=1)
    ax[0].plot(mdates.date2num(merged_data['UTC']), merged_data['jb08_rho'], label='JB08', linestyle='--' , color="#FF6347", linewidth=1)
    ax[0].plot(mdates.date2num(merged_data['UTC']), merged_data['dtm2000_rho'], label='DTM2000', linestyle='--', color="xkcd:cherry red", linewidth=1)
    ax[0].plot(mdates.date2num(merged_data['UTC']), merged_data['nrlmsise00_rho'], label='NRLMSISE-00', linestyle='--', color="#1E90FF", linewidth=1)
    ax[0].set_ylabel('Density (kg/m\u00B3)')
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax[0].grid(which='both', linestyle='-.')
    ax[0].set_xticklabels([])
    ax[0].set_yscale('log')
    ax[0].set_title('GRACE-FO-A Density Comparison:\n' + date_of_data)
    
    initial_ACT = merged_data.loc[merged_data['UTC'].dt.floor('H') == merged_data['UTC'].dt.floor('H').min(), 'ACT_Computed Density'].iloc[0]
    initial_EDR = merged_data.loc[merged_data['UTC'].dt.floor('H') == merged_data['UTC'].dt.floor('H').min(), 'EDR_rolling'].iloc[0]
    initial_POD = POD_and_ACT_data.loc[POD_and_ACT_data['UTC'].dt.floor('H') == POD_and_ACT_data['UTC'].dt.floor('H').min(), 'POD_rolling'].iloc[0]
    initial_JB08 = merged_data.loc[merged_data['UTC'].dt.floor('H') == merged_data['UTC'].dt.floor('H').min(), 'jb08_rho'].iloc[0]

    ax[1].plot(mdates.date2num(merged_data['UTC']), 100 * (merged_data['ACT_Computed Density'] - initial_ACT) / initial_ACT, label='ACT Density Change', color="xkcd:teal", linewidth=1)
    ax[1].plot(mdates.date2num(merged_data['UTC']), 100 * (merged_data['EDR_rolling'] - initial_EDR) / initial_EDR, label='EDR Density Change', color="xkcd:goldenrod", linewidth=1)
    ax[1].plot(mdates.date2num(POD_and_ACT_data['UTC']), 100 * (POD_and_ACT_data['POD_rolling'] - initial_POD) / initial_POD, label='POD Density Change', color="xkcd:purple", linewidth=1)
    ax[1].plot(mdates.date2num(merged_data['UTC']), 100 * (merged_data['jb08_rho'] - initial_JB08) / initial_JB08, label='JB08 Density Change', linestyle='--', color="#FF6347", linewidth=1)
    ax[1].grid(which='both', linestyle='--')
    ax[1].set_ylabel('Relative Change (%)')
    ax[1].set_xlabel('UTC')
    ax[1].set_title('ρᵢ - ρ₀ / ρ₀ (%)')

    #calculate the RMS between the ACT and all the other densities
    rms_ACT_EDR = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - merged_data['EDR_rolling'])**2))
    rms_ACT_POD = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - POD_and_ACT_data['POD_rolling'])**2))
    rms_ACT_JB08 = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - merged_data['jb08_rho'])**2))
    rms_ACT_DTM2000 = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - merged_data['dtm2000_rho'])**2))
    rms_ACT_MSISE00 = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - merged_data['nrlmsise00_rho'])**2))
    print(f"RMS between ACT and EDR: {rms_ACT_EDR}")
    print(f"RMS between ACT and POD: {rms_ACT_POD}")
    print(f"RMS between ACT and JB08: {rms_ACT_JB08}")
    print(f"RMS between ACT and DTM2000: {rms_ACT_DTM2000}")
    print(f"RMS between ACT and MSISE00: {rms_ACT_MSISE00}")

    #now do it for the relative changes
    rms_ACT_EDR = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (merged_data['EDR_rolling'] - initial_EDR))**2))
    rms_ACT_POD = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (POD_and_ACT_data['POD_rolling'] - initial_POD))**2))
    rms_ACT_JB08 = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (merged_data['jb08_rho'] - initial_JB08))**2))
    rms_ACT_DTM2000 = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (merged_data['dtm2000_rho'] - merged_data['dtm2000_rho'].iloc[0]))**2))
    rms_ACT_MSISE00 = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (merged_data['nrlmsise00_rho'] - merged_data['nrlmsise00_rho'].iloc[0]))**2))
    print(f"RMS between ACT and EDR: {rms_ACT_EDR}")
    print(f"RMS between ACT and POD: {rms_ACT_POD}")
    print(f"RMS between ACT and JB08: {rms_ACT_JB08}")
    print(f"RMS between ACT and DTM2000: {rms_ACT_DTM2000}")
    print(f"RMS between ACT and MSISE00: {rms_ACT_MSISE00}")

    ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/ACT_EDR_POD_Comparison_{timenow}.png", dpi=600)

def act_edr_pod_stats(POD_and_ACT_data, EDR_data):
    POD_and_ACT_data['UTC'] = pd.to_datetime(POD_and_ACT_data['UTC'])
    EDR_data['UTC'] = pd.to_datetime(EDR_data['UTC'])
    merged_data = pd.merge(POD_and_ACT_data, EDR_data, on='UTC')
    merged_data['UTC'] = pd.to_datetime(merged_data['UTC'])
    merged_data['EDR_rolling'] = (merged_data['rho_eff']*10).rolling(window=180).mean()
    POD_and_ACT_data['POD_rolling'] = POD_and_ACT_data['Velocity_Computed Density'].rolling(window=90, center=True).mean()

    merged_data = merged_data.iloc[100:]

    median_ACT = merged_data['ACT_Computed Density'].median()
    merged_data['ACT_Computed Density'] = 2 * median_ACT - merged_data['ACT_Computed Density']

    date_of_data = merged_data['UTC'].iloc[0].strftime("%Y-%m-%d")
    merged_data = merged_data[(merged_data['UTC'].dt.hour >= 2) & (merged_data['UTC'].dt.hour <= 18)]
    POD_and_ACT_data = POD_and_ACT_data[(POD_and_ACT_data['UTC'].dt.hour >= 2) & (POD_and_ACT_data['UTC'].dt.hour <= 18)]

    merged_data['ACT_Computed Density'] = -1 * merged_data['ACT_Computed Density']

    initial_ACT = merged_data.loc[merged_data['UTC'].dt.floor('H') == merged_data['UTC'].dt.floor('H').min(), 'ACT_Computed Density'].iloc[0]
    initial_EDR = merged_data.loc[merged_data['UTC'].dt.floor('H') == merged_data['UTC'].dt.floor('H').min(), 'EDR_rolling'].iloc[0]
    initial_POD = POD_and_ACT_data.loc[POD_and_ACT_data['UTC'].dt.floor('H') == POD_and_ACT_data['UTC'].dt.floor('H').min(), 'POD_rolling'].iloc[0]
    initial_JB08 = merged_data.loc[merged_data['UTC'].dt.floor('H') == merged_data['UTC'].dt.floor('H').min(), 'jb08_rho'].iloc[0]

    rms_ACT_EDR_abs = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - merged_data['EDR_rolling'])**2))
    rms_ACT_POD_abs = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - POD_and_ACT_data['POD_rolling'])**2))
    rms_ACT_JB08_abs = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - merged_data['jb08_rho'])**2))
    rms_ACT_DTM2000_abs = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - merged_data['dtm2000_rho'])**2))
    rms_ACT_MSISE00_abs = np.sqrt(np.mean((merged_data['ACT_Computed Density'] - merged_data['nrlmsise00_rho'])**2))

    rms_ACT_EDR_rel = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (merged_data['EDR_rolling'] - initial_EDR))**2)) / initial_ACT * 100
    rms_ACT_POD_rel = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (POD_and_ACT_data['POD_rolling'] - initial_POD))**2)) / initial_ACT * 100
    rms_ACT_JB08_rel = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (merged_data['jb08_rho'] - initial_JB08))**2)) / initial_ACT * 100
    rms_ACT_DTM2000_rel = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (merged_data['dtm2000_rho'] - merged_data['dtm2000_rho'].iloc[0]))**2)) / initial_ACT * 100
    rms_ACT_MSISE00_rel = np.sqrt(np.mean(((merged_data['ACT_Computed Density'] - initial_ACT) - (merged_data['nrlmsise00_rho'] - merged_data['nrlmsise00_rho'].iloc[0]))**2)) / initial_ACT * 100

    rmse_abs = [rms_ACT_EDR_abs, rms_ACT_POD_abs, rms_ACT_JB08_abs, rms_ACT_DTM2000_abs, rms_ACT_MSISE00_abs]
    rmse_rel = [rms_ACT_EDR_rel, rms_ACT_POD_rel, rms_ACT_JB08_rel, rms_ACT_DTM2000_rel, rms_ACT_MSISE00_rel]
    labels = ['EDR', 'POD', 'JB08', 'DTM2000', 'MSISE00']

    fig, ax = plt.subplots(1, 2, figsize=(7, 4))

    bars_abs = ax[0].bar(labels, rmse_abs, color=['xkcd:teal', 'xkcd:goldenrod', '#FF6347', 'xkcd:cherry red', '#1E90FF'])
    ax[0].set_title('Absolute RMSE')
    ax[0].set_ylabel('RMSE (kg/m³)')
    #rotate the x-axis labels by 25 degrees
    plt.setp(ax[0].get_xticklabels(), rotation=25, ha="right")
    #add grid but only horizontal lines and make them dashed
    ax[0].grid(axis='y', linestyle='--')

    for bar in bars_abs:
        yval = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2e}", ha='center', va='bottom')

    bars_rel = ax[1].bar(labels, rmse_rel, color=['xkcd:teal', 'xkcd:goldenrod', '#FF6347', 'xkcd:cherry red', '#1E90FF'])
    ax[1].set_title('Relative RMSE')
    ax[1].set_ylabel('RMSE (%)')
    plt.setp(ax[1].get_xticklabels(), rotation=25, ha="right")

    ax[1].grid(axis='y', linestyle='--')
    plt.tight_layout()
    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/ACT_EDR_POD_RMSE_{timenow}.png", dpi=600)
    plt.show()

def ACT_vs_POD(acc_data_path, quat_data_path, sat_name="GRACE-FO-A", max_time=24, calculate_rho_from_vel=True, models_to_query=['JB08', 'DTM2000', 'NRLMSISE00']):
    force_model_config = {
        '120x120gravity': True,
        '3BP': True,
        'solid_tides': True,
        'ocean_tides': True,
        'knocke_erp': True,
        'relativity': True,
        'SRP': True
    }

    inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)
    inertial_gfo_data['UTC'] = pd.to_datetime(inertial_gfo_data['utc_time'])
    inertial_gfo_data.drop(columns=['utc_time'], inplace=True)

    intertial_t0 = inertial_gfo_data['UTC'].iloc[0]
    inertial_act_gfo_data = inertial_gfo_data[
        (inertial_gfo_data['UTC'] >= intertial_t0) & 
        (inertial_gfo_data['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))
    ]
    
    sp3_ephemeris_df = sp3_ephem_to_df(satellite=sat_name, date=intertial_t0.strftime("%Y-%m-%d"))
    sp3_ephemeris_df['UTC'] = pd.to_datetime(sp3_ephemeris_df['UTC'])
    sp3_ephemeris_df = sp3_ephemeris_df[
        (sp3_ephemeris_df['UTC'] >= intertial_t0) & 
        (sp3_ephemeris_df['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))
    ]

    inertial_act_gfo_ephem = pd.merge(inertial_act_gfo_data, sp3_ephemeris_df, on='UTC', how='inner')
    print(f"head of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.head()}")
    print(f"columns of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.columns}")

    act_x_acc_col, act_y_acc_col, act_z_acc_col = 'inertial_x_acc', 'inertial_y_acc', 'inertial_z_acc'
    rp_fm_config = {
    'knocke_erp': True,
    'SRP': True}
    #this rp fm config also appears in density inveersion function
    #TODO: just keep one pls.
    rho_from_ACT = density_inversion(
        sat_name, inertial_act_gfo_ephem, 
        act_x_acc_col, act_y_acc_col, act_z_acc_col, 
        rp_fm_config, nc_accs=True, 
        models_to_query=models_to_query, density_freq='15S'
    )
    rho_from_ACT.rename(columns={'Computed Density': 'ACT_Computed Density'}, inplace=True)

    print(f"head of rho_from_ACT: {rho_from_ACT.head()}")

    if calculate_rho_from_vel:
        interp_ephemeris_df = interpolate_positions(sp3_ephemeris_df, '0.01S')
        sp3_velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
        sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z = 'vel_acc_x', 'vel_acc_y', 'vel_acc_z'
        rho_from_vel = density_inversion(
            sat_name, sp3_velacc_ephem, 
            sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z, 
            force_model_config=force_model_config, nc_accs=False, 
            models_to_query=[None], density_freq='15S'
        )
        rho_from_vel.rename(columns={'Computed Density': 'Velocity_Computed Density'}, inplace=True)

        print(f"head of rho_from_vel: {rho_from_vel.head()}")

        merged_df = pd.merge(
            rho_from_ACT[['UTC', 'ACT_Computed Density'] + models_to_query], 
            rho_from_vel[['UTC', 'Velocity_Computed Density']], 
            on='UTC', 
            how='inner'
        )
    else:
        merged_df = rho_from_ACT[['UTC', 'ACT_Computed Density'] + models_to_query]

    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    merged_df.dropna(inplace=True)

    print(f"head of merged_df: {merged_df.head()}")
    print(f"columns of merged_df: {merged_df.columns}")
    merged_df.to_csv(f"output/DensityInversion/PODBasedAccelerometry/Data/{sat_name}/Accelerometer_benchmark/{timenow}_bench.csv", index=False)

if __name__ == '__main__':

    # File paths
    # acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-06_C_04.txt"
    # quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-06_C_04.txt"

    # # Call the function with appropriate parameters
    # ACT_vs_POD(
    #     acc_data_path=acc_data_path,
    #     quat_data_path=quat_data_path,
    #     sat_name="GRACE-FO-A",
    #     max_time=24,
    #     calculate_rho_from_vel=True,
    #     models_to_query=['JB08', 'DTM2000', 'NRLMSISE00'])

    POD_and_ACT_data = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/2024-05-30_11-56-15_bench.csv")
    EDR_data = pd.read_csv("output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/EDR_GRACE-FO-A__2023-05-05 18:00:12_2023-05-06 18:00:12_2024-05-13.csv")

    # ACT_vs_EDR_vs_POD_plot(POD_and_ACT_data, EDR_data)
    act_edr_pod_stats(POD_and_ACT_data, EDR_data)


#     sat_name = "GRACE-FO-A"
#     force_model_config = {'120x120gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
#     max_time = 24
#     acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-06_C_04.txt"
#     quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-06_C_04.txt"
#     inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)
#     inertial_gfo_data['UTC'] = pd.to_datetime(inertial_gfo_data['utc_time'])
#     inertial_gfo_data.drop(columns=['utc_time'], inplace=True)

#     intertial_t0 = inertial_gfo_data['UTC'].iloc[0]
#     inertial_act_gfo_data = inertial_gfo_data[(inertial_gfo_data['UTC'] >= intertial_t0) & (inertial_gfo_data['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))]
#     sp3_ephemeris_df = sp3_ephem_to_df(satellite="GRACE-FO-A", date="2023-05-06")
#     sp3_ephemeris_df['UTC'] = pd.to_datetime(sp3_ephemeris_df['UTC'])
#     sp3_ephemeris_df = sp3_ephemeris_df[(sp3_ephemeris_df['UTC'] >= intertial_t0) & (sp3_ephemeris_df['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))]

#     inertial_act_gfo_ephem = pd.merge(inertial_act_gfo_data, sp3_ephemeris_df, on='UTC', how='inner')
#     print(f"head of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.head()}")
#     print(f"columns of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.columns}")

#     act_x_acc_col, act_y_acc_col, act_z_acc_col = 'inertial_x_acc', 'inertial_y_acc', 'inertial_z_acc'
#     rho_from_ACT = density_inversion(sat_name, inertial_act_gfo_ephem, 
#                                      act_x_acc_col, act_y_acc_col, act_z_acc_col, 
#                                      force_model_config, nc_accs=True, 
#                                      models_to_query=['JB08', "DTM2000", "NRLMSISE00"], density_freq='15S')
#     rho_from_ACT.rename(columns={'Computed Density': 'ACT_Computed Density'}, inplace=True)

#     print(f"head of rho_from_ACT: {rho_from_ACT.head()}")

#     interp_ephemeris_df = interpolate_positions(sp3_ephemeris_df, '0.01S')
#     sp3_velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
#     sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z = 'vel_acc_x', 'vel_acc_y', 'vel_acc_z'
#     rho_from_vel = density_inversion(sat_name, sp3_velacc_ephem, 
#                                     sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z, 
#                                     force_model_config=force_model_config, nc_accs=False, 
#                                     models_to_query=[None], density_freq='15S')
#     rho_from_vel.rename(columns={'Computed Density': 'Velocity_Computed Density'}, inplace=True)

#     print(f"head of rho_from_vel: {rho_from_vel.head()}")

#     merged_df = pd.merge(rho_from_ACT[['UTC', 'ACT_Computed Density', 'JB08', 'DTM2000', 'NRLMSISE00']], rho_from_vel[['UTC', 'Velocity_Computed Density']], on='UTC', how='inner')

#     timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     merged_df.dropna(inplace=True)

#     print(f"head of merged_df: {merged_df.head()}")
#     print(f"columns of merged_df: {merged_df.columns}")
#     merged_df.to_csv(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/{timenow}_bench.csv", index=False)

    #load the data from path 
    # path = "output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/2024-05-09_10-47-36_bench.csv"
    # merged_df = pd.read_csv(path)
    # merged_df = merged_df.iloc[10:-10]
    # median_ACT = merged_df['ACT_Computed Density'].median()
    # flipped_ACT = 2 * median_ACT - merged_df['ACT_Computed Density'] #TODO: figure out why the accelerometer density is inverted? Is the X/along track sign different in the ACT file?
    # rolling_av45 = merged_df['Velocity_Computed Density'].rolling(window=90, center=True).mean()
    # plt.plot(merged_df['UTC'], flipped_ACT, label='ACT Density')
    # # plt.plot(merged_df['UTC'], merged_df['Velocity_Computed Density'], label='Velocity Computed Density')
    # plt.plot(merged_df['UTC'], rolling_av45, label='POD Density', linewidth=2)
    # plt.plot(merged_df['UTC'], merged_df['JB08'], label='JB08 Density', linestyle='--')
    # plt.plot(merged_df['UTC'], merged_df['DTM2000'], label='DTM2000 Density', linestyle='--')
    # plt.plot(merged_df['UTC'], merged_df['NRLMSISE00'], label='MSISE00 Density', linestyle='--')
    # plt.xlabel('Time')
    # plt.ylabel('Density (kg/m^3)')
    # #log scale
    # #only display every hour on the x-axis (the points come every 30s)
    # plt.xticks(merged_df['UTC'][::120], rotation=45)
    # plt.yscale('log')
    # plt.legend()
    # plt.tight_layout()
    # plt.grid(which='both', linestyle='--')
    # plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/{timenow}_bench.png")
    # # plt.show()
    # plt.close()

    # valid_start = 45  # half the window size - 1
    # valid_end = -45   # negative to drop the tail end

    # # Compute rolling average again to ensure consistency
    # rolling_av45 = merged_df['Velocity_Computed Density'].rolling(window=90, center=True).mean()

    # # Use valid data avoiding NaNs from rolling mean
    # valid_data = merged_df.iloc[valid_start:valid_end]
    # valid_rolling_av45 = rolling_av45.iloc[valid_start:valid_end]

    # # Baselines based on the first valid value after removing NaNs from rolling average
    # baseline_ACT = valid_data['ACT_Computed Density'].iloc[0]
    # baseline_Velocity = valid_rolling_av45.iloc[0]
    # baseline_JB08 = valid_data['JB08'].iloc[0]
    # baseline_DTM2000 = valid_data['DTM2000'].iloc[0]
    # baseline_MSISE00 = valid_data['NRLMSISE00'].iloc[0]

    # # Compute relative values against their respective baselines
    # relative_flipped_ACT = 2 * baseline_ACT - valid_data['ACT_Computed Density'] - baseline_ACT
    # relative_rolling_av45 = valid_rolling_av45 - baseline_Velocity
    # relative_JB08 = valid_data['JB08'] - baseline_JB08
    # relative_DTM2000 = valid_data['DTM2000'] - baseline_DTM2000
    # relative_MSISE00 = valid_data['NRLMSISE00'] - baseline_MSISE00

    # # Plotting
    # plt.plot(valid_data['UTC'], relative_flipped_ACT, label='Relative ACT Computed Density')
    # plt.plot(valid_data['UTC'], relative_rolling_av45, label='Relative Velocity Computed (MA)', linewidth=2)
    # plt.plot(valid_data['UTC'], relative_JB08, label='Relative JB08 Density', linestyle='--')
    # plt.plot(valid_data['UTC'], relative_DTM2000, label='Relative DTM2000 Density', linestyle='--')
    # plt.plot(valid_data['UTC'], relative_MSISE00, label='Relative MSISE00 Density', linestyle='--')

    # plt.xlabel('Time')
    # plt.ylabel('Relative Density Change (kg/m^3)')
    # plt.xticks(valid_data['UTC'][::120], rotation=45)
    # plt.yscale('linear')
    # plt.legend()
    # plt.tight_layout()
    # plt.grid(which='both', linestyle='--')
    # plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/{timenow}_rel_self_bench.png")
    # # plt.show()
    # plt.close()

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.signal import welch

    # # Given sampling rate from your setup
    # sampling_rate = 1 / 30  # seconds^-1

    # # Compute the power spectral densities for each signal
    # # Frequency and power spectral density for flipped ACT
    # f_ACT, Pxx_ACT = welch(flipped_ACT, fs=sampling_rate, nperseg=1024)

    # # Frequency and power spectral density for velocity computed (rolling average)
    # rolling_av45 = rolling_av45[~np.isnan(rolling_av45)]  # Removing NaN values from rolling_av45
    # f_Velocity, Pxx_Velocity = welch(rolling_av45, fs=sampling_rate, nperseg=1024)

    # # Frequency and power spectral density for JB08
    # f_JB08, Pxx_JB08 = welch(merged_df['JB08'], fs=sampling_rate, nperseg=1024)
    # f_dtm2000, Pxx_dtm2000 = welch(merged_df['DTM2000'], fs=sampling_rate, nperseg=1024)
    # f_msise00, Pxx_msise00 = welch(merged_df['NRLMSISE00'], fs=sampling_rate, nperseg=1024)

    # # Convert frequency from Hz to cycles per minute (cpm)
    # f_ACT_minutes = f_ACT * 60
    # f_Velocity_minutes = f_Velocity * 60
    # f_JB08_minutes = f_JB08 * 60
    # f_dtm2000_minutes = f_dtm2000 * 60
    # f_msise00_minutes = f_msise00 * 60

    # # 90 minutes mark in cycles per minute
    # one_orbit = 1 / 97
    # one_half_orbit = 1 / 48.5
    # one_third_orbit = 1/32.33

    # # Plotting the PSDs on a log-log scale
    # plt.figure(figsize=(8, 4))

    # # ACT Computed Density
    # plt.loglog(f_ACT_minutes, Pxx_ACT, label='ACT Computed Density', linewidth=1, color='xkcd:jade')

    # # Velocity Computed Density
    # plt.loglog(f_Velocity_minutes, Pxx_Velocity, label='Velocity Computed Density', linewidth=1, color='xkcd:azure')

    # # JB08 Density
    # plt.loglog(f_JB08_minutes, Pxx_JB08, label='JB08 Density', linestyle='--', linewidth=1, color='xkcd:coral')
    # plt.loglog(f_dtm2000_minutes, Pxx_dtm2000, label='DTM2000 Density', linestyle='--', linewidth=1, color='xkcd:teal')
    # plt.loglog(f_msise00_minutes, Pxx_msise00, label='MSISE00 Density', linestyle='--', linewidth=1, color='xkcd:purple')

    # # Convert x-axis label to cycles per minute
    # plt.xlabel('Frequency')
    # plt.ylabel('PSD (kg^2/m^6 Hz)')

    # # Add a vertical line at the 90-minute mark
    # plt.axvline(x=one_orbit, color='xkcd:mustard', linewidth=4, label='~1 orbit', alpha=0.5)
    # plt.axvline(x=one_half_orbit, color='xkcd:purple', linewidth=4, label='~1/2 orbit', alpha=0.5)
    # plt.axvline(x=one_third_orbit, color='xkcd:cyan', linewidth=4, label='~1/3 orbit', alpha=0.5)

    # plt.title('Power Spectral Density of the Density Signals')
    # plt.legend()
    # plt.grid(True, which='both', ls='-', linewidth=0.5)
    # plt.tight_layout()

    # # Save the figure
    # plt.savefig(f"output/DensityInversion/PODBasedAccelerometry/Plots/GRACE-FO-A/Accelerometer_benchmark/{timenow}_PSD_minutes.png")
    # # plt.show()
    # plt.close()


