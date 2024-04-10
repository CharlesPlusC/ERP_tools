import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants

import os
from tools.utilities import extract_acceleration, utc_to_mjd, HCL_diff, get_satellite_info, pos_vel_from_orekit_ephem, EME2000_to_ITRF, ecef_to_lla, posvel_to_sma
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import state2acceleration, query_jb08
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy.special import lpmn
from scipy.integrate import trapz
from scipy.interpolate import UnivariateSpline

mu = Constants.WGS84_EARTH_MU
R_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
J2 =Constants.EGM96_EARTH_C20

def interpolate_ephemeris(df, start_time, end_time, freq='0.0001S', stitch=False):
    # Ensure UTC is the index and not duplicated
    df = df.drop_duplicates(subset='UTC').set_index('UTC')
    # Sort by UTC
    df = df.sort_index()
    # Create a new DataFrame with resampled frequency between the specified start and end times
    df_resampled = df.reindex(pd.date_range(start=start_time, end=end_time, freq=freq), method='nearest').asfreq(freq)
    # Interpolate values using a spline method
    interp_funcs = {col: UnivariateSpline(df.index.astype(int), df[col], s=0, ext='extrapolate') for col in ['x', 'y', 'z', 'xv', 'yv', 'zv']}
    for col in ['x', 'y', 'z', 'xv', 'yv', 'zv']:
        df_resampled[col] = interp_funcs[col](df_resampled.index.astype(int))
    # Filter out the part of the resampled DataFrame within the start and end time
    df_filtered = df_resampled.loc[start_time:end_time].reset_index().rename(columns={'index': 'UTC'})
    if stitch:
        # Concatenate the original DataFrame with the filtered resampled DataFrame
        df_stitched = pd.concat([
            df.loc[:start_time - pd.Timedelta(freq), :],  # Data before the interpolation interval
            df_filtered.set_index('UTC'),
            df.loc[end_time + pd.Timedelta(freq):, :]    # Data after the interpolation interval
        ]).reset_index()
        return df_stitched
    return df_filtered

def U_J2(r, phi, lambda_):
    theta = np.pi / 2 - phi  # Convert latitude to colatitude
    # J2 potential term calculation, excluding the monopole term
    V_j2 = -mu / r * J2 * (R_earth / r)**2 * (3 * np.cos(theta)**2 - 1) / 2
    return V_j2

def associated_legendre_polynomials(degree, order, x):
    # Precompute all needed Legendre polynomials at once
    P = {}
    for n in range(degree + 1):
        for m in range(min(n, order) + 1):
            P_nm, _ = lpmn(m, n, x)
            P[(n, m)] = P_nm[0][-1]
    return P

def compute_gravitational_potential(r, phi, lambda_, degree, order, date):
    provider = GravityFieldFactory.getUnnormalizedProvider(degree, order)
    harmonics = provider.onDate(date)
    V_dev = np.zeros_like(phi)
    
    mu_over_r = mu / r
    r_ratio = R_earth / r
    sin_phi = np.sin(phi)

    # Initialize coefficient arrays with zeros
    max_order = max(order, degree) + 1
    C_nm = np.zeros((degree + 1, max_order))
    S_nm = np.zeros((degree + 1, max_order))

    # Fill in the coefficient arrays
    for n in range(degree + 1):
        for m in range(min(n, order) + 1):
            C_nm[n, m] = harmonics.getUnnormalizedCnm(n, m)
            S_nm[n, m] = harmonics.getUnnormalizedSnm(n, m)

    # Compute all necessary Legendre polynomials once
    P = associated_legendre_polynomials(degree, order, sin_phi)

    for n in range(1, degree + 1):
        r_ratio_n = r_ratio ** n
        for m in range(min(n, order) + 1):
            P_nm = P[(n, m)]
            sectorial_term = P_nm * (C_nm[n, m] * np.cos(m * lambda_) + S_nm[n, m] * np.sin(m * lambda_))
            V_dev += mu_over_r * r_ratio_n * sectorial_term

    return -V_dev

def compute_rho_eff(EDR, velocity, CD, A_ref, mass, MJDs):
    # Calculate the integral part for rho_eff computation
    time_diffs = np.diff(MJDs) * 86400  # Convert MJDs to seconds
    rho_eff = np.zeros(len(velocity))
    for i in range(1, len(velocity)):
        # Use the time differences for the trapezoidal integration
        # We multiply velocity^3 by the time interval (in seconds) to get the integral value
        if i < len(time_diffs):  # Ensure we don't go out of bounds
            integral_value = trapz(-0.5 * CD * A_ref * velocity[i-1:i+1]**3, dx=time_diffs[i-1])
            rho_eff[i] = EDR[i] / (integral_value * mass)
    
    return rho_eff

folder_save = "output/DensityInversion/OrbitEnergy"

def main():
    # sat_names_to_test = ["GRACE-FO-B", "TerraSAR-X", "TanDEM-X", "CHAMP"]
    sat_names_to_test = ["GRACE-FO-A"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        print(f'length of ephemeris_df: {len(ephemeris_df)}')
        # ephemeris_df = ephemeris_df.head(250)
        # take the UTC column and convert to mjd
        ephemeris_df['MJD'] = [utc_to_mjd(dt) for dt in ephemeris_df['UTC']]
        # start_date_utc = ephemeris_df['UTC'].iloc[0]
        # end_date_utc = ephemeris_df['UTC'].iloc[-1]
        # file_name = os.path.join(folder_save, f"{sat_name}_energy_components_{start_date_utc}_{end_date_utc}.csv")
        
        # if os.path.exists(file_name):
        #     print(f"Fetching data from {file_name}")
        #     ephemeris_df = pd.read_csv(file_name)
        # else:
        #     print(f"Computing data for {sat_name}")
        #     x_ecef, y_ecef, z_ecef, xv_ecef, yv_ecef, zv_ecef = ([] for _ in range(6))
        #     # Convert EME2000 to ITRF for each point
        #     for _, row in ephemeris_df.iterrows():
        #         pos = [row['x'], row['y'], row['z']]
        #         vel = [row['xv'], row['yv'], row['zv']]
        #         mjd = [row['MJD']]
        #         itrs_pos, itrs_vel = EME2000_to_ITRF(np.array([pos]), np.array([vel]), pd.Series(mjd))
        #         # Append the converted coordinates to their respective lists
        #         x_ecef.append(itrs_pos[0][0])
        #         y_ecef.append(itrs_pos[0][1])
        #         z_ecef.append(itrs_pos[0][2])
        #         xv_ecef.append(itrs_vel[0][0])
        #         yv_ecef.append(itrs_vel[0][1])
        #         zv_ecef.append(itrs_vel[0][2])
        #     # Assign the converted coordinates back to the dataframe
        #     ephemeris_df['x_ecef'] = x_ecef
        #     ephemeris_df['y_ecef'] = y_ecef
        #     ephemeris_df['z_ecef'] = z_ecef
        #     ephemeris_df['xv_ecef'] = xv_ecef
        #     ephemeris_df['yv_ecef'] = yv_ecef
        #     ephemeris_df['zv_ecef'] = zv_ecef

        #     #conver the ecef coordinates to lla
        #     converted_coords = np.vectorize(ecef_to_lla, signature='(),(),()->(),(),()')(ephemeris_df['x_ecef'], ephemeris_df['y_ecef'], ephemeris_df['z_ecef'])
        #     ephemeris_df[['lat', 'lon', 'alt']] = np.column_stack(converted_coords)

        #     kinetic_energy = (np.linalg.norm([ephemeris_df['xv'], ephemeris_df['yv'], ephemeris_df['zv']], axis=0))**2/2
        #     print(f"first five values of kinetic energy: {kinetic_energy[:5]}")
        #     ephemeris_df['kinetic_energy'] = kinetic_energy

        #     monopole_potential = mu / np.linalg.norm([ephemeris_df['x'], ephemeris_df['y'], ephemeris_df['z']], axis=0)
        #     print(f"first five values of monopole potential: {monopole_potential[:5]}")
        #     ephemeris_df['monopole'] = monopole_potential

        #     U_non_sphericals = []
        #     U_j2s = []
        #     for _, row in ephemeris_df.iterrows():
        #         print(f"computing potential for {row['lat']}, {row['lon']}, {row['alt']}")
        #         phi_rad = np.radians(row['lat'])
        #         lambda_rad = np.radians(row['lon'])
        #         r = row['alt']
        #         degree = 120
        #         order = 120
        #         date = datetime_to_absolutedate(row['UTC'])
        #         U_non_spher =  compute_gravitational_potential(r, phi_rad, lambda_rad, degree, order, date)
        #         U_j2 = U_J2(r, phi_rad, lambda_rad)
        #         U_j2s.append(U_j2)
        #         U_non_sphericals.append(U_non_spher)

        #     #total energies
        #     energy_total_spherical = kinetic_energy  - monopole_potential
        #     energy_total_J2 = kinetic_energy  - monopole_potential + U_j2s
        #     energy_total_HOT = kinetic_energy  - monopole_potential + U_non_sphericals

        #     #difference between total energies
        #     spherical_total_diff = energy_total_spherical[0] - energy_total_spherical
        #     j2_total_diff = energy_total_J2[0] - energy_total_J2
        #     HOT_total_diff = energy_total_HOT[0] - energy_total_HOT
            
        #     #individual energy components diff
        #     kinetic_energy_diff = kinetic_energy[0] - kinetic_energy
        #     monopole_potential_diff = monopole_potential[0] - monopole_potential
        #     j2_diff = U_j2s[0] - U_j2s
        #     HOT_diff = U_non_sphericals[0] - U_non_sphericals
            
        #     #in a dataframe save , x,y,z,u,v,w, lat, lon, alt, kinetic_energy, monopole, U_j2, U_non_spherical, j2_total_diff, HOT_total_diff
        #     ephemeris_df['U_j2'] = U_j2s
        #     ephemeris_df['U_non_spherical'] = U_non_sphericals
        #     ephemeris_df['j2_total_diff'] = j2_total_diff
        #     ephemeris_df['HOT_total_diff'] = HOT_total_diff
        #     export_df = ephemeris_df[['MJD','x', 'y', 'z', 'xv', 'yv', 'zv', 'lat', 'lon', 'alt', 'kinetic_energy', 'monopole', 'U_j2', 'U_non_spherical', 'j2_total_diff', 'HOT_total_diff']]
        #     export_df.to_csv(os.path.join(folder_save, f"{sat_name}_energy_components_{start_date_utc}_{end_date_utc}.csv"), index=False)

        # #if the ephemeris does not contain a columns called MJD, then use start_date_utc and assume 30s time steps
        # if 'MJD' not in ephemeris_df.columns:
        #     ephemeris_df['MJD'] = [utc_to_mjd(start_date_utc) + 30/86400 * i for i in range(len(ephemeris_df))]
        # EDR = -ephemeris_df['HOT_total_diff']
        # #now get the 180-point rolling average of the EDR
        # EDR_180 = EDR.rolling(window=180, min_periods=1).mean()
        # velocity = np.linalg.norm([ephemeris_df['xv'], ephemeris_df['yv'], ephemeris_df['zv']], axis=0)
        # time_steps = ephemeris_df['MJD']
        # CD = 3.2  # From Mehta et al 2013
        # A_ref = 1.004  # From Mehta et al 2013
        mass = 600.0
        # rho_eff = compute_rho_eff(EDR, velocity, CD, A_ref, mass, time_steps)
        # rho_eff_180 = compute_rho_eff(EDR_180, velocity, CD, A_ref, mass, time_steps)

        # x = ephemeris_df['x']
        # y = ephemeris_df['y']
        # z = ephemeris_df['z']
        # u = ephemeris_df['xv']
        # v = ephemeris_df['yv']
        # w = ephemeris_df['zv']
        # #use posvel_to_sma to get the semi-major axis for each point
        # print(f"x: {x[:5]}, y: {y[:5]}, z: {z[:5]}, u: {u[:5]}, v: {v[:5]}, w: {w[:5]}")
        # for i in range(len(x)):
        #     sma = posvel_to_sma(x[i], y[i], z[i], u[i], v[i], w[i])
        #     ephemeris_df.at[i, 'sma'] = sma

        # smas = ephemeris_df['sma']

        # #get the rolling average of the sma
        # smas_180 = smas.rolling(window=180, min_periods=1).mean()

        # #slice the rolling sma and rho_eff by 180 points at the start and at the finish
        # smas_180 = smas_180[180:-180]
        # rho_eff_180 = rho_eff_180[180:-180]
        # #slice the MJDs to be the same length as the rolling sma and rho_eff
        # MJDs = ephemeris_df['MJD'][180:-180]


        # fig, ax1 = plt.subplots()
        # ax1.plot(MJDs, smas_180, 'b-')
        # ax1.set_xlabel('Modified Julian Date')
        # ax1.set_ylabel('SMA', color='b')
        # ax1.tick_params('y', colors='b')

        # ax2 = ax1.twinx()
        # ax2.plot(MJDs, rho_eff_180, 'r-')
        # ax2.set_ylabel('Effective Density (kg/m^3)', color='r')
        # ax2.tick_params('y', colors='r')
        # ax2.set_title(f"{sat_name}: Effective Density")

        # ax2.plot(ephemeris_df['MJD'], np.linalg.norm([ephemeris_df['x'], ephemeris_df['y'], ephemeris_df['z']], axis=0), 'r-')
        # ax2.set_ylabel('r (m)', color='r')
        # ax2.tick_params('y', colors='r')

        # fig.tight_layout()
        # plt.title(f"{sat_name}: Effective Density and SMA")
        # plt.show()

        # plt.plot(ephemeris_df['MJD'], rho_eff)
        # plt.xlabel('Modified Julian Date')
        # plt.ylabel('Effective Density (kg/m^3)')
        # plt.title(f"{sat_name}: Effective Density")
        # plt.grid(True)
        # plt.show()

        force_model_config = {
            '120x120gravity': True, '3BP': True, 'solid_tides': True, 
            'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True
        }
        cr = 1.5
        cd = 3.2
        cross_section = 1.004
        mass = 600.0

        # Maintain original interpolation at 0.01-second intervals
        # ephemeris_df = ephemeris_df.head(no_points_to_process)
        interp_ephemeris_df = interpolate_ephemeris(ephemeris_df, ephemeris_df['UTC'].iloc[0], ephemeris_df['UTC'].iloc[-1], freq='0.01S')

        o_minus_cs = []
        v_components_of_o_minus_cs = []
        computed_rhos = []
        jb08_rhos = []

        # Calculate V2 - V1 at every 0.01-second interval but only perform this calculation every 30 seconds
        for i in range(3000, len(interp_ephemeris_df), 3000):  # Start at 30 seconds and step by 30 seconds
            print(f"Processing {i} of {len(interp_ephemeris_df)}")
            # Use i for V2 (current) and i-1 for V1 (previous) to calculate the difference at 0.01s interval
            v1 = np.array([interp_ephemeris_df['xv'][i - 1], interp_ephemeris_df['yv'][i - 1], interp_ephemeris_df['zv'][i - 1]])
            v2 = np.array([interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]])
            t1 = pd.to_datetime(interp_ephemeris_df['UTC'][i - 1])
            t2 = pd.to_datetime(interp_ephemeris_df['UTC'][i])
            delta_t = (t2 - t1).total_seconds()  # This should be approximately 0.01 seconds
            print(f"Delta T: {delta_t}")
            observed_accelerations = (v2 - v1) / delta_t
            print(f"Observed Accelerations: {observed_accelerations}")

            # Now compute the accelerations and other parameters using the current timestep 'i'
            state_vector = np.array([interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i], v2[0], v2[1], v2[2]])
            computed_accelerations_dict = state2acceleration(state_vector, t2, cr, cd, cross_section, mass, **force_model_config)
            computed_accelerations_sum = np.sum(list(computed_accelerations_dict.values()), axis=0)
            print(f"Computed Accelerations: {computed_accelerations_sum}")

            a_aero = computed_accelerations_sum - observed_accelerations
            o_minus_cs.append(a_aero)
            r = np.array((interp_ephemeris_df['x'][i], interp_ephemeris_df['y'][i], interp_ephemeris_df['z'][i]))
            v = np.array((interp_ephemeris_df['xv'][i], interp_ephemeris_df['yv'][i], interp_ephemeris_df['zv'][i]))
            atm_rot = np.array([0, 0, 72.9211e-6])
            v_rel = v - np.cross(atm_rot, r)
            unit_v_rel = v_rel / np.linalg.norm(v_rel)
            
            a_aero_dotv = np.dot(a_aero, unit_v_rel)
            v_components_of_o_minus_cs.append(a_aero_dotv)
            rho = -2 * (a_aero_dotv / (cd * cross_section)) * (mass / np.abs(np.linalg.norm(v_rel)**2))
            rho /= 1000  # Convert to kg/m^3
            computed_rhos.append(rho)
            print(f"Computed Density: {rho}")
            jb08_rho = query_jb08(r, t2)
            print(f"JB08 Density: {jb08_rho}")
            jb08_rhos.append(jb08_rho)

        #plot 
        utc_for_plotting = interp_ephemeris_df['UTC'][1:len(v_components_of_o_minus_cs) + 1]

        plt.plot( v_components_of_o_minus_cs)
        plt.xlabel('Modified Julian Date')
        plt.ylabel('v_components_of_o_minus_cs (m/s^2)')
        plt.title(f"{sat_name}: v_components_of_o_minus_cs")
        plt.grid(True)
        plt.show()

        #plot rho and jb08_rho 
        plt.plot(utc_for_plotting, computed_rhos, label='Computed Density')
        plt.plot(utc_for_plotting, jb08_rhos, label='JB08 Density')
        plt.xlabel('Modified Julian Date')
        plt.ylabel('Density (kg/m^3)')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()

#### PLOTTING ####
        # #make 3 subplots
        # #set sns style
        # sns.set_theme(style='whitegrid')
        # fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        # #plot total energy differences
        # axs[0].plot(ephemeris_df['MJD'], spherical_total_diff, label='Spherical Total Energy Difference', color='xkcd:hot magenta')
        # axs[0].plot(ephemeris_df['MJD'], j2_total_diff, label='J2 Total Energy Difference', color='xkcd:tangerine')
        # axs[0].plot(ephemeris_df['MJD'], HOT_total_diff, label='HOT Total Energy Difference', color='xkcd:greenish', linestyle='--')
        # axs[0].grid(True)
        # axs[0].legend()
        # axs[0].set_xlabel('')
        # axs[0].set_ylabel('Energy_i - Energy_0 (J/kg)')
        # axs[0].set_title('Total Relative Energy Differences')

        # #plot same as the top but only J2 and HOT
        # axs[1].plot(ephemeris_df['MJD'], j2_total_diff, label='J2 Total Energy Difference', color='xkcd:tangerine')
        # axs[1].plot(ephemeris_df['MJD'], HOT_total_diff, label='HOT Total Energy Difference', color='xkcd:greenish', linestyle='--')
        # axs[1].legend()
        # axs[1].grid(True)
        # axs[1].set_xlabel('')
        # axs[1].set_ylabel('Energy_i - Energy_0 (J/kg)')
        # axs[1].set_title('Total Relative Energy Differences (J2 and H.O.T only)')
        
        # #plot individual energy differences
        # axs[2].plot(ephemeris_df['MJD'], kinetic_energy_diff, label='Kinetic Energy Difference', color='xkcd:minty green')
        # axs[2].plot(ephemeris_df['MJD'], monopole_potential_diff, label='Monopole Potential Difference', color='xkcd:easter purple')
        # axs[2].plot(ephemeris_df['MJD'], j2_diff, label='J2 Potential Difference', color='xkcd:tangerine')
        # axs[2].plot(ephemeris_df['MJD'], HOT_diff, label='HOT Potential Difference', color='xkcd:greenish', linestyle='--')
        # axs[2].legend()
        # axs[2].grid(True)
        # axs[2].set_xlabel('Modified Julian Date')
        # axs[2].set_ylabel('Energy_i - Energy_0 (J/kg)')
        # axs[2].set_title('Individual Relative Energy Differences')

        # #main title with satellite name
        # fig.suptitle(f"{sat_name}: Energy Budget", fontsize=14)

        # plt.tight_layout()

        # plt.savefig(os.path.join(folder_save, f"{sat_name}_energy_components_{start_date_utc}_{end_date_utc}.png"))

        # import cartopy.crs as ccrs
        # import cartopy.feature as cfeature

        # # Plot a lat-lon map with landmasses outlined and the rest in light grey
        # fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        # ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey')
        # ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
        # ax.add_feature(cfeature.COASTLINE)
        # ax.add_feature(cfeature.BORDERS, linestyle=':')

        # # Ensure the colormap is centered at zero
        # max_abs_val = np.max(np.abs(HOT_total_diff))
        # norm = plt.Normalize(-max_abs_val, max_abs_val)

        # # Scatter plot for HOT total energy differences with 'seismic' colormap
        # sc = ax.scatter(ephemeris_df['lon'], ephemeris_df['lat'], c=HOT_total_diff, cmap='seismic', norm=norm, s=5, transform=ccrs.PlateCarree())

        # ax.set_title(f"{sat_name}: HOT Total Energy Differences")
        # cbar = plt.colorbar(sc)
        # cbar.set_label('Energy_i - Energy_0 (J/kg)')

        # plt.savefig(os.path.join(folder_save, f"{sat_name}_HOT_total_energy_diff_map_{start_date_utc}_{end_date_utc}.png"))
        # plt.show()

        # # plt.show()