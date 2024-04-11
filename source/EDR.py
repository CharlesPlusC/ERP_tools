import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants

import os
from tools.utilities import extract_acceleration, utc_to_mjd, HCL_diff, get_satellite_info, pos_vel_from_orekit_ephem, EME2000_to_ITRF, ecef_to_lla, posvel_to_sma, interpolate_ephemeris
from tools.sp3_2_ephemeris import sp3_ephem_to_df
from tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy.special import lpmn
from scipy.integrate import trapz

mu = Constants.WGS84_EARTH_MU
R_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
J2 =Constants.EGM96_EARTH_C20


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
        start_date_utc = ephemeris_df['UTC'].iloc[0]
        end_date_utc = ephemeris_df['UTC'].iloc[-1]
        file_name = os.path.join(folder_save, f"{sat_name}_energy_components_{start_date_utc}_{end_date_utc}.csv")
        
        if os.path.exists(file_name):
            print(f"Fetching data from {file_name}")
            energy_ephemeris_df = pd.read_csv(file_name)
        else:
            print(f"Computing data for {sat_name}")
            x_ecef, y_ecef, z_ecef, xv_ecef, yv_ecef, zv_ecef = ([] for _ in range(6))
            # Convert EME2000 to ITRF for each point
            for _, row in energy_ephemeris_df.iterrows():
                pos = [row['x'], row['y'], row['z']]
                vel = [row['xv'], row['yv'], row['zv']]
                mjd = [row['MJD']]
                itrs_pos, itrs_vel = EME2000_to_ITRF(np.array([pos]), np.array([vel]), pd.Series(mjd))
                # Append the converted coordinates to their respective lists
                x_ecef.append(itrs_pos[0][0])
                y_ecef.append(itrs_pos[0][1])
                z_ecef.append(itrs_pos[0][2])
                xv_ecef.append(itrs_vel[0][0])
                yv_ecef.append(itrs_vel[0][1])
                zv_ecef.append(itrs_vel[0][2])
            # Assign the converted coordinates back to the dataframe
            energy_ephemeris_df['x_ecef'] = x_ecef
            energy_ephemeris_df['y_ecef'] = y_ecef
            energy_ephemeris_df['z_ecef'] = z_ecef
            energy_ephemeris_df['xv_ecef'] = xv_ecef
            energy_ephemeris_df['yv_ecef'] = yv_ecef
            energy_ephemeris_df['zv_ecef'] = zv_ecef

            #conver the ecef coordinates to lla
            converted_coords = np.vectorize(ecef_to_lla, signature='(),(),()->(),(),()')(energy_ephemeris_df['x_ecef'], energy_ephemeris_df['y_ecef'], energy_ephemeris_df['z_ecef'])
            energy_ephemeris_df[['lat', 'lon', 'alt']] = np.column_stack(converted_coords)

            kinetic_energy = (np.linalg.norm([energy_ephemeris_df['xv'], energy_ephemeris_df['yv'], energy_ephemeris_df['zv']], axis=0))**2/2
            print(f"first five values of kinetic energy: {kinetic_energy[:5]}")
            energy_ephemeris_df['kinetic_energy'] = kinetic_energy

            monopole_potential = mu / np.linalg.norm([energy_ephemeris_df['x'], energy_ephemeris_df['y'], energy_ephemeris_df['z']], axis=0)
            print(f"first five values of monopole potential: {monopole_potential[:5]}")
            energy_ephemeris_df['monopole'] = monopole_potential

            U_non_sphericals = []
            U_j2s = []
            for _, row in energy_ephemeris_df.iterrows():
                print(f"computing potential for {row['lat']}, {row['lon']}, {row['alt']}")
                phi_rad = np.radians(row['lat'])
                lambda_rad = np.radians(row['lon'])
                r = row['alt']
                degree = 64
                order = 64
                date = datetime_to_absolutedate(row['UTC'])
                U_non_spher =  compute_gravitational_potential(r, phi_rad, lambda_rad, degree, order, date)
                U_j2 = U_J2(r, phi_rad, lambda_rad)
                U_j2s.append(U_j2)
                U_non_sphericals.append(U_non_spher)

            #total energies
            energy_total_spherical = kinetic_energy  - monopole_potential
            energy_total_J2 = kinetic_energy  - monopole_potential + U_j2s
            energy_total_HOT = kinetic_energy  - monopole_potential + U_non_sphericals

            #difference between total energies
            spherical_total_diff = energy_total_spherical[0] - energy_total_spherical
            j2_total_diff = energy_total_J2[0] - energy_total_J2
            HOT_total_diff = energy_total_HOT[0] - energy_total_HOT
            
            #individual energy components diff
            kinetic_energy_diff = kinetic_energy[0] - kinetic_energy
            monopole_potential_diff = monopole_potential[0] - monopole_potential
            j2_diff = U_j2s[0] - U_j2s
            HOT_diff = U_non_sphericals[0] - U_non_sphericals
            
            #in a dataframe save , x,y,z,u,v,w, lat, lon, alt, kinetic_energy, monopole, U_j2, U_non_spherical, j2_total_diff, HOT_total_diff
            energy_ephemeris_df['U_j2'] = U_j2s
            energy_ephemeris_df['U_non_spherical'] = U_non_sphericals
            energy_ephemeris_df['j2_total_diff'] = j2_total_diff
            energy_ephemeris_df['HOT_total_diff'] = HOT_total_diff
            export_df = energy_ephemeris_df[['MJD','x', 'y', 'z', 'xv', 'yv', 'zv', 'lat', 'lon', 'alt', 'kinetic_energy', 'monopole', 'U_j2', 'U_non_spherical', 'j2_total_diff', 'HOT_total_diff']]
            export_df.to_csv(os.path.join(folder_save, f"{sat_name}_energy_components_{start_date_utc}_{end_date_utc}.csv"), index=False)

        #if the ephemeris does not contain a columns called MJD, then use start_date_utc and assume 30s time steps
        if 'MJD' not in energy_ephemeris_df.columns:
            energy_ephemeris_df['MJD'] = [utc_to_mjd(start_date_utc) + 30/86400 * i for i in range(len(energy_ephemeris_df))]
        if 'UTC' not in energy_ephemeris_df.columns:
            energy_ephemeris_df['UTC'] = [start_date_utc + pd.Timedelta(seconds=30) * i for i in range(len(energy_ephemeris_df))]
            #TODO: the above will only really work if the ephemeris is 30s time steps
        EDR = -energy_ephemeris_df['HOT_total_diff']
        #now get the 180-point rolling average of the EDR
        EDR_180 = EDR.rolling(window=180, min_periods=1).mean()
        velocity = np.linalg.norm([energy_ephemeris_df['xv'], energy_ephemeris_df['yv'], energy_ephemeris_df['zv']], axis=0)
        time_steps = energy_ephemeris_df['MJD']
        CD = 3.2  # From Mehta et al 2013
        A_ref = 1.004  # From Mehta et al 2013
        mass = 600.0
        rho_eff = compute_rho_eff(EDR, velocity, CD, A_ref, mass, time_steps)
        rho_eff_180 = compute_rho_eff(EDR_180, velocity, CD, A_ref, mass, time_steps)
        jb08_rhos = []
        dtm2000_rhos = []
        nrlmsise00_rhos = []
        x = energy_ephemeris_df['x']
        y = energy_ephemeris_df['y']
        z = energy_ephemeris_df['z']
        u = energy_ephemeris_df['xv']
        v = energy_ephemeris_df['yv']
        w = energy_ephemeris_df['zv']
        print(f"columns in ephemeris_df: {energy_ephemeris_df.columns}")
        t = pd.to_datetime(energy_ephemeris_df['UTC'])
        
        #use posvel_to_sma to get the semi-major axis for each point
        print(f"iterating over num of points: {len(x)}")
        for i in range(len(x)):
            sma = posvel_to_sma(x[i], y[i], z[i], u[i], v[i], w[i])
            energy_ephemeris_df.at[i, 'sma'] = sma
            pos = np.array([x[i], y[i], z[i]])
            print(f"pos: {pos}, t: {t[i]}")
            #every hour get the rho_eff from the JB08, DTM2000, and NRLMSISE00 models
            if i % 1000 == 0:
                print(f"querying JB08, DTM2000, and NRLMSISE00 at time: {t[i]}")
                jb08_rho = query_jb08(pos, t[i])
                dtm2000_rho = query_dtm2000(pos, t[i])
                nrlmsise00_rho = query_nrlmsise00(pos, t[i])
                energy_ephemeris_df.at[i, 'jb08_rho'] = jb08_rho
                energy_ephemeris_df.at[i, 'dtm2000_rho'] = dtm2000_rho
                energy_ephemeris_df.at[i, 'nrlmsise00_rho'] = nrlmsise00_rho
                jb08_rhos.append((jb08_rho, t[i]))
                dtm2000_rhos.append((dtm2000_rho, t[i]))
                nrlmsise00_rhos.append((nrlmsise00_rho, t[i]))
            
        smas = energy_ephemeris_df['sma']

        #get the rolling average of the sma
        smas_180 = smas.rolling(window=180, min_periods=1).mean()

        #slice the rolling sma and rho_eff by 180 points at the start and at the finish
        smas_180 = smas_180[180:-180]
        rho_eff_180 = rho_eff_180[180:-180]
        #slice the MJDs to be the same length as the rolling sma and rho_eff
        UTCs = energy_ephemeris_df['UTC'][180:-180]

        # plt.plot(UTCs, rho_eff)
        # plt.xlabel('Modified Julian Date')
        # plt.ylabel('EDR Density (kg/m^3)')
        # plt.title(f"{sat_name}: \"Instantaneous\" Effective Density")
        # plt.grid(True)
        # plt.show()

        #now same as above but with the 180-point rolling average
        jb08_rhos = np.array(jb08_rhos)
        dtm2000_rhos = np.array(dtm2000_rhos)
        nrlmsise00_rhos = np.array(nrlmsise00_rhos)

        # Now you can properly index the arrays
        plt.plot(UTCs, rho_eff_180, label='EDR Density')
        plt.plot(jb08_rhos[:, 1], jb08_rhos[:, 0], label='JB08 Density')
        plt.plot(dtm2000_rhos[:, 1], dtm2000_rhos[:, 0], label='DTM2000 Density')
        plt.plot(nrlmsise00_rhos[:, 1], nrlmsise00_rhos[:, 0], label='NRLMSISE00 Density')
        plt.xlabel('Modified Julian Date')
        plt.ylabel('Effective Density (kg/m^3)')
        plt.title(f"{sat_name}: 180-point Rolling Average Effective Density")
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig("output/DensityInversion/EDR_plots/EDR_{sat_name}_effective_density_{start_date_utc}_{end_date_utc}.png")

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

if __name__ == "__main__":
    main()
#### PLOTTING ####
