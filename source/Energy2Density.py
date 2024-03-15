import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants

import os
from tools.utilities import utc_to_mjd, HCL_diff, get_satellite_info, pos_vel_from_orekit_ephem, EME2000_to_ITRF, ecef_to_lla
from tools.sp3_2_ephemeris import sp3_ephem_to_df
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy.special import lpmn


mu = Constants.WGS84_EARTH_MU
R_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
J2 =Constants.EGM96_EARTH_C20

def U_J2(r, phi, lambda_):
    theta = np.pi / 2 - phi  # Convert latitude to colatitude
    # J2 potential term calculation, excluding the monopole term
    V_j2 = -mu / r * J2 * (R_earth / r)**2 * (3 * np.cos(theta)**2 - 1) / 2
    return V_j2

def associated_legendre_polynomials(degree, order, x):
    for n in range(degree + 1):
        for m in range(min(n, order) + 1):
            P_nm, _ = lpmn(m, n, x)
    return P_nm[0][-1]

def compute_gravitational_potential(r, phi, lambda_, degree, order, date):
    # provider = GravityFieldFactory.getNormalizedProvider(degree, order)
    provider = GravityFieldFactory.getUnnormalizedProvider(degree, order)
    harmonics = provider.onDate(date)
    mu = 3.986004418e14  # Earth's gravitational constant
    R_earth = 6378137  # Earth's radius in meters
    V_dev = 0.0

    for n in range(degree + 1):
        for m in range(min(n, order) + 1):
            C_nm = harmonics.getUnnormalizedCnm(n, m)
            S_nm = harmonics.getUnnormalizedSnm(n, m)
            P_nm = associated_legendre_polynomials(n, m, np.sin(phi))
            sectorial_term = P_nm * (C_nm * np.cos(m * lambda_) + S_nm * np.sin(m * lambda_))
            contribution = (mu / r) * (R_earth / r)**n * sectorial_term
            if not (n == 0 and m == 0):  # Skip the central term in V_dev
                V_dev += contribution
    V_total = -V_dev
    return V_total

folder_save = "output/DensityInversion/OrbitEnergy"

def main():
    sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    # sat_names_to_test = ["CHAMP"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        ephemeris_df = ephemeris_df.head(5000)
        # take the UTC column and convert to mjd
        ephemeris_df['MJD'] = [utc_to_mjd(dt) for dt in ephemeris_df['UTC']]
        start_date_utc = ephemeris_df['UTC'].iloc[0]
        end_date_utc = ephemeris_df['UTC'].iloc[-1]
        x_ecef, y_ecef, z_ecef, xv_ecef, yv_ecef, zv_ecef = ([] for _ in range(6))
        # Convert EME2000 to ITRF for each point
        for _, row in ephemeris_df.iterrows():
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
        ephemeris_df['x_ecef'] = x_ecef
        ephemeris_df['y_ecef'] = y_ecef
        ephemeris_df['z_ecef'] = z_ecef
        ephemeris_df['xv_ecef'] = xv_ecef
        ephemeris_df['yv_ecef'] = yv_ecef
        ephemeris_df['zv_ecef'] = zv_ecef

        #conver the ecef coordinates to lla
        converted_coords = np.vectorize(ecef_to_lla, signature='(),(),()->(),(),()')(ephemeris_df['x_ecef'], ephemeris_df['y_ecef'], ephemeris_df['z_ecef'])
        ephemeris_df[['lat', 'lon', 'alt']] = np.column_stack(converted_coords)

        kinetic_energy = (np.linalg.norm([ephemeris_df['xv'], ephemeris_df['yv'], ephemeris_df['zv']], axis=0))**2/2
        print(f"first five values of kinetic energy: {kinetic_energy[:5]}")
        ephemeris_df['kinetic_energy'] = kinetic_energy

        monopole_potential = mu / np.linalg.norm([ephemeris_df['x'], ephemeris_df['y'], ephemeris_df['z']], axis=0)
        print(f"first five values of monopole potential: {monopole_potential[:5]}")
        ephemeris_df['monopole'] = monopole_potential

        U_non_sphericals = []
        U_j2s = []
        for _, row in ephemeris_df.iterrows():
            print(f"computing potential for {row['lat']}, {row['lon']}, {row['alt']}")
            phi_rad = np.radians(row['lat'])
            lambda_rad = np.radians(row['lon'])
            r = row['alt']
            degree = 4
            order = 4
            date = datetime_to_absolutedate(row['UTC'])
            U_non_spher =  compute_gravitational_potential(r, phi_rad, lambda_rad, degree, order, date)
            # print(f"U_non_spher: {U_non_spher}")
            U_j2 = U_J2(r, phi_rad, lambda_rad)
            # print(f"U_j2: {U_j2}")
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

        #make 3 subplots
        #set sns style
        sns.set_theme(style='whitegrid')
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        #plot total energy differences
        axs[0].plot(ephemeris_df['MJD'], spherical_total_diff, label='Spherical Total Energy Difference', color='xkcd:hot magenta')
        axs[0].plot(ephemeris_df['MJD'], j2_total_diff, label='J2 Total Energy Difference', color='xkcd:tangerine')
        axs[0].plot(ephemeris_df['MJD'], HOT_total_diff, label='HOT Total Energy Difference', color='xkcd:greenish', linestyle='--')
        axs[0].grid(True)
        axs[0].legend()
        axs[0].set_xlabel('')
        axs[0].set_ylabel('Energy_i - Energy_0 (J/kg)')
        axs[0].set_title('Total Relative Energy Differences')

        #plot same as the top but only J2 and HOT
        axs[1].plot(ephemeris_df['MJD'], j2_total_diff, label='J2 Total Energy Difference', color='xkcd:tangerine')
        axs[1].plot(ephemeris_df['MJD'], HOT_total_diff, label='HOT Total Energy Difference', color='xkcd:greenish', linestyle='--')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_xlabel('')
        axs[1].set_ylabel('Energy_i - Energy_0 (J/kg)')
        axs[1].set_title('Total Relative Energy Differences (J2 and H.O.T only)')
        
        #plot individual energy differences
        axs[2].plot(ephemeris_df['MJD'], kinetic_energy_diff, label='Kinetic Energy Difference', color='xkcd:minty green')
        axs[2].plot(ephemeris_df['MJD'], monopole_potential_diff, label='Monopole Potential Difference', color='xkcd:easter purple')
        axs[2].plot(ephemeris_df['MJD'], j2_diff, label='J2 Potential Difference', color='xkcd:tangerine')
        axs[2].plot(ephemeris_df['MJD'], HOT_diff, label='HOT Potential Difference', color='xkcd:greenish', linestyle='--')
        axs[2].legend()
        axs[2].grid(True)
        axs[2].set_xlabel('Modified Julian Date')
        axs[2].set_ylabel('Energy_i - Energy_0 (J/kg)')
        axs[2].set_title('Individual Relative Energy Differences')

        #main title with satellite name
        fig.suptitle(f"{sat_name}: Energy Budget", fontsize=14)

        plt.tight_layout()

        plt.savefig(os.path.join(folder_save, f"{sat_name}_energy_components_{start_date_utc}_{end_date_utc}.png"))
        # plt.show()

if __name__ == "__main__":
    main()