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
    # sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    sat_names_to_test = ["CHAMP"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        ephemeris_df = ephemeris_df.head(400)
        # take the UTC column and convert to mjd
        ephemeris_df['MJD'] = [utc_to_mjd(dt) for dt in ephemeris_df['UTC']]
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

        monopole_potential = mu / np.linalg.norm([ephemeris_df['x_ecef'], ephemeris_df['y_ecef'], ephemeris_df['z_ecef']], axis=0)
        print(f"first five values of monopole potential: {monopole_potential[:5]}")
        ephemeris_df['monopole'] = monopole_potential

        earth_rotational_energy = (1/2) * (7.2921150e-5)**2 * (ephemeris_df['x_ecef']**2 + ephemeris_df['y_ecef']**2)
        print(f"first five values of earth rotational energy: {earth_rotational_energy[:5]}")
        ephemeris_df['earth_rotational_energy'] = earth_rotational_energy

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
            # U_non_spher =  compute_gravitational_potential(r, phi_rad, lambda_rad, degree, order, date)
            # print(f"U_non_spher: {U_non_spher}")
            U_j2 = U_J2(r, phi_rad, lambda_rad)
            print(f"U_j2: {U_j2}")
            U_j2s.append(U_j2)
            # U_non_sphericals.append(U_non_spher)


        #plot kinetic - monopole potential
        plt.figure()
        plt.plot(ephemeris_df['MJD'], kinetic_energy - monopole_potential, label='Kinetic - Monopole Potential', color='r')
        plt.plot(ephemeris_df['MJD'], kinetic_energy - monopole_potential - earth_rotational_energy, label='Kinetic - Monopole Potential - Earth Rotational Energy', color='g')
        plt.plot(ephemeris_df['MJD'], kinetic_energy - monopole_potential + earth_rotational_energy - U_j2s, label='Kinetic - Monopole Potential - Earth Rotational Energy - U_non_spherical', color='b')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('Potential Energy Change (J/kg)')
        plt.title('Potential Energy Change')
        plt.grid(True)
        plt.savefig(os.path.join(folder_save, f"{sat_name}_kinetic_minus_monopole_potential_energy_change.png"))
        plt.show()


        energy_spherical = kinetic_energy - earth_rotational_energy - monopole_potential
        energy_non_spherical = kinetic_energy - earth_rotational_energy - monopole_potential - U_non_sphericals

        #plot monopole potential
        plt.figure()
        plt.plot(ephemeris_df['MJD'], monopole_potential, label='Monopole Potential', color='r')
        monopole_potential_diff = monopole_potential - U_non_sphericals
        plt.plot(ephemeris_df['MJD'], monopole_potential_diff, label='Monopole minus potential', color='g')
        plt.legend()
        plt.twinx()
        plt.plot(ephemeris_df['MJD'], ephemeris_df['alt'], label='Altitude', color='b')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('Potential Energy Change (J/kg)')
        plt.title('Potential Energy Change')
        plt.grid(True)
        plt.savefig(os.path.join(folder_save, f"{sat_name}_monopole_potential_energy_change.png"))
        # plt.show()


        #for monopole, J2, and U_non_spherical, plot the difference betyween the potential energy at t0 and ti
        monopole_potential_diff = monopole_potential[0] - monopole_potential
        u_non_spherical_diff = U_non_sphericals[0] - U_non_sphericals
        kinetic_energy_diff = kinetic_energy[0] - kinetic_energy
        earth_rotational_energy_diff = earth_rotational_energy[0] - earth_rotational_energy
        spherical_total_diff = energy_spherical[0] - energy_spherical
        non_spherical_total_diff = energy_non_spherical[0] - energy_non_spherical
        plt.figure()
        plt.plot(ephemeris_df['MJD'], monopole_potential_diff, label='Monopole Potential Energy Change', color='r')
        plt.plot(ephemeris_df['MJD'], u_non_spherical_diff, label='U_non_spherical Potential Energy Change', color='g')
        plt.plot(ephemeris_df['MJD'], kinetic_energy_diff, label='Kinetic Energy Change', color='y')
        plt.plot(ephemeris_df['MJD'], earth_rotational_energy_diff, label='Earth Rotational Energy Change', color='m')
        plt.plot(ephemeris_df['MJD'], spherical_total_diff, label='Spherical Total Energy Change', color='c')
        plt.plot(ephemeris_df['MJD'], non_spherical_total_diff, label='Non-Spherical Total Energy Change', color='k')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('Potential Energy Change (J/kg)')
        plt.title('Potential Energy Change')
        plt.grid(True)
        plt.savefig(os.path.join(folder_save, f"{sat_name}_energy_change.png"))
        # plt.show()

        plt.figure()
        #plot lat, lon, J2
        plt.scatter(ephemeris_df['lon'], ephemeris_df['lat'], c=U_j2s, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        #make sure the plot ranges from -180 to 180 for longitude and -90 to 90 for latitude
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.title('J2 Potential Energy')
        plt.grid(True)
        plt.savefig(os.path.join(folder_save, f"{sat_name}_J2_potential_energy.png"))
        # plt.show()
        
        #plot difference between U_j2 and U_non_spherical
        grav_pot_diff = np.array(U_j2s) - np.array(U_non_sphericals)
        plt.figure()
        plt.plot(ephemeris_df['MJD'], grav_pot_diff, label='U_j2 - H.O.T', color='r') 
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('Potential Energy (J/kg)')
        plt.title('Difference between U_j2 and U_non_spherical')
        plt.grid(True)
        plt.savefig(os.path.join(folder_save, f"{sat_name}_U_j2_minus_U_non_spherical.png"))
        # plt.show()
        #first five energy spherical values
        print(f"first five energy spherical values: {energy_spherical[:5]}")
        #first five U_non_sphericals values
        print(f"first five U_non_sphericals values: {U_non_sphericals[:5]}")
        energy_non_spherical = energy_spherical - U_non_sphericals
        energy_j2 = energy_spherical - U_j2s

        #now plot the energy change from t0 to ti for energy spherical and energy non-spherical
        energy_diff_non_spherical = energy_non_spherical[0] - energy_non_spherical
        energy_diff_j2 = energy_j2[0] - energy_j2

        plt.figure()
        # plt.plot(ephemeris_df['MJD'], energy_diff_spherical, label='Spherical Energy Change', color='r')
        plt.plot(ephemeris_df['MJD'], energy_diff_non_spherical, label='Non-Spherical Energy Change', color='b')
        plt.plot(ephemeris_df['MJD'], energy_diff_j2, label='Energy Change', color='g')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('Energy Change (J/kg)')
        plt.title('Spherical Potential Energy Change')
        plt.grid(True)
        plt.savefig(os.path.join(folder_save, f"{sat_name}_spherical_vs_non_spherical_potential_energy_change.png"))
        # plt.show()

        #same plot as above but as a functon of latitude
        plt.figure()
        # plt.plot(ephemeris_df['lat'], energy_diff_spherical, label='Spherical Potential Energy Change', color='r')
        plt.plot(ephemeris_df['lat'], energy_diff_non_spherical, label='Non-Spherical Potential Energy Change', color='b')
        plt.plot(ephemeris_df['lat'], energy_diff_j2, label='J2 Potential Energy Change', color='g')
        plt.legend()
        plt.xlabel('Latitude')
        plt.ylabel('Energy Change (J/kg)')
        plt.title('Spherical Potential Energy Change')
        plt.grid(True)
        plt.savefig(os.path.join(folder_save, f"{sat_name}_spherical_vs_non_spherical_potential_energy_change_lat.png"))
        # plt.show()


if __name__ == "__main__":
    # latitudes = np.linspace(-90, 90, 100)
    # V_j2_values = []
    # V_all_values = []
    # for lat in latitudes:
    #     phi_rad = np.radians(lat)
    #     lambda_rad = 0  # Assuming longitude is 0 for simplicity
    #     # Assuming altitude is at 1000km
    #     r = 6378137 + 1000000
    #     V_j2 = U_J2(r, phi_rad, lambda_rad)
    #     V_j2_values.append(V_j2)
    #     V_all = compute_gravitational_potential(r, phi_rad, lambda_rad, 6, 6, datetime_to_absolutedate(datetime.datetime.now()))
    #     V_all_values.append(V_all)

    # plt.figure()
    # plt.plot(latitudes, V_j2_values, label='V_j2', color='r')
    # plt.plot(latitudes, V_all_values, label='V_all', color='b')
    # plt.xlabel('Latitude')
    # plt.ylabel('V_j2')
    # plt.title('V_j2 vs Latitude')
    # plt.grid(True)
    # plt.savefig(os.path.join(folder_save, "V_j2_vs_latitude.png"))
    # # plt.show()

    main()