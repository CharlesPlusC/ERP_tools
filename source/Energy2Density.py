import orekit
from orekit.pyhelpers import setup_orekit_curdir

# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.utils import PVCoordinates
from org.orekit.orbits import CartesianOrbit
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity.potential import GravityFieldFactory, TideSystem
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, SolidTides, OceanTides, ThirdBodyAttraction, Relativity, NewtonianAttraction
from orekit import JArray_double
from org.orekit.orbits import OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import TimeScalesFactory   
from org.orekit.propagation import SpacecraftState
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
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants

mu = Constants.WGS84_EARTH_MU
R_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

def associated_legendre_polynomials(n, m, sin_phi):
    pnm = np.zeros((n + 1, m + 1))
    pnm_unnorm, _ = lpmn(m, n, sin_phi)
    pnm_unnorm = pnm_unnorm.T
    for ni in range(n + 1):
        for mi in range(min(ni, m) + 1):
            normalization_factor = np.sqrt(((2 * ni + 1) * math.factorial(ni - mi)) / (2 * math.factorial(ni + mi)))
            pnm[ni, mi] = normalization_factor * pnm_unnorm[ni, mi]
    return pnm  # Use the normalized pnm for calculations.

# def compute_gravitational_potential(r, phi, lambda_, degree, order, P_nm_all, date):
#     provider = GravityFieldFactory.getNormalizedProvider(degree, order)
#     harmonics = provider.onDate(date)

#     # Initialize gravitational potential deviation (V_dev) as 0.0, not including the central term mu/r
#     V_dev = 0.0
#     for n in range(1, degree + 1):  # Start from n = 1 to exclude the central potential
#         radial_term = (R_earth / r)**n
#         for m in range(0, n + 1):
#             C_nm = harmonics.getNormalizedCnm(n, m)
#             S_nm = harmonics.getNormalizedSnm(n, m)
#             P_nm_sinphi = P_nm_all[n][m]

#             if m == 0:
#                 sectorial_term = P_nm_sinphi * C_nm  # cos(0) = 1 implicitly used
#             else:
#                 sectorial_term = P_nm_sinphi * (C_nm * np.cos(m * lambda_) + S_nm * np.sin(m * lambda_))

#             # Accumulate the deviation from spherical potential
#             V_dev = mu / r * radial_term * sectorial_term
            
#             # Print J2, J3, and J4 energies inside the loop
#             #Sanity checks
#             # if n == 2 and m == 0:
#             #     J2_energy_inside = -mu / r * radial_term * sectorial_term
#             #     print(f"J2 energy inside the loop: {J2_energy_inside} J/kg")
#             # if n == 3 and m == 0:
#             #     J3_energy_inside = -mu / r * radial_term * sectorial_term
#             #     print(f"J3 energy inside the loop: {J3_energy_inside} J/kg")
#             # if n == 4 and m == 0:
#             #     J4_energy_inside = -mu / r * radial_term * sectorial_term
#             #     print(f"J4 energy inside the loop: {J4_energy_inside} J/kg")

#     V_total = -mu / r + V_dev
#     # print(f"for degree {degree} and order {order}, the total gravitational potential is {V_total} J/kg")

#     return V_total

def compute_gravitational_potential(r, phi, lambda_, degree, order, P_nm_all, date):
    # Compute the J2 perturbation term, assuming phi is latitude in radians
    J2 = 1.08262668e-3
    theta = np.pi / 2 - phi  # Convert latitude to colatitude

    # J2 potential term and monopole potential
    V_m = -mu / r
    V_j2 = -mu / r * (1 - J2 * (R_earth / r)**2 * (3 * np.cos(theta)**2 - 1) / 2)

    # The total gravitational potential includes J2 contribution
    V_total = V_j2

    return V_total

# def energy(ephemeris_df, degree, order):
#     mu = Constants.WGS84_EARTH_MU
#     x = ephemeris_df['x_ecef']
#     y = ephemeris_df['y_ecef']
#     z = ephemeris_df['z_ecef']
#     rvec = np.linalg.norm([x, y, z], axis=0)
#     kinetic_energy = ephemeris_df['kinetic_energy']
#     earth_rotational_energy = ephemeris_df['earth_rotational_energy']
#     monopole = -mu / rvec
#     ephemeris_df['monopole'] = monopole
#     #now get the non-spherical potential energy
#     converted_coords = np.vectorize(ecef_to_lla, signature='(),(),()->(),(),()')(ephemeris_df['x_ecef'], ephemeris_df['y_ecef'], ephemeris_df['z_ecef'])
#     ephemeris_df[['lat', 'lon', 'alt']] = np.column_stack(converted_coords)
#     legendre_cache = {phi: associated_legendre_polynomials(degree, order, np.sin(phi)) for phi in np.unique(np.radians(ephemeris_df['lat']))}

#     potentials = []
#     for _, row in ephemeris_df.iterrows():
#         print(f"computing potential for {row['lat']}, {row['lon']}, {row['alt']}")
#         phi_rad = np.radians(row['lat'])
#         P_nm_all = legendre_cache[phi_rad]
#         lambda_rad = np.radians(row['lon'])
#         r = row['alt']
#         date = datetime_to_absolutedate(row['UTC'])
#         potential = compute_gravitational_potential(r, phi_rad, lambda_rad, degree, order, P_nm_all, date)
#         #difference between the monopole and the non-spherical potential
#         potentials.append(potential)

#     #plot the difference between the monopole and the non-spherical potential as a function of latitude
#     grav_diff = ephemeris_df['monopole'] - potentials
#     plt.figure()
#     plt.plot(ephemeris_df['lat'], grav_diff, label='Deviation from Monopole')
#     plt.xlabel('Latitude (degrees)')
#     plt.ylabel('Energy (J/kg)')
#     plt.title('Non-Spherical Potential Energy vs Monopole Energy')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     print(f"first five values of monopole: {ephemeris_df['monopole'].head()}")
#     print(f"first five values of potentials: {potentials[:5]}")
#     ephemeris_df['U_grav_field'] =  monopole - potentials
#     ephemeris_df['deviations'] = ephemeris_df['U_grav_field'] - ephemeris_df['monopole']

#     ephemeris_df['U_spherical'] = kinetic_energy - earth_rotational_energy - monopole
#     ephemeris_df['U_non_spherical'] = kinetic_energy - earth_rotational_energy  - monopole + potentials

#     #plot U_grav_field and monopole difference
#     plt.figure()
#     plt.plot(ephemeris_df['MJD'], ephemeris_df['deviations'], label='Deviation from Monopole')
#     plt.xlabel('MJD')
#     plt.ylabel('Energy (J/kg)')
#     plt.title('Non-Spherical Potential Energy vs Monopole Energy')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig("output/DensityInversion/OrbitEnergy/NonSphericalPotentialEnergy.png")
#     # plt.show()

#     #plot difference between kinetic and monopole
#     plt.figure()
#     plt.plot(ephemeris_df['MJD'], ephemeris_df['U_non_spherical'], label='Non Spherical Potential Energy')
#     plt.plot(ephemeris_df['MJD'], ephemeris_df['U_spherical'], label='Spherical Potential Energy')
#     plt.xlabel('MJD')
#     plt.ylabel('Energy (J/kg)')
#     plt.title('Kinetic Energy vs Monopole Energy')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig("output/DensityInversion/OrbitEnergy/KineticEnergy.png")
#     # plt.show()

#     print(f"first five values of monopole: {ephemeris_df['monopole'].head()}")
#     print(f"first five values of potentials: {potentials[:5]}")
#     print(f'first five monopole vs potentials difference: {ephemeris_df["monopole"].head() - potentials[:5]}')

#     # energy_diff is the difference between the energy at the first point and the energy at the current point
#     ephemeris_df['energy_diff_kinetic'] = ephemeris_df['kinetic_energy'] - ephemeris_df['kinetic_energy'].iloc[0]
#     ephemeris_df['energy_diff_rotational'] = ephemeris_df['earth_rotational_energy'] - ephemeris_df['earth_rotational_energy'].iloc[0]
#     ephemeris_df['energy_diff_monopole'] = ephemeris_df['monopole'] - ephemeris_df['monopole'].iloc[0]
#     ephemeris_df['energy_diff_spherical'] = ephemeris_df['U_spherical'] - ephemeris_df['U_spherical'].iloc[0]
#     ephemeris_df['energy_diff_non_spherical'] = ephemeris_df['U_non_spherical'] - ephemeris_df['U_non_spherical'].iloc[0]
#     ephemeris_df['energy_diff_grav_field'] = ephemeris_df['U_grav_field'] - ephemeris_df['U_grav_field'].iloc[0]
#     return ephemeris_df

def energy(ephemeris_df, degree, order):
    mu = Constants.WGS84_EARTH_MU
    omega_earth = 7.2921159e-5  # Earth's angular velocity in rad/s

    # Kinetic energy calculation
    kinetic_energy = -np.linalg.norm([ephemeris_df['xv_ecef'], ephemeris_df['yv_ecef'], ephemeris_df['zv_ecef']], axis=0)**2
    ephemeris_df['kinetic_energy'] = kinetic_energy

    # Earth's rotational energy calculation
    rotational_energy = 0.5 * omega_earth**2 * (ephemeris_df['x_ecef']**2 + ephemeris_df['y_ecef']**2)
    ephemeris_df['earth_rotational_energy'] = rotational_energy

    # Monopole gravitational potential energy
    rvec = np.linalg.norm([ephemeris_df['x_ecef'], ephemeris_df['y_ecef'], ephemeris_df['z_ecef']], axis=0)
    ephemeris_df['monopole'] = -mu / rvec

    converted_coords = np.vectorize(ecef_to_lla, signature='(),(),()->(),(),()')(ephemeris_df['x_ecef'], ephemeris_df['y_ecef'], ephemeris_df['z_ecef'])
    ephemeris_df[['lat', 'lon', 'alt']] = np.column_stack(converted_coords)

    # Calculate the gravitational potential energy with J2 perturbation
    potentials = []
    for _, row in ephemeris_df.iterrows():
        phi_rad = np.radians(row['lat'])
        lambda_rad = np.radians(row['lon'])
        r = np.linalg.norm([row['x_ecef'], row['y_ecef'], row['z_ecef']])
        date = datetime_to_absolutedate(row['UTC'])
        potential = compute_gravitational_potential(r, phi_rad, lambda_rad, degree, order, None, date)
        potentials.append(potential)

    ephemeris_df['U_grav_field'] = potentials

    # Total energy with and without J2 contribution
    print(f"first five values of monopole: {ephemeris_df['monopole'].head()}")
    print(f"first five values of potentials: {potentials[:5]}")
    print(f"first five values of kinetic energy: {ephemeris_df['kinetic_energy'].head()}")
    print(f"first five values of rotational energy: {ephemeris_df['earth_rotational_energy'].head()}")
    ephemeris_df['total_energy'] = ephemeris_df['kinetic_energy'] - ephemeris_df['earth_rotational_energy'] - ephemeris_df['U_grav_field'] 
    ephemeris_df['total_energy_no_J2'] = ephemeris_df['kinetic_energy'] - ephemeris_df['earth_rotational_energy'] - ephemeris_df['monopole'] 

    #plot change in energy from the first point
    ephemeris_df["energy_diff_no_j2"] = ephemeris_df['total_energy_no_J2'] - ephemeris_df['total_energy_no_J2'].iloc[0]
    ephemeris_df["energy_diff_all"] = ephemeris_df['total_energy'] - ephemeris_df['total_energy'].iloc[0]
    
    plt.figure()
    plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_no_j2'], label='Total Energy Difference (No J2)')
    plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_all'], label='Total Energy Difference (All)')
    plt.xlabel('MJD')
    plt.ylabel('Energy (J/kg)')
    plt.title('Total Energy Difference')
    plt.grid(True)
    plt.legend()
    plt.show()


    # Plotting the total energy with and without J2 contribution
    # plt.figure(figsize=(12, 6))
    # plt.plot(ephemeris_df['MJD'], ephemeris_df['total_energy'], label='Total Energy (with J2)')
    # plt.plot(ephemeris_df['MJD'], ephemeris_df['total_energy_no_J2'], label='Total Energy (without J2)', linestyle='--')
    # plt.xlabel('MJD')
    # plt.ylabel('Energy (J/kg)')
    # plt.title('Total Energy Comparison')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return ephemeris_df

def main():
    # sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    sat_names_to_test = ["CHAMP"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        #take every 10th po
        ephemeris_df = ephemeris_df.head(300)
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

        kinetic_energy = (np.linalg.norm([ephemeris_df['xv'], ephemeris_df['yv'], ephemeris_df['zv']], axis=0))**2/2
        ephemeris_df['kinetic_energy'] = kinetic_energy

        omega_earth = 7.2921159e-5
        earth_rotational_energy = omega_earth**2 * ((ephemeris_df['x_ecef']**2 + ephemeris_df['y_ecef']**2)/2)
        ephemeris_df['earth_rotational_energy'] = earth_rotational_energy

        ephemeris_df= energy(ephemeris_df, 2, 2)

        plt.figure()
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_non_spherical'], label='Total Energy Difference')
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_kinetic'], label='Kinetic only')
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_rotational'], label='Rotational only')
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_monopole'], label='Monopole only')
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_grav_field'], label='Non-Spherical Potential only')
        # plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_spherical'], label='With Monopole')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Energy Comparison for {sat_name}')
        plt.grid(True)
        plt.legend()
        timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"output/DensityInversion/OrbitEnergy/Relative_Orbital_Energy_budget_{sat_name}_{timenow}.png")
        # plt.show()

        #now plot the raw energy values each in their own subplot
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        plt.plot(ephemeris_df['MJD'], ephemeris_df['kinetic_energy'], label='Kinetic Energy')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Kinetic Energy for {sat_name}')
        plt.grid(True)

        plt.subplot(2, 3, 2)
        plt.plot(ephemeris_df['MJD'], ephemeris_df['earth_rotational_energy'], label='Rotational Energy')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Rotational Energy for {sat_name}')
        plt.grid(True)

        plt.subplot(2, 3, 3)
        plt.plot(ephemeris_df['MJD'], ephemeris_df['monopole'], label='Monopole Energy')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Monopole Energy for {sat_name}')
        plt.grid(True)

        plt.subplot(2, 3, 4)
        plt.plot(ephemeris_df['MJD'], ephemeris_df['U_spherical'], label='Point Mass Potential Energy')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Spherical Potential Energy for {sat_name}')
        plt.grid(True)

        plt.subplot(2, 3, 5)
        plt.plot(ephemeris_df['MJD'], ephemeris_df['deviations'], label='Gravity Anomaly Energy')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Gravity Anomaly Energy')
        plt.grid(True)

        plt.subplot(2, 3, 6)
        plt.plot(ephemeris_df['MJD'], ephemeris_df['U_non_spherical'], label='Total Energy')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Total Energy')
        plt.grid(True)

        plt.suptitle(f'Energy Budget for {sat_name}')
        
        plt.tight_layout()
        
        plt.savefig(f"output/DensityInversion/OrbitEnergy/Orbital_Energy_budget_{sat_name}.png")
        # plt.show()

if __name__ == "__main__":
    main()