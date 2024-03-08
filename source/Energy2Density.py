# Density Inversion Pseudo-Code


#1, Calculate orbital energy at each ephemeris data point
#1.1 energy = v**2/2 - omega_earth**2 *(x**2+y**2)/2 - mu/r - U_non_spherical
     #where r and v and pos and velocity vector norms in ECEF, w is the earth's rotation rate, mu is the gravitational parameter, and U_non_spherical is the gravitational potential
     # in the absence of non-conservatice forces, energy is conserved along an orbit
     #then add 3bp
#2, track the change in this quantity between orbits 


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
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants
from org.orekit.utils import IERSConventions

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

def compute_gravitational_potential(r, phi, lambda_, degree, order, P_nm_all, date):
    provider = GravityFieldFactory.getNormalizedProvider(degree, order)
    harmonics = provider.onDate(date)

    # Initialize gravitational potential deviation (V_dev) as 0.0, not including the central term mu/r
    V_dev = 0.0

    for n in range(1, degree + 1):  # Start from n = 1 to exclude the central potential
        radial_term = (R_earth / r)**n
        for m in range(0, n + 1):
            C_nm = harmonics.getNormalizedCnm(n, m)
            S_nm = harmonics.getNormalizedSnm(n, m)
            P_nm_sinphi = P_nm_all[n][m]

            if m == 0:
                sectorial_term = P_nm_sinphi * C_nm  # cos(0) = 1 implicitly used
            else:
                sectorial_term = P_nm_sinphi * (C_nm * np.cos(m * lambda_) + S_nm * np.sin(m * lambda_))

            # Accumulate the deviation from spherical potential
            V_dev += (R_earth / r) * radial_term * sectorial_term

            # Print J2, J3, and J4 energies inside the loop for specific conditions
            #Sanity checks
            # if n == 2 and m == 0:
            #     J2_energy_inside = -mu / r * radial_term * sectorial_term
            #     print(f"J2 energy inside the loop: {J2_energy_inside} J/kg")
            # if n == 3 and m == 0:
            #     J3_energy_inside = -mu / r * radial_term * sectorial_term
            #     print(f"J3 energy inside the loop: {J3_energy_inside} J/kg")
            # if n == 4 and m == 0:
            #     J4_energy_inside = -mu / r * radial_term * sectorial_term
            #     print(f"J4 energy inside the loop: {J4_energy_inside} J/kg")

    V_total = -mu / r + V_dev
    print(f"for degree {degree} and order {order}, the total gravitational potential is {V_total} J/kg")

    return V_total


def energy(ephemeris_df, degree, order):
    mu = Constants.WGS84_EARTH_MU
    satellite_mass = 1
    omega_earth = 7.2921159e-5
    x = ephemeris_df['x_ecef']
    y = ephemeris_df['y_ecef']
    z = ephemeris_df['z_ecef']
    xv = ephemeris_df['xv_ecef']
    yv = ephemeris_df['yv_ecef']
    zv = ephemeris_df['zv_ecef']
    vvec = np.linalg.norm([xv, yv, zv], axis=0)
    rvec = np.linalg.norm([x, y, z], axis=0)
    # kinetic_energy = satellite_mass*(vvec**2)/2
    # earth_rotational_energy = omega_earth**2 * ((x**2 + y**2)/2)
    kinetic_energy = 0
    earth_rotational_energy = 0
    monopole = -mu / rvec # that should probs be negative?
    # ephemeris_df['U_spherical'] = kinetic_energy - earth_rotational_energy - monopole

    # ephemeris_df['kinetic_energy'] = kinetic_energy
    ephemeris_df['kinetic_energy'] = 0
    # ephemeris_df['earth_rotational_energy'] = earth_rotational_energy
    ephemeris_df['earth_rotational_energy'] = 0
    ephemeris_df['monopole'] = monopole
    
    #now get the non-spherical potential energy
    converted_coords = np.vectorize(ecef_to_lla, signature='(),(),()->(),(),()')(ephemeris_df['x_ecef'], ephemeris_df['y_ecef'], ephemeris_df['z_ecef'])
    ephemeris_df[['lat', 'lon', 'alt']] = np.column_stack(converted_coords)
    legendre_cache = {phi: associated_legendre_polynomials(degree, order, np.sin(phi)) for phi in np.unique(np.radians(ephemeris_df['lat']))}

    potentials = []
    for _, row in ephemeris_df.iterrows():
        phi_rad = np.radians(row['lat'])
        P_nm_all = legendre_cache[phi_rad]
        lambda_rad = np.radians(row['lon'])
        r = row['alt']
        date = datetime_to_absolutedate(row['UTC'])
        potential = compute_gravitational_potential(r, phi_rad, lambda_rad, degree, order, P_nm_all, date)
        potentials.append(potential)
    ephemeris_df['U_grav_field'] = potentials
    # Calculate total energy
    ephemeris_df['U_non_spherical'] = ephemeris_df['U_grav_field'] - ephemeris_df['monopole']
    # ephemeris_df['energy'] = ephemeris_df['U_spherical'] - ephemeris_df['U_grav_field']
    ephemeris_df['energy'] = ephemeris_df['U_non_spherical']

    # energy_diff is the difference between the energy at the first point and the energy at the current point
    ephemeris_df['energy_diff'] = ephemeris_df['energy'] - ephemeris_df['energy'].iloc[0]
    ephemeris_df['energy_diff_kinetic'] = ephemeris_df['kinetic_energy'] - ephemeris_df['kinetic_energy'].iloc[0]
    ephemeris_df['energy_diff_rotational'] = ephemeris_df['earth_rotational_energy'] - ephemeris_df['earth_rotational_energy'].iloc[0]
    ephemeris_df['energy_diff_monopole'] = ephemeris_df['monopole'] - ephemeris_df['monopole'].iloc[0]
    ephemeris_df['energy_diff_non_spherical'] = ephemeris_df['U_non_spherical'] - ephemeris_df['U_non_spherical'].iloc[0]
    ephemeris_df['energy_diff_grav_field'] = ephemeris_df['U_grav_field'] - ephemeris_df['U_grav_field'].iloc[0]
    return ephemeris_df

def main():
    sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        #skip the first 100 points
        ephemeris_df = ephemeris_df[100:]         
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

        plt.figure()
        ephemeris_df= energy(ephemeris_df, 6, 6)
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff'], label=f'Net Energy Diff', linestyle='dotted')
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_kinetic'], label='Kinetic only')
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_rotational'], label='Rotational only')
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_monopole'], label='Monopole only')
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy_diff_grav_field'], label='Non-Spherical only')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Energy Comparison for {sat_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"output/DensityInversion/OrbitEnergy/Relative_Orbital_Energy_budget_{sat_name}.png")
        plt.show()

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
        plt.plot(ephemeris_df['MJD'], ephemeris_df['U_non_spherical'], label='Point Mass Related Potential Energy')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Spherical Potential Energy for {sat_name}')
        plt.grid(True)

        plt.subplot(2, 3, 5)
        plt.plot(ephemeris_df['MJD'], ephemeris_df['U_grav_field'], label='Non-Spherical Potential Energy')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Non-monopole Gravity Energy for {sat_name}')
        plt.grid(True)

        plt.subplot(2, 3, 6)
        plt.plot(ephemeris_df['MJD'], ephemeris_df['U_non_spherical'], label='monopole-gravity field')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Total Energy')
        plt.grid(True)

        plt.suptitle(f'Energy Budget for {sat_name}')
        
        plt.tight_layout()
        
        plt.savefig(f"output/DensityInversion/OrbitEnergy/Orbital_Energy_budget_{sat_name}.png")
        plt.show()


if __name__ == "__main__":
    main()