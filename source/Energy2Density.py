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
    mu = Constants.WGS84_EARTH_MU
    a = provider.getAe()
    V = 0.0
    for n in range(0, degree + 1):
        for m in range(0, n + 1):
            C_nm = harmonics.getNormalizedCnm(n, m)
            S_nm = harmonics.getNormalizedSnm(n, m)
            P_nm_sinphi = P_nm_all[n, m]
            sectorial_term = P_nm_sinphi * (C_nm * np.cos(m * lambda_) + S_nm * np.sin(m * lambda_))

            V += ((a / r) ** (n + 1)) * sectorial_term
            if n == 2 and m == 0:
                # Optional: print J2 energy component, though it's more useful to consider the entire sum.
                J2_energy_component = -mu / r * ((a / r) ** (n + 1)) * sectorial_term
                print(f"energy due to J2: {J2_energy_component} J/kg")

    V *= -mu / r
    return V

def energy(ephemeris_df, degree, order):
    satellite_mass = 600.2  # kg
    mu = Constants.WGS84_EARTH_MU
    omega_earth = 7.2921159e-5
    v = np.sqrt(ephemeris_df['xv_ecef']**2 + ephemeris_df['yv_ecef']**2 + ephemeris_df['zv_ecef']**2)
    x = ephemeris_df['x_ecef']
    y = ephemeris_df['y_ecef']
    z = ephemeris_df['z_ecef']
    rvec = np.linalg.norm([x, y, z], axis=0)
    kinetic_energy = 0.5 * v**2
    earth_rotational_energy = 0.5 * omega_earth**2 * (x**2 + y**2)
    monopole = mu / rvec
    ephemeris_df['kinetic_energy'] = kinetic_energy
    ephemeris_df['earth_rotational_energy'] = earth_rotational_energy
    ephemeris_df['monopole'] = monopole
    ephemeris_df['U_spherical'] = kinetic_energy - earth_rotational_energy - (ephemeris_df['monopole'] * satellite_mass)
    ephemeris_df['monopole'] *= satellite_mass
    # kinetic and earth rotational energies are already in absolute terms  and should not be multiplied by the satellite's mass
    converted_coords = np.vectorize(ecef_to_lla, signature='(),(),()->(),(),()')(ephemeris_df['x_ecef'], ephemeris_df['y_ecef'], ephemeris_df['z_ecef'])
    ephemeris_df[['lat', 'lon', 'alt']] = np.column_stack(converted_coords)

    legendre_cache = {phi: associated_legendre_polynomials(degree, order, np.sin(phi)) for phi in np.unique(np.radians(ephemeris_df['lat']))}

    potentials = []
    for _, row in ephemeris_df.iterrows():
        phi_rad = np.radians(row['lat'])
        P_nm_all = legendre_cache[phi_rad]
        lambda_rad = np.radians(row['lon'])
        r = row['alt'] + Constants.EGM96_EARTH_EQUATORIAL_RADIUS
        date = datetime_to_absolutedate(row['UTC'])
        potential = compute_gravitational_potential(r, phi_rad, lambda_rad, degree, order, P_nm_all, date)
        potentials.append(potential)
    ephemeris_df['U_grav_field'] = potentials
    ephemeris_df['U_grav_field'] *= satellite_mass

    # Calculate total energy
    ephemeris_df['energy'] = ephemeris_df['U_spherical'] - ephemeris_df['U_grav_field']
    
    #add two energy diff columns (one for diff between Energy at t0 and t_i, and another for diff between t0 and t_i but not accounting for non-spherical potential)
    # energy_diff is the difference between the energy at the first point and the energy at the current point
    ephemeris_df['energy_diff'] = ephemeris_df['energy'] - ephemeris_df['energy'].iloc[0]
    ephemeris_df['energy_diff_kinetic'] = ephemeris_df['kinetic_energy'] - ephemeris_df['kinetic_energy'].iloc[0]
    ephemeris_df['energy_diff_rotational'] = ephemeris_df['earth_rotational_energy'] - ephemeris_df['earth_rotational_energy'].iloc[0]
    ephemeris_df['energy_diff_monopole'] = ephemeris_df['monopole'] - ephemeris_df['monopole'].iloc[0]
    ephemeris_df['energy_diff_grav_field'] = ephemeris_df['U_grav_field'] - ephemeris_df['U_grav_field'].iloc[0]
    return ephemeris_df

def main():
    sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        ephemeris_df = ephemeris_df.head(200) 
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
        ephemeris_df= energy(ephemeris_df, 64, 64)
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
        # plt.show()

        #now plot the raw energy values each in their own subplot
        plt.figure()
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
        plt.plot(ephemeris_df['MJD'], ephemeris_df['U_spherical'], label='Point Mass Related Potential Energy')
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
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy'], label='Total Energy')
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