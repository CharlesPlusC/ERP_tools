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
    """
    Compute the normalized associated Legendre polynomials P_nm(sin(phi)).
    """
    # Initialize an array for the polynomials
    pnm = np.zeros((n + 1, m + 1))
    
    # Compute unnormalized values using scipy's lpmn function
    # Note: lpmn returns all values up to n and m, so we trim the excess
    pnm_unnorm, _ = lpmn(m, n, sin_phi)
    pnm_unnorm = pnm_unnorm.T  # Transpose to align with our indexing

    # Apply the normalization factor to each term
    for ni in range(n + 1):
        for mi in range(min(ni, m) + 1):
            normalization_factor = np.sqrt(((2 * ni + 1) * math.factorial(ni - mi)) / (2 * math.factorial(ni + mi)))
            pnm[ni, mi] = normalization_factor * pnm_unnorm[ni, mi]

    return pnm

def compute_gravitational_potential(r, phi, lambda_, degree, order, P_nm_all, date):
    provider = GravityFieldFactory.getNormalizedProvider(degree, order)
    harmonics = provider.onDate(date)
    mu = provider.getMu()
    print(f"mu: {mu}")
    a = provider.getAe()
    print(f"a: {a}")
    V = 0.0
    # Start from n = 1 to exclude the central gravitational potential
    for n in range(1, degree + 1):
        for m in range(min(n, order) + 1):
            C_nm = harmonics.getNormalizedCnm(n, m)
            S_nm = harmonics.getNormalizedSnm(n, m)
            P_nm = P_nm_all[n, m]
            sectorial_term = (C_nm * np.cos(m * lambda_) + S_nm * np.sin(m * lambda_)) * P_nm
            V += ((a / r) ** n) * sectorial_term
    V *= mu / r # Multiply by the gravitational parameter and the radius
    return V

def energy(ephemeris_df):
    mu = Constants.WGS84_EARTH_MU
    omega_earth = 7.2921150e-5
    # Calculate spherical potential energy
    ephemeris_df['U_spherical'] = - 0.5 * (ephemeris_df['xv_ecef']**2 + ephemeris_df['yv_ecef']**2 + ephemeris_df['zv_ecef']**2) \
                                  - 0.5 * omega_earth**2 * (ephemeris_df['x_ecef']**2 + ephemeris_df['y_ecef']**2) \
                                  - mu / np.sqrt(ephemeris_df['x_ecef']**2 + ephemeris_df['y_ecef']**2 + ephemeris_df['z_ecef']**2)

    # Convert ECEF to latitude, longitude, and altitude
    converted_coords = np.vectorize(ecef_to_lla, signature='(),(),()->(),(),()')(ephemeris_df['x_ecef'], ephemeris_df['y_ecef'], ephemeris_df['z_ecef'])
    ephemeris_df[['lat', 'lon', 'alt']] = np.column_stack(converted_coords)

    # Pre-compute and cache unique associated Legendre polynomials
    unique_phis = np.unique(ephemeris_df['lat'].apply(np.radians))
    legendre_cache = {phi: associated_legendre_polynomials(36, 36, np.sin(phi)) for phi in unique_phis}

    # Compute non-spherical potential
    potentials = []
    for _, row in ephemeris_df.iterrows():
        phi_rad = np.radians(row['lat'])
        P_nm_all = legendre_cache[phi_rad]
        lambda_rad = np.radians(row['lon'])
        r = row['alt']
        print(f"r: {r}")
        date = datetime_to_absolutedate(row['UTC'])
        potential = compute_gravitational_potential(r, phi_rad, lambda_rad, 36, 36, P_nm_all, date)
        potentials.append(potential)
    ephemeris_df['U_non_spherical'] = potentials

    # Calculate total energy
    ephemeris_df['energy'] = ephemeris_df['U_spherical'] + ephemeris_df['U_non_spherical']
    return ephemeris_df

def main():
    sat_names_to_test = ["GRACE-FO-A"]
    # sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    for sat_name in sat_names_to_test:
        ephemeris_df = sp3_ephem_to_df(sat_name)
        #slice the dataframe to keep only the first 100 rows
        ephemeris_df = ephemeris_df.head(100)
        # take the UTC column and convert to mjd
        ephemeris_df['MJD'] = [utc_to_mjd(dt) for dt in ephemeris_df['UTC']]
        x_ecef, y_ecef, z_ecef, xv_ecef, yv_ecef, zv_ecef = ([] for _ in range(6))
        # Convert EME2000 to ITRF for each point
        for index, row in ephemeris_df.iterrows():
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

        ephemeris_df = energy(ephemeris_df)
        
        print(f"First five ephemeris data points for {sat_name}: \n{ephemeris_df.head()}")
        # Plot the energy
        plt.figure()
        plt.plot(ephemeris_df['MJD'], ephemeris_df['energy'], label='Potential with Non-Spherical')
        plt.plot(ephemeris_df['MJD'], ephemeris_df['U_spherical'], label='Potential with Spherical only ')
        # plt.plot(ephemeris_df['MJD'], ephemeris_df['U_non_spherical'], label='Non-Spherical Potential')
        plt.xlabel('MJD')
        plt.ylabel('Energy (J/kg)')
        plt.title(f'Energy for {sat_name}')
        plt.grid(True)
        plt.legend()
        title = f'Energy for {sat_name}'
        folder_path = "output/DensityInversion/OrbitEnergy"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(f"{folder_path}/{title}.png")
        # plt.show()

        print(f"First five ephemeris data points for {sat_name}: \n{ephemeris_df.head()}")


    # Constants
    radius_at_500km = 6378136.3 + 500000  # Earth's radius plus 500 km
    degree = 64
    order = 64
    date = AbsoluteDate(2000, 1, 1, 12, 0, 0.0, TimeScalesFactory.getUTC())

    # Define a sparser range of latitude and longitude for faster computation
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, 64)  # Reduced points for latitude
    longitudes = np.linspace(-np.pi, np.pi, 64)  # Reduced points for longitude

    # Initialize an array to store the potential values
    potential_map = np.zeros((len(latitudes), len(longitudes)))

    # Compute the potential at each point
    for i, phi in enumerate(latitudes):
        print(f"Computing latitude {i+1} out of {len(latitudes)}...")
        for j, lambda_ in enumerate(longitudes):
            potential_map[i, j] = compute_gravitational_potential(radius_at_500km, phi, lambda_, degree, order, date)

    print("Computation complete. Plotting...")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.contourf(longitudes * 180 / np.pi, latitudes * 180 / np.pi, potential_map, 100, cmap='viridis')
    plt.colorbar(label='Gravitational Potential (m^2/s^2)')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.title('Gravitational Potential at 500 km Altitude (Coarse Resolution)')
    plt.savefig("output/DensityInversion/GravitationalPotential.png")
if __name__ == "__main__":
    main()