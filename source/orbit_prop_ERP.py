import netCDF4 as nc
import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.hipparchus.geometry.euclidean.threed import Vector3D, FieldVector3D
from org.orekit.forces import PythonForceModel, BoxAndSolarArraySpacecraft
from org.orekit.orbits import OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from orekit import JArray_double
from java.util import Collections
from java.util.stream import Stream
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.utils import IERSConventions
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.frames import FramesFactory
from org.orekit.forces.radiation import RadiationSensitive, KnockeRediffusedForceModel, IsotropicRadiationSingleCoefficient
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.utils import ParameterDriver
from org.orekit.utils import PVCoordinates
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
import numpy as np
import orekit
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()
import matplotlib.dates as mdates


from tools.utilities import find_nearest_index, jd_to_utc, lla_to_ecef, julian_day_to_ceres_time
from tools.data_processing import extract_hourly_ceres_data
from tools.TLE_tools import twoLE_parse, tle_convert, sgp4_prop_TLE
from tools.data_processing import extract_hourly_ceres_data ,combine_lw_sw_data, calculate_satellite_fov, is_within_fov_vectorized, sat_normal_surface_angle_vectorized

def HCL_diff(eph1: np.ndarray, eph2: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate the Height, Cross-Track, and Along-Track differences at each time step between two ephemerides.

    Parameters
    ----------
    eph1 : np.ndarray
        List or array of state vectors for a satellite.
    eph2 : np.ndarray
        List or array of state vectors for another satellite.

    Returns
    -------
    tuple
        Three lists, each containing the height, cross-track, and along-track differences at each time step.
    """
    #check that the starting conditions are the same
    # if (eph1[0][0:3]) != (eph2[0][0:3]) or (eph1[0][3:6]) != (eph2[0][3:6]):
    #     warnings.warn('The two orbits do not have the same starting conditions. Make sure this is intentional.')

    H_diffs = []
    C_diffs = []
    L_diffs = []

    for i in range(0, len(eph1), 1):
        #calculate the HCL difference at each time step
        
        r1 = np.array(eph1[i][0:3])
        r2 = np.array(eph2[i][0:3])
        
        v1 = np.array(eph1[i][3:6])
        v2 = np.array(eph2[i][3:6])
        
        unit_radial = r1/np.linalg.norm(r1)
        unit_cross_track = np.cross(r1, v1)/np.linalg.norm(np.cross(r1, v1))
        unit_along_track = np.cross(unit_radial, unit_cross_track)

        #put the three unit vectors into a matrix
        unit_vectors = np.array([unit_radial, unit_cross_track, unit_along_track])

        #subtract the two position vectors
        r_diff = r1 - r2

        #relative position in HCL frame
        r_diff_HCL = np.matmul(unit_vectors, r_diff)

        #height, cross track and along track differences
        h_diff = r_diff_HCL[0]
        c_diff = r_diff_HCL[1]
        l_diff = r_diff_HCL[2]

        H_diffs.append(h_diff)
        C_diffs.append(c_diff)
        L_diffs.append(l_diff)

    return H_diffs, C_diffs, L_diffs

def compute_erp_at_sc(ceres_time_index, radiation_data, sat_lat, sat_lon, sat_alt, horizon_dist):
    # Earth radius in meters
    R = 6371000.0
    # Latitude and longitude arrays
    lat = np.arange(-89.5, 90.5, 1)  # 1-degree step from -89.5 to 89.5
    lon = np.arange(-179.5, 180.5, 1)  # 1-degree step from -179.5 to 179.5

    # Mesh grid creation
    lon2d, lat2d = np.meshgrid(lon, lat)

    # FOV calculations
    fov_mask = is_within_fov_vectorized(sat_lat, sat_lon, horizon_dist, lat2d, lon2d)
    radiation_data_fov = np.ma.masked_where(~fov_mask, radiation_data[ceres_time_index, :, :])
    cos_thetas = sat_normal_surface_angle_vectorized(sat_alt, sat_lat, sat_lon, lat2d[fov_mask], lon2d[fov_mask])
    cosine_factors_2d = np.zeros_like(radiation_data_fov)
    cosine_factors_2d[fov_mask] = cos_thetas

    # Adjusting radiation data
    adjusted_radiation_data = radiation_data_fov * cosine_factors_2d

    # Satellite position and distance calculations
    sat_ecef = np.array(lla_to_ecef(sat_lat, sat_lon, sat_alt))
    ecef_x, ecef_y, ecef_z = lla_to_ecef(lat2d, lon2d, np.zeros_like(lat2d))
    ecef_pixels = np.stack((ecef_x, ecef_y, ecef_z), axis=-1)
    vector_diff = sat_ecef.reshape((1, 1, 3)) - ecef_pixels
    distances = np.linalg.norm(vector_diff, axis=2) * 1000  # Convert to meters

    # Radiation calculation
    delta_lat = np.abs(lat[1] - lat[0])
    delta_lon = np.abs(lon[1] - lon[0])
    area_pixel = R**2 * np.radians(delta_lat) * np.radians(delta_lon) * np.cos(np.radians(lat2d))  # Convert to m^2
    P_rad = adjusted_radiation_data * area_pixel / (np.pi * distances**2) # map of power flux in W/m^2 for each pixel
    print("sumj of all P_rad:", np.sum(P_rad))

    #need to convert the vector_diff to eci frame
    unit_vectors = vector_diff / distances[..., np.newaxis] * 1000 # Convert to unit vectors in meters
    radiation_force_vectors = unit_vectors * P_rad[..., np.newaxis] / (299792458)  # Convert to Newtons
    print("sum of all radiation_force_vectors:", np.sum(radiation_force_vectors))
    # Summing all force vectors
    total_radiation_force_vector = np.sum(radiation_force_vectors[fov_mask], axis=0)

    print("total_radiation_force_vector:", total_radiation_force_vector)

    # Satellite area in square meters
    satellite_area = 10.0

    # Total force due to radiation pressure for perfectly reflecting satellite
    total_force = total_radiation_force_vector * satellite_area
    print("total_force:", total_force)

    # Satellite mass in kilograms
    satellite_mass = 500.0

    # Calculate acceleration
    acceleration_vector = total_force / satellite_mass

    scalar_acc = np.linalg.norm(acceleration_vector)
    print("scalar_acc:", scalar_acc)

    down_vector = - sat_ecef / np.linalg.norm(sat_ecef)  # Normalize the satellite's position vector to get the down vector
    total_radiation_vector_normalized = total_radiation_force_vector / np.linalg.norm(total_radiation_force_vector)  # Normalize the total radiation vector

    cos_theta = np.dot(total_radiation_vector_normalized, down_vector)  # Cosine of the angle
    angle_radians = np.arccos(cos_theta)  # Angle in radians
    angle_degrees = np.rad2deg(angle_radians)  # Convert to degrees

    print("ERP angle:", angle_degrees)

    return np.array(acceleration_vector), scalar_acc, angle_degrees

class CERES_ERP_ForceModel(PythonForceModel):
    def __init__(self):
        super().__init__()
        self.altitude = 0.0
        self.scalar_acc_data = []
        self.erp_angle_data = []
        self.time_data = []  

    def acceleration(self, spacecraftState, doubleArray):
        # Compute the current altitude within the acceleration method
        pos = spacecraftState.getPVCoordinates().getPosition()
        alt_m = pos.getNorm() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        alt_km = alt_m / 1000.0
        horizon_dist = calculate_satellite_fov(alt_km)
        # Get the AbsoluteDate
        absolute_date = spacecraftState.getDate()
        # Convert AbsoluteDate to Python datetime
        date_time = absolutedate_to_datetime(absolute_date)
        # convert date_time to julianday
        jd_time = date_time.toordinal() + 1721425.5 + date_time.hour / 24 + date_time.minute / (24 * 60) + date_time.second / (24 * 60 * 60)
        # Get the ECI and ECEF frames
        ecef = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        # Transform the position vector to the ECEF frame
        pv_ecef = spacecraftState.getPVCoordinates(ecef).getPosition()
        # Define the Earth model
        earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
            Constants.WGS84_EARTH_FLATTENING, 
            ecef)
        # Convert ECEF coordinates to geodetic latitude, longitude, and altitude
        geo_point = earth.transform(pv_ecef, ecef, spacecraftState.getDate())
        # Extract latitude and longitude in radians
        latitude = geo_point.getLatitude()
        longitude = geo_point.getLongitude()
        # Convert radians to degrees if needed
        latitude_deg = np.rad2deg(latitude)
        longitude_deg = np.rad2deg(longitude)
        ceres_time = julian_day_to_ceres_time(jd_time)
        ceres_indices = find_nearest_index(ceres_times, ceres_time)

        erp_vec, scalar_acc, erp_angle = compute_erp_at_sc(ceres_indices, combined_radiation_data, latitude_deg, longitude_deg, alt_km, horizon_dist)
        # convert the erp_vec to ECI
        eci = FramesFactory.getEME2000()
        # Transform the position vector to the ECI frame
        transform = ecef.getTransformTo(eci, spacecraftState.getDate())
        erp_vec_ecef_pv = PVCoordinates(Vector3D(float(erp_vec[0]), float(erp_vec[1]), float(erp_vec[2])))
        # Transform the ERP vector to ECI frame
        erp_vec_eci_pv = transform.transformPVCoordinates(erp_vec_ecef_pv)
        # Extract the transformed vector components
        erp_vec_eci = erp_vec_eci_pv.getPosition()
        # Store the scalar acceleration and other data
        self.scalar_acc_data.append(scalar_acc)
        self.erp_angle_data.append(erp_angle)
        self.time_data.append(jd_time)
        # Create the Orekit vector for the ERP force in ECI frame
        orekit_erp_vec = Vector3D(erp_vec_eci.getX(), erp_vec_eci.getY(), erp_vec_eci.getZ())
        print("orekit_erp_vec components:", orekit_erp_vec.getX(), orekit_erp_vec.getY(), orekit_erp_vec.getZ())
        return orekit_erp_vec
    
    def addContribution(self, spacecraftState, timeDerivativesEquations):
        # Add the conditional acceleration to the propagator
        timeDerivativesEquations.addNonKeplerianAcceleration(self.acceleration(spacecraftState, None))

    def getParametersDrivers(self):
        # This model does not have any adjustable parameters
        return Collections.emptyList()

    def init(self, spacecraftState, absoluteDate):
        pos = spacecraftState.getPVCoordinates().getPosition()
        self.altitude = pos.getNorm() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        pass

    def getEventDetectors(self):
        # No event detectors are used in this model
        return Stream.empty()

if __name__ == "__main__":
    dataset_path = 'external/data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20230501-20230630.nc'  # Hourly data

    data = nc.Dataset(dataset_path)

    # Extract data from the CERES dataset
    ceres_times, lat, lon, lw_radiation_data, sw_radiation_data = extract_hourly_ceres_data(data)

    # Combine longwave and shortwave radiation data
    combined_radiation_data = combine_lw_sw_data(lw_radiation_data, sw_radiation_data)    

    #oneweb TLE
    # TLE = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993\n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"
    #starlink TLE
    TLE = "1 58214U 23170J   23345.43674150  .00003150  00000+0  17305-3 0  9997\n2 58214  42.9996 329.1219 0001662 255.3130 104.7534 15.15957346  7032"
    jd_start = 2460069.5000000  # Force time to be within the CERES dataset that I downloaded
    jd_end = jd_start + 1/24 # 1 hr later
    dt = 60  # Seconds
    # sgp4_ephem = sgp4_prop_TLE(TLE=TLE, jd_start=jd_start, jd_end=jd_end, dt=dt)
    # tle_time = TLE_time(TLE) ##The TLE I have is not actually in the daterange of the CERES dataset I downloaded so not using this now
    utc_start = jd_to_utc(jd_start)

    # Segment the time stamp into year, month, day, hour, minute, second components
    YYYY = int(utc_start.strftime("%Y"))
    MM = int(utc_start.strftime("%m"))
    DD = int(utc_start.strftime("%d"))
    H = int(utc_start.strftime("%H"))
    M = int(utc_start.strftime("%M"))
    S = float(utc_start.strftime("%S"))

    #convert JD start epoch to UTC and pass to AbsoluteDate
    utc = TimeScalesFactory.getUTC() #instantiate UTC time scale
    #also add modified Julian date
    TLE_epochDate = AbsoluteDate(YYYY, MM, DD, H, M, S, utc)
    print("orekit AbsoluteDate:", TLE_epochDate)

    #Convert the initial position and velocity to keplerian elements
    tle_dict = twoLE_parse(TLE)
    kep_elems = tle_convert(tle_dict)

    a = float(kep_elems['a'])*1000
    e = float(kep_elems['e'])
    i = float(kep_elems['i'])
    omega = float(kep_elems['arg_p'])
    raan = float(kep_elems['RAAN'])
    lv = (float(kep_elems['true_anomaly']))

    ## Instantiate the inertial frame where the orbit is defined
    inertialFrame = FramesFactory.getEME2000()

    # ## Orbit construction as Keplerian
    initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lv,
                                PositionAngleType.TRUE,
                                inertialFrame, TLE_epochDate, Constants.WGS84_EARTH_MU)
    print("initial orekit orbit:", initialOrbit)

    # #Set parameters for numerical propagation
    minStep = 0.001
    maxstep = 1000.0
    initStep = 60.0
    positionTolerance = 1e-5
    tolerances = NumericalPropagator.tolerances(positionTolerance, 
                                                initialOrbit, 
                                                initialOrbit.getType())
    integrator = DormandPrince853Integrator(minStep, maxstep, 
        JArray_double.cast_(tolerances[0]),  # Double array of doubles needs to be casted in Python
        JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(initStep)
    satellite_mass = 500.0
    prop_time = 3600.0 *2

    #Initial state
    initialState = SpacecraftState(initialOrbit, satellite_mass) 
    propagator_num = NumericalPropagator(integrator)
    propagator_num.setOrbitType(OrbitType.CARTESIAN)
    propagator_num.setInitialState(initialState)

    # Add 10x10 gravity field
    gravityProvider = GravityFieldFactory.getNormalizedProvider(10, 10)
    propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    
    ##### CERES ERP model ######
    # Initialize propagator with CERES ERP force model
    propagator_ceres_erp = NumericalPropagator(integrator)
    propagator_ceres_erp.setOrbitType(OrbitType.CARTESIAN)
    propagator_ceres_erp.setInitialState(initialState)
    propagator_ceres_erp.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    ceres_erp_force_model = CERES_ERP_ForceModel()
    propagator_ceres_erp.addForceModel(ceres_erp_force_model)

    # Propagate the orbit with CERES ERP force model
    ephemGen_CERES = propagator_ceres_erp.getEphemerisGenerator() # Get the ephemeris generator
    end_state_ceres = propagator_ceres_erp.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(prop_time))

    # Assuming erp_force_model.time_data is in a datetime format compatible with matplotlib
    time_data = ceres_erp_force_model.time_data
    scalar_acc_data = ceres_erp_force_model.scalar_acc_data
    erp_angle_data = ceres_erp_force_model.erp_angle_data

    ###### Knocke ERP model ######
    propagator_knocke_erp = NumericalPropagator(integrator)
    propagator_knocke_erp.setOrbitType(OrbitType.CARTESIAN)
    propagator_knocke_erp.setInitialState(initialState)
    propagator_knocke_erp.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))

    # Get the Sun as an ExtendedPVCoordinatesProvider
    sun = CelestialBodyFactory.getSun()

    # Create an instance of the IsotropicRadiationSingleCoefficient model
    spacecraft = IsotropicRadiationSingleCoefficient(10.0, 1.0) #area and Cr are both 1.0
    onedeg_in_rad = np.radians(1.0)
    angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
    knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
    propagator_knocke = NumericalPropagator(integrator)
    propagator_knocke.setOrbitType(OrbitType.CARTESIAN)
    propagator_knocke.setInitialState(initialState)
    propagator_knocke.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    propagator_knocke.addForceModel(knockeModel)
    ephemGen_knocke = propagator_knocke.getEphemerisGenerator() # Get the ephemeris generator
    end_state_with_knocke = propagator_knocke.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(prop_time))

    #### Now propagate with No ERP model ####
    propagator_no_erp = NumericalPropagator(integrator)
    propagator_no_erp.setOrbitType(OrbitType.CARTESIAN)
    propagator_no_erp.setInitialState(initialState)
    propagator_no_erp.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    ephemGen_no_erp = propagator_no_erp.getEphemerisGenerator() # Get the ephemeris generator
    end_state_no_erp = propagator_no_erp.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(prop_time))

    print("norm of 3D difference between No ERP and CERES end states:", end_state_no_erp.getPVCoordinates().getPosition().subtract(end_state_ceres.getPVCoordinates().getPosition()).getNorm())
    print("norm of 3D difference between No ERP and Knocke end states:", end_state_no_erp.getPVCoordinates().getPosition().subtract(end_state_with_knocke.getPVCoordinates().getPosition()).getNorm())
    print("norm of 3D difference between Knocke and CERES states:", end_state_with_knocke.getPVCoordinates().getPosition().subtract(end_state_ceres.getPVCoordinates().getPosition()).getNorm())
    
    ephemeris_generators = {
        'CERES ERP': ephemGen_CERES,
        'Knocke ERP': ephemGen_knocke,
        'No ERP': ephemGen_no_erp
    }

    def extract_position_velocity_data(ephemeris, initial_date, end_date, step, inertialFrame):
        times = []
        state_vectors = []  # Store position and velocity vectors

        current_date = initial_date
        while current_date.compareTo(end_date) <= 0:
            state = ephemeris.propagate(current_date)
            position = state.getPVCoordinates().getPosition().toArray()
            velocity = state.getPVCoordinates().getVelocity().toArray()
            state_vector = np.concatenate([position, velocity])  # Combine position and velocity

            times.append(current_date.durationFrom(initial_date))
            state_vectors.append(state_vector)

            current_date = current_date.shiftedBy(step)

        return times, state_vectors


    time_step = 20.0 
    state_vector_data = {}
    for ephem_name, ephem in ephemeris_generators.items():
        ephemeris = ephem.getGeneratedEphemeris()
        end_date = TLE_epochDate.shiftedBy(prop_time)
        times, state_vectors = extract_position_velocity_data(ephemeris, TLE_epochDate, end_date, time_step, inertialFrame)
        state_vector_data[ephem_name] = (times, state_vectors)

    # Compute HCL differences
    HCL_diffs = {}
    for name in ['CERES ERP', 'Knocke ERP']:
        _, state_vectors = state_vector_data[name]
        _, no_erp_state_vectors = state_vector_data['No ERP']
        H_diffs, C_diffs, L_diffs = HCL_diff(np.array(state_vectors), np.array(no_erp_state_vectors))
        HCL_diffs[name] = (H_diffs, C_diffs, L_diffs)

    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    # Set titles for each subplot
    axes[0].set_title('Height Differences Over Time')
    axes[1].set_title('Cross-Track Differences Over Time')
    axes[2].set_title('Along-Track Differences Over Time')

    # Set labels for x-axis and y-axis
    for ax in axes:
        ax.set_xlabel('Time (seconds from start)')
        ax.set_ylabel('Difference (meters)')
        ax.grid(True)

    # Plotting data for each model
    colors = {'CERES ERP': 'tab:green', 'Knocke ERP': 'tab:orange'}
    for name, diffs in HCL_diffs.items():
        H_diffs, C_diffs, L_diffs = diffs
        time_data = state_vector_data['No ERP'][0]

        axes[0].plot(time_data, H_diffs, label=f'{name} - No ERP', color=colors[name], linestyle='--')
        axes[1].plot(time_data, C_diffs, label=f'{name} - No ERP', color=colors[name], linestyle='--')
        axes[2].plot(time_data, L_diffs, label=f'{name} - No ERP', color=colors[name], linestyle='--')

    # Adding legends to each subplot
    for ax in axes:
        ax.legend()

    # Adjust layout to prevent overlap
    plt.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout

    # Show plot
    plt.show()

    # Calculate the average value of scalar_acc_data
    average_scalar_acc = np.mean(ceres_erp_force_model.scalar_acc_data)

    # Plotting the scalar acceleration data
    plt.figure(figsize=(10, 6))
    plt.plot(ceres_erp_force_model.time_data, ceres_erp_force_model.scalar_acc_data, label='Scalar Acceleration', color='tab:blue', linestyle='-')
    plt.xlabel('Time (seconds from start)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Scalar Acceleration Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Average Scalar Acceleration: {average_scalar_acc} m/s²")

    ##### altitude over time #####
    # # Time step for iterating over ephemeris (in seconds)
    # time_step = 60.0  # 1 minute, adjust as needed

    # # Extract data from each ephemeris
    # altitude_eccentricity_data = {}
    # for ephem_name, ephem in ephemeris_generators.items():
    #     ephemeris = ephem.getGeneratedEphemeris()
    #     end_date = TLE_epochDate.shiftedBy(prop_time)
    #     times, altitudes, eccentricities = extract_altitude_time_eccentricity_data(ephemeris, TLE_epochDate, end_date, time_step, inertialFrame)
    #     altitude_eccentricity_data[ephem_name] = (times, altitudes, eccentricities)

    # altitude_diff = {}
    # eccentricity_data = {}
    # for name in ['CERES ERP', 'Knocke ERP', 'No ERP']:
    #     times, altitudes, eccentricities = altitude_eccentricity_data[name]
    #     eccentricity_data[name] = eccentricities
    #     print(f"Data for {name}: {len(times)} time points")
    #     print("eccentricities:", eccentricities)
    #     if name != 'No ERP':
    #         _, no_erp_altitudes, _ = altitude_eccentricity_data['No ERP']
    #         altitude_diff[name] = [alt - no_erp_alt for alt, no_erp_alt in zip(altitudes, no_erp_altitudes)]

    # # Plotting eccentricity on secondary y-axis
    # fig, ax1 = plt.subplots(figsize=(10, 6))

    # # Plot settings for altitude difference
    # color_alt_diff_ceres = 'tab:green'
    # color_alt_diff_knocke = 'tab:orange'
    # ax1.set_xlabel('Time (seconds from start)')
    # ax1.set_ylabel('Altitude Difference (meters)', color=color_alt_diff_ceres)
    # ax1.plot(altitude_eccentricity_data['No ERP'][0], altitude_diff['CERES ERP'], label='CERES ERP - No ERP', linewidth=2, linestyle='--', color=color_alt_diff_ceres)
    # ax1.plot(altitude_eccentricity_data['No ERP'][0], altitude_diff['Knocke ERP'], label='Knocke ERP - No ERP', linewidth=2, linestyle='--', color=color_alt_diff_knocke)
    # ax1.tick_params(axis='y', labelcolor=color_alt_diff_ceres)
    # ax1.legend(loc='upper left')
    # ax1.grid(True)

    # fig.tight_layout()  
    # plt.title('Altitude Difference and Eccentricity Over Time')
    # plt.show()

    # # Extract data from each ephemeris
    # altitude_data = {}
    # for ephem_name, ephem in ephemeris_generators.items():
    #     ephemeris = ephem.getGeneratedEphemeris()
    #     end_date = TLE_epochDate.shiftedBy(prop_time)
    #     times, altitudes = extract_altitude_time_data(ephemeris, TLE_epochDate, end_date, time_step, inertialFrame)
    #     altitude_data[ephem_name] = (times, altitudes)
    #     # After extracting data
    #     for name, (times, altitudes) in altitude_data.items():
    #         print(f"Data for {name}: {len(times)} time points")
    #         print("all altitudes:", altitudes)
    #         if len(times) == 0:
    #             print(f"No data for {name}")

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # for name, (times, altitudes) in altitude_data.items():
    #     plt.plot(times, altitudes, label=name, linewidth=2, linestyle='--')

    # plt.xlabel('Time (seconds from start)')
    # plt.ylabel('Altitude (meters)')
    # plt.title('Altitude Comparison Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    ########################################