import netCDF4 as nc
import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.hipparchus.geometry.euclidean.threed import Vector3D, FieldVector3D
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces import PythonForceModel
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit
from org.orekit.orbits import OrbitType
from org.orekit.orbits import PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from orekit import JArray_double
from java.util import Collections
from java.util.stream import Stream
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants
from org.orekit.utils import IERSConventions
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.utils import IERSConventions
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from orekit import JArray_double
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.frames import FramesFactory

import orekit
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()

import numpy as np

from tools.utilities import jd_to_utc
from tools.data_processing import extract_hourly_ceres_data
from tools.TLE_tools import twoLE_parse, tle_convert, TLE_time, sgp4_prop_TLE

# class CustomForceModel(PythonForceModel):
#     print("CustomForceModel instantiated")
#     def __init__(self):
#         super().__init__()
#         print('CustomForceModel init')

#     def acceleration(self, fieldSpacecraftState, tArray):
#         """
#             Compute simple acceleration.

#         """
#         acceleration = Vector3D(1, 0, 0)
#         print("field spacecraft state:", fieldSpacecraftState)
#         print("tArray:", tArray)
#         # FieldVector3D = fieldSpacecraftState.getPosition()
#         # print("FieldVector3D:", FieldVector3D)
#         return acceleration

#     def addContribution(self, fieldSpacecraftState, fieldTimeDerivativesEquations):
#         print("addContribution called")
#         pass

#     def getParametersDrivers(self):
#         print("getParametersDrivers called")
#         return Collections.emptyList()

#     def init(self, fieldSpacecraftState, fieldAbsoluteDate):
#         print("init called")
#         pass

#     def getEventDetectors(self):
#         print("getEventDetectors called")
#         return Stream.empty()

# class SimpleConstantForceModel(PythonForceModel):
#     def __init__(self, acceleration):
#         super().__init__()
#         self.constant_acceleration = acceleration

#     def acceleration(self, spacecraftState, doubleArray):
#         # This method returns a constant acceleration
#         # regardless of the spacecraft state or parameters.
#         return self.constant_acceleration

#     def addContribution(self, spacecraftState, timeDerivativesEquations):
#         # Add the constant acceleration to the propagator
#         timeDerivativesEquations.addNonKeplerianAcceleration(self.acceleration(spacecraftState, None))

#     def getParametersDrivers(self):
#         # This simple model does not have any adjustable parameters
#         return Collections.emptyList()

#     def init(self, spacecraftState, absoluteDate):
#         # No specific initialization required for this simple model
#         pass

#     def getEventDetectors(self):
#         # No event detectors are used in this simple model
#         return Stream.empty()


class AltitudeDependentForceModel(PythonForceModel):
    def __init__(self, acceleration, threshold_altitude):
        super().__init__()
        self.constant_acceleration = acceleration
        self.threshold_altitude = threshold_altitude
        self.altitude = 0.0

    def acceleration(self, spacecraftState, doubleArray):
        # Compute the current altitude within the acceleration method
        pos = spacecraftState.getPVCoordinates().getPosition()
        current_altitude = pos.getNorm() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        
        # Apply the constant acceleration only if below the threshold altitude
        if current_altitude < self.threshold_altitude:
            print("current altitude:", current_altitude)
            print("applying constant acceleration:", self.constant_acceleration)
            return self.constant_acceleration
        else:
            # print("not applying constant acceleration")
            return Vector3D.ZERO

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

   #oneweb TLE
    TLE = "1 56719U 23068K   23330.91667824 -.00038246  00000-0 -10188+0 0  9993\n2 56719  87.8995  84.9665 0001531  99.5722 296.6576 13.15663544 27411"
    jd_start = 2460069.5000000  # Force time to be within the CERES dataset that I downloaded
    jd_end = jd_start + 1 # 1 day later
    dt = 60  # Seconds
    sgp4_ephem = sgp4_prop_TLE(TLE=TLE, jd_start=jd_start, jd_end=jd_end, dt=dt)
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
    positionTolerance = 1.0 
    tolerances = NumericalPropagator.tolerances(positionTolerance, 
                                                initialOrbit, 
                                                initialOrbit.getType())
    integrator = DormandPrince853Integrator(minStep, maxstep, 
        JArray_double.cast_(tolerances[0]),  # Double array of doubles needs to be casted in Python
        JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(initStep)
    satellite_mass = 500.0  # The models need a spacecraft mass, unit kg. 500kg is a complete guesstimate.

    #Initial state
    initialState = SpacecraftState(initialOrbit, satellite_mass) 
    propagator_num = NumericalPropagator(integrator)
    propagator_num.setOrbitType(OrbitType.CARTESIAN)
    propagator_num.setInitialState(initialState)

    print("initial altitude:", initialState.getA())

    # Add 10x10 gravity field
    gravityProvider = GravityFieldFactory.getNormalizedProvider(10, 10)
    propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    
    #### ADDITION OF ERP CUSTOM FORCE MODEL TO GO HERE ####
    # Example usage
    threshold_altitude = 1204959.0 
    const_acceleration = Vector3D(-10.0, -10.0, -10.0) # 1 m/s^2
    simple_force_model = AltitudeDependentForceModel(const_acceleration, threshold_altitude)
    propagator_num.addForceModel(simple_force_model)

    end_state = propagator_num.propagate(TLE_epochDate, TLE_epochDate.shiftedBy(3600.0 * 24))
    end_state

    print("Initial state:")
    print(initialState)
    print("Final state:")
    print(end_state)
    print("final altitude:", end_state.getA())
    print("final orbit:", end_state.getOrbit())