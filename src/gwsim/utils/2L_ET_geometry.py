""" Script to compute the ET 2L aligned/misaligned geometry at a given location """

import numpy as np
import pymap3d as pm

from pycbc.detector import add_detector_on_earth, Detector

# TODO Solve import issue


def get_unit_vector_angles(unit_vector: np.ndarray, ellipsoid_position: np.ndarray) -> np.ndarray:
    """
    Compute the azimuthal angle and altitude (elevation) of a given unit vector relative to the local tangent plane at the specified ellipsoid position.

    Args:
        unit vector (np.ndarray): A 3-element array representing the unit vector in geocentric (ECEF) coordinates.
        ellipsoid_position (np.ndarray): A 3-element array specifying the reference position [latitude (rad), longitude (rad), height (meters)] on the Earth's ellipsoid

    Returns:
        (np.ndarray): A 2-element array [azimuth (rad), altitude (rad)], where:
            - azimuth is the angle from local north (0 to 2π, increasing eastward),
            - altitude is the elevation angle from the local horizontal plane (-π/2 to π/2).
    """
    lat, lon, _ = ellipsoid_position
    normal_vector = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])
    north_vector = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat)
    ])
    east_vector = np.array([
        -np.sin(lon),
        np.cos(lon),
        0
    ])
    altitude = np.arcsin(np.dot(unit_vector, normal_vector))
    azimuth = np.mod(np.arctan2(np.dot(unit_vector, east_vector),
                     np.dot(unit_vector, north_vector)), 2 * np.pi)

    return np.array([azimuth, altitude])


def add_ET_2L_detectors_at_location(E1_latitude: float,
                                    E1_longitude: float,
                                    E1_height: float,
                                    E2_latitude: float,
                                    E2_longitude: float,
                                    E2_height: float,
                                    alpha: float,
                                    E1_location_name: str,
                                    E2_location_name: str,
                                    ETArmL: float = 15000
                                    ) -> (Detector, Detector):
    """
    Add the 2L Einstein Telescope detectors with PyCBC at two given locations and heights, for a given relative angle alpha.
    The arms of the detectors are defined on the tangent plane at their vertex position.
    The arm 1 of E1 has the same azimuth angle and altitude of the Virgo arm 1 in the local horizontal coordinate system center at the E1 vertex.
    All the angles are measured clockwise in the local horizontal coordinate system (North to East).

    Args:
        E1_latitude (float): E1 vertex latitude (rad)
        E1_longitude (float): E1 vertex longitude (rad)
        E1_height (float): E1 vertex height above the standard reference ellipsoidal earth (meters)
        E2_latitude (float): E2 vertex latitude (rad)
        E2_longitude (float): E2 vertex longitude (rad)
        E2_height (float): E2 vertex height above the standard reference ellipsoidal earth (meters)
        alpha (float): Relative orientation angle alpha in radians. Alpha is defined as the relative angle between the two detectors, oriented w.r.t their local North.
        E1_location_name (str): Name of the E1 location (e.g., Sardinia, EMR, Cascina, ...) for detector naming convention
        E2_location_name (str): Name of the E1 location (e.g., Sardinia, EMR, Cascina, ...) for detector naming convention
        ETArmL (float, optional): ET arm length (meters). Default to 10000 meters.

    Returns:
        (Detector, Detector): pycbc.detector.Detector objects for E1, E2.
    """

    if alpha == 0:
        config = "Aligned"
    elif alpha == np.pi/4:
        config = "Misaligned"
    else:
        raise ValueError(
            "Only alpha = 0 (aligned configuration) and π/4 (misaligned configuration) are supported.")

    # === Detector E1 ===

    E1_ellipsoid = [E1_latitude, E1_longitude, E1_height]

    # E1 vertex location in geocentric (ECEF) coordinates
    E1 = np.array(pm.geodetic2ecef(*E1_ellipsoid, deg=False))

    # Normal vector to the tangent plane at the E1 vertex (ECEF coordinates)
    E1normVec = np.array([
        np.cos(E1_latitude) * np.cos(E1_longitude),
        np.cos(E1_latitude) * np.sin(E1_longitude),
        np.sin(E1_latitude)
    ])

    # Azimuth and altitude of Virgo arm 1 from LAL
    V1Arm1_az = 0.3391628563404083
    V1Arm1_alt = 0.0

    # Define the arm 1 of E1 with the same azimuth and altitude of the Virgo arm 1 (ECEF coordinates)
    E1Arm1 = np.array(pm.aer2ecef(
        az=V1Arm1_az,
        el=V1Arm1_alt,
        srange=1,
        lat0=E1_latitude,
        lon0=E1_longitude,
        alt0=E1_height,
        deg=False
    ) - E1)

    # Vector perpendicular to E1Arm1 on the same plane
    E1Arm2 = np.cross(E1Arm1, E1normVec)

    E1Arm1Angles = get_unit_vector_angles(E1Arm1, E1_ellipsoid)
    E1Arm2Angles = get_unit_vector_angles(E1Arm2, E1_ellipsoid)

    add_detector_on_earth(
        name=f"E1_{config}_"+E1_location_name,
        latitude=E1_ellipsoid[0],
        longitude=E1_ellipsoid[1],
        height=E1_ellipsoid[2],
        xangle=E1Arm1Angles[0],
        yangle=E1Arm2Angles[0],
        xaltitude=E1Arm1Angles[1],
        yaltitude=E1Arm2Angles[1],
        xlength=ETArmL,
        ylength=ETArmL
    )

    # === Detector E2 ===

    E2_ellipsoid = [E2_latitude, E2_longitude, E2_height]

    # E2 vertex location in geocentric (ECEF) coordinates
    E2 = np.array(pm.geodetic2ecef(*E2_ellipsoid, deg=False))

    # Normal vector to the tangent plane at the E2 vertex (ECEF coordinates)
    E2normVec = np.array([
        np.cos(E2_latitude) * np.cos(E2_longitude),
        np.cos(E2_latitude) * np.sin(E2_longitude),
        np.sin(E2_latitude)
    ])

    # Define the arm 1 of E2 with the same azimuth and altitude of the Virgo arm 1 + alpha (ECEF coordinates)
    E2Arm1_az = V1Arm1_az + alpha
    E2Arm1 = np.array(pm.aer2ecef(
        az=E2Arm1_az,
        el=V1Arm1_alt,
        srange=1,
        lat0=E2_latitude,
        lon0=E2_longitude,
        alt0=E2_height,
        deg=False
    ) - E2)

    # Vector perpendicular to E2Arm1 on the same plane
    E2Arm2 = np.cross(E2Arm1, E2normVec)

    E2Arm1Angles = get_unit_vector_angles(E2Arm1, E2_ellipsoid)
    E2Arm2Angles = get_unit_vector_angles(E2Arm2, E2_ellipsoid)

    add_detector_on_earth(
        name=f"E2_{config}_"+E2_location_name,
        latitude=E2_ellipsoid[0],
        longitude=E2_ellipsoid[1],
        height=E2_ellipsoid[2],
        xangle=E2Arm1Angles[0],
        yangle=E2Arm2Angles[0],
        xaltitude=E2Arm1Angles[1],
        yaltitude=E2Arm2Angles[1],
        xlength=ETArmL,
        ylength=ETArmL
    )

    return Detector(f"E1_{config}_"+E1_location_name), Detector(f"E2_{config}_"+E2_location_name)


if __name__ == "__main__":

    V1_lat = 0.7615118398400004
    V1_lon = 0.18333805213
    V1_height = 51.88399887084961

    Sardinia_lat = np.deg2rad(40 + 31/60)
    Sardinia_lon = np.deg2rad(9 + 25/60)

    EMR_lat = np.deg2rad(50 + 43/60 + 23/3600)
    EMR_lon = np.deg2rad(5 + 55/60 + 14/3600)

    ifo1, ifo2, = add_ET_2L_detectors_at_location(E1_latitude=Sardinia_lat,
                                                  E1_longitude=Sardinia_lon,
                                                  E1_height=V1_height,
                                                  E2_latitude=EMR_lat,
                                                  E2_longitude=EMR_lon,
                                                  E2_height=V1_height,
                                                  alpha=0,
                                                  E1_location_name='Sardinia',
                                                  E2_location_name='EMR',
                                                  ETArmL=15000
                                                  )
