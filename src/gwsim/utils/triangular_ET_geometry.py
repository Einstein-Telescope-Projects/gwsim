""" Script to compute the ET Triangular geometry at a given location """

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


def add_ET_Triangular_detector_at_location(E1_latitude: float, E1_longitude: float, E1_height: float, location_name: str, ETArmL: float = 10000) -> (Detector, Detector, Detector):
    """
    Add the triangular Einstein Telescope detector with PyCBC at a given location and height.
    The ET triangular configuration follows T1400308. The arms 1 and 2 of E1 are defined on the tangent plane at the E1 vertex position.
    The arm 1 has the same azimuth angle and altitude of the Virgo arm 1 in the local horizontal coordinate system center at the E1 vertex.

    Args:
        E1_latitude (float): E1 vertex latitude (rad)
        E1_longitude (float): E1 vertex longitude (rad)
        E1_height (float): E1 vertex height above the standard reference ellipsoidal earth (meters)
        location_name (str): Name of the ET location (e.g., Sardinia, EMR, Cascina, ...) for detector naming convention
        ETArmL (float, optional): ET arm length (meters). Default to 10000 meters.

    Returns:
        (Detector, Detector, Detector): pycbc.detector.Detector objects for E1, E2 and E3.
    """

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

    # E2 vertex location
    E2 = E1 + (ETArmL * E1Arm1)

    # Calculating rotation matrix to define E2 and E3 arms
    ux, uy, uz = E1normVec
    theta = 60
    cosT = np.cos(np.deg2rad(theta))
    sinT = np.sin(np.deg2rad(theta))
    RE1 = np.array([
        [cosT + ux**2 * (1 - cosT), ux * uy * (1 - cosT) - uz *
         sinT, ux * uz * (1 - cosT) + uy * sinT],
        [ux * uy * (1 - cosT) + uz * sinT, cosT + uy**2 *
         (1 - cosT), uy * uz * (1 - cosT) - ux * sinT],
        [ux * uz * (1 - cosT) - uy * sinT, uy * uz * (1 - cosT) +
         ux * sinT, cosT + uz**2 * (1 - cosT)]
    ])

    # Apply rotational matrix to E1 arm 1 vector to define E1 arm 2
    E1Arm2 = RE1 @ E1Arm1

    # E3 vertex location
    E3 = E1 + (ETArmL * E1Arm2)

    # E2 arm vectors
    E2Arm1 = -E1Arm1 + E1Arm2
    E2Arm2 = -E1Arm1

    # E3 arm vectors
    E3Arm1 = -E1Arm2
    E3Arm2 = -E2Arm1

    # Calculate the vertex positions in geodetic (ellipsoidal) coordinates
    E2_ellipsoid = np.array(pm.ecef2geodetic(*E2, deg=False))
    E3_ellipsoid = np.array(pm.ecef2geodetic(*E3, deg=False))

    # Calculate the unit vector angles (azimuth and altitude)
    E1Arm1Angles = get_unit_vector_angles(E1Arm1, E1_ellipsoid)
    E1Arm2Angles = get_unit_vector_angles(E1Arm2, E1_ellipsoid)
    E2Arm1Angles = get_unit_vector_angles(E2Arm1, E2_ellipsoid)
    E2Arm2Angles = get_unit_vector_angles(E2Arm2, E2_ellipsoid)
    E3Arm1Angles = get_unit_vector_angles(E3Arm1, E3_ellipsoid)
    E3Arm2Angles = get_unit_vector_angles(E3Arm2, E3_ellipsoid)

    # Add detectors with PyCBC
    add_detector_on_earth(
        name="E1_60deg_"+location_name,
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
    add_detector_on_earth(
        name="E2_60deg_"+location_name,
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
    add_detector_on_earth(
        name="E3_60deg_"+location_name,
        latitude=E3_ellipsoid[0],
        longitude=E3_ellipsoid[1],
        height=E3_ellipsoid[2],
        xangle=E3Arm1Angles[0],
        yangle=E3Arm2Angles[0],
        xaltitude=E3Arm1Angles[1],
        yaltitude=E3Arm2Angles[1],
        xlength=ETArmL,
        ylength=ETArmL
    )

    return Detector("E1_60deg_"+location_name), Detector("E2_60deg_"+location_name), Detector("E3_60deg_"+location_name)


if __name__ == "__main__":

    V1_lat = 0.7615118398400004
    V1_lon = 0.18333805213
    V1_height = 51.88399887084961

    Sardinia_lat = np.deg2rad(40 + 31/60)
    Sardinia_lon = np.deg2rad(9 + 25/60)

    EMR_lat = np.deg2rad(50 + 43/60 + 23/3600)
    EMR_lon = np.deg2rad(5 + 55/60 + 14/3600)

    ifo1, ifo2, ifo3 = add_ET_Triangular_detector_at_location(
        E1_latitude=V1_lat,
        E1_longitude=V1_lon,
        E1_height=V1_height,
        location_name="Cascina",
        ETArmL=10000
    )

    E1 = Detector('E1')
    E2 = Detector('E2')
    E3 = Detector('E3')

    # ===  COMPARE ANTENNA PATTERNS WITH PYCBC BUILT-IN ET DETECTOR  ===

    ra, dec = np.meshgrid(np.arange(0, np.pi*2.0, 0.01),
                          np.arange(-np.pi / 2.0, np.pi / 2.0, 0.01))
    ra = ra.flatten()
    dec = dec.flatten()
    pol = 0
    time = 1e10 + 8000

    fp_pycbc, fc_pycbc = E1.antenna_pattern(ra, dec, pol, time)
    fp, fc = ifo1.antenna_pattern(ra, dec, pol, time)

    print(np.allclose(fp, fp_pycbc, atol=1e-6, rtol=1e-5))
    print(np.allclose(fc, fc_pycbc, atol=1e-6, rtol=1e-5))

    color = (fp_pycbc - fp)**2 + (fc_pycbc - fc)**2

    plt.figure(figsize=(8, 21))
    ax1 = plt.subplot(311, projection="mollweide")
    ra[ra > np.pi] -= np.pi * 2.0
    sc = ax1.scatter(ra, dec, c=fp**2 + fc**2)
    plt.colorbar(sc, ax=ax1, orientation="horizontal", pad=0.1)
    ax1.grid(True)
    ax1.set_title('This code')

    ax2 = plt.subplot(312, projection="mollweide")
    ra[ra > np.pi] -= np.pi * 2.0
    sc = ax2.scatter(ra, dec, c=fp_pycbc**2 + fc_pycbc**2)
    plt.colorbar(sc, ax=ax2, orientation="horizontal", pad=0.1)
    ax2.grid(True)
    ax2.set_title('PyCBC')

    ax3 = plt.subplot(313, projection="mollweide")
    ra[ra > np.pi] -= np.pi * 2.0
    sc = ax3.scatter(ra, dec, c=color)
    plt.colorbar(sc, ax=ax3, orientation="horizontal", pad=0.1)
    ax3.grid(True)
    ax3.set_title('Difference')
    plt.show()

    # ===  COMPARE CONFIGURATIONS WITH PYCBC BUILT-IN ET DETECTOR  ===

    print("\n=== Difference with E1 PyCBC configuration ===")
    for key, val in ifo1.info.items():
        print(key, "\n", val - E1.info[key])

    print("\n=== Difference with E2 PyCBC configuration ===")
    for key, val in ifo2.info.items():
        print(key, "\n", val - E2.info[key])

    print("\n=== Difference with E3 PyCBC configuration ===")
    for key, val in ifo3.info.items():
        print(key, "\n", val - E3.info[key])

    print(f"\nNull Stream:\n", ifo1.response + ifo2.response + ifo3.response)
    print(f"Null Stream from PyCBC configuration:\n", E1.response +
          E2.response + E3.response)
