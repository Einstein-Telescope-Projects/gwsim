"Script to generate a blip glitches population file from the GravitySpy database of LIGO L1 during O3 run"

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

if __name__ == "__main__":

    # Configuration
    start_time = Time("2030-01-01T00:00:00", format="isot", scale="utc")
    end_time = Time("2040-01-01T00:00:00", format="isot", scale="utc")

    OUT_FILE = "./blip_glitch_population.hdf5"
    ZENODO_ADDRESS = "https://zenodo.org/record/5649212/files"

    # Blip glitch rate
    RATE_PER_HOUR = 4
    RATE_PER_SEC = RATE_PER_HOUR / 3600

    # Load GravitySpy blip glitch population for LIGO Livingston O3 run
    df_l1_o3a = pd.read_csv(f"{ZENODO_ADDRESS}/L1_O3a.csv")
    df_l1_o3b = pd.read_csv(f"{ZENODO_ADDRESS}/L1_O3b.csv")
    df_l1_o3 = pd.concat([df_l1_o3a, df_l1_o3b])
    threshold = 0.9
    blip_df = df_l1_o3[df_l1_o3["Blip"] >= threshold]

    # Extract the SNR distribution
    snr_distr = blip_df["snr"]

    # Expected number of blip glitches
    total_duration_sec = (end_time - start_time).sec
    expected_events = RATE_PER_SEC * total_duration_sec
    N = np.random.poisson(expected_events)

    # Exponential waiting times
    inter_arrival_times = np.random.exponential(1.0 / RATE_PER_SEC, size=N)
    arrival_times = start_time + TimeDelta(inter_arrival_times.cumsum(), format="sec")

    # Keep only those within the time window
    mask = arrival_times < end_time
    arrival_times = arrival_times[mask]

    # Convert to GPS seconds
    gps_times = arrival_times.gps

    # Sample from SNR distribution
    snrs = np.random.choice(snr_distr, size=len(gps_times), replace=True)

    # Write output population file
    with h5py.File(OUT_FILE, "w") as f:
        f.create_dataset("gps_time", data=gps_times)
        f.create_dataset("snr", data=snrs)
