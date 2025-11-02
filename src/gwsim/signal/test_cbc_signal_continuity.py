import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
import tempfile
import os
import bilby
import matplotlib.pyplot as plt
from tqdm import tqdm

from gwsim.signal.cbc_signal import CBCSignal
from pycbc.types.timeseries import TimeSeries

# Parameters for the test
approximant = 'IMRPhenomD'
flow = 2.0
sampling_frequency = 4096.0
frame_duration = 4096.0
detector_names = ['E1']
earth_rotation = True
time_dependent_timedelay = True
start_time = 0.0

# BBH parameters
mass1 = 30.0
mass2 = 30.0
spin1x = 0.0
spin1y = 0.0
spin1z = 0.0
spin2x = 0.0
spin2y = 0.0
spin2z = 0.0
ra = 1.0
dec = 1.0
pol = 0.0
iota = 0.0
phase = 0.0
dl = 1000.0
redshift = 0.0
geocent_time = 4300

duration = bilby.gw.utils.calculate_time_to_merger(flow, mass1*(1 + redshift),
                                                   mass2*(1 + redshift),
                                                   safety=1.2)


# Create population DataFrame
data = {
    'mass_1': [mass1],
    'mass_2': [mass2],
    'geocent_time': [geocent_time],
    'luminosity_distance': [dl],
    'spin_1x': [spin1x],
    'spin_1y': [spin1y],
    'spin_1z': [spin1z],
    'spin_2x': [spin2x],
    'spin_2y': [spin2y],
    'spin_2z': [spin2z],
    'right_ascension': [ra],
    'declination': [dec],
    'polarization_angle': [pol],
    'iota': [iota],
    'phase': [phase],
    'redshift': [redshift],
    'duration': [duration]
}
df = pd.DataFrame(data)

# Save to temporary CSV
with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
    population_file = temp_file.name
df.to_csv(population_file, index=False)

# Instantiate the generator
gen = CBCSignal(
    detector_names=detector_names,
    population_file=population_file,
    approximant=approximant,
    flow=flow,
    sampling_frequency=sampling_frequency,
    duration=frame_duration,
    earth_rotation=earth_rotation,
    time_dependent_timedelay=time_dependent_timedelay,
    start_time=start_time,
)

# Generate two adjacent frames
frame1 = gen.next()
gen.update_state()
frame2 = gen.next()

# Create a pycbc TimeSeries
time_series1 = TimeSeries(initial_array=frame1[0], delta_t=1/sampling_frequency, epoch=start_time)
times1 = time_series1.get_sample_times()
time_series2 = TimeSeries(
    initial_array=frame2[0], delta_t=1/sampling_frequency, epoch=start_time+frame_duration)
times2 = time_series2.get_sample_times()

# Plot time series
fig, ax = plt.subplots()
ax.plot(times1, time_series1.numpy())
ax.plot(times2, time_series2.numpy())

fig, ax = plt.subplots()
ax.scatter(time_series1.sample_times[-10:], time_series1.data[-10:])
ax.scatter(time_series2.sample_times[:10], time_series2.data[:10])
plt.show()

# Concatenate time series
concat_time_series = np.concatenate((time_series1, time_series2))

# Generate reference long frame
ref_duration = 2 * frame_duration
ref_start_time = 0.0
ref_end_time = ref_duration
ref_data = np.zeros(int(ref_duration * sampling_frequency))

# Get parameters
parameters = df.iloc[0].to_dict()

# Get hp, hc
hp, hc = gen.get_polarization_at_time(parameters, gen.waveform_arguments)

# Inject into reference
det = gen.detectors[0]
ref_data = gen.inject_signal_in_frame(
    hp, hc, parameters, det, ref_data, sampling_frequency, ref_start_time, ref_end_time)

# Check if they match
if np.allclose(concat_time_series, ref_data, atol=1e-10):
    print("The signal is continuous across the two adjacent frames.")
else:
    print("The signal is NOT continuous across the two adjacent frames.")

# Clean up temp file
os.remove(population_file)
