from __future__ import annotations

import gengli
import matplotlib.pyplot as plt
import numpy as np

"""
Example script to generate whitened blip glitch in time domain.
"""


# Setting some configs
generator = gengli.glitch_generator("L1")
seed = 200
glitch_snr = 25
sampling_frequency = 2048

# Generate glitch
glitch_instance = generator.get_glitch(seed=seed, snr=glitch_snr, srate=sampling_frequency)
sample_times = np.arange(0, len(glitch_instance), 1) * (1 / sampling_frequency)

# Plot the glitch
fig, ax = plt.subplots(1, 1)
ax.plot(sample_times, glitch_instance)
ax.set_xlabel("Time [s]")
ax.grid()
ax.set_ylabel(r"Whitened Strain [$\sigma$]")
fig.savefig("glitch_instance_1.pdf")


print("Job complete!")
