from __future__ import annotations

import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from scipy.interpolate import interp1d

from .generator import Generator


class CBCSignalGenerator(Generator):
    def __init__(self, detector_prefixes, population_df, waveform_arguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detectors = [Detector(prefix) for prefix in detector_prefixes]
        self.population_df = population_df
        self.index = 0

        ## Adding Earth rotation setup
        self.earth_rotation = 0
        if "earth_rotation" in waveform_arguments.keys():
            self.earth_rotation = waveform_arguments.pop("earth_rotation")
        self.earth_rotation_timestep = 100  # Unit: seconds
        if "earth_rotation_timestep" in waveform_arguments.keys():
            self.earth_rotation_timestep = waveform_arguments.pop("earth_rotation_timestep")

        if "time_dependent_timedelay" in waveform_arguments.keys():
            self.time_dependent_timedelay = waveform_arguments.pop("time_dependent_timedelay")

        self.waveform_arguments = waveform_arguments

    def next(self):
        if self.index < len(self.population_df):
            parameters = self.population_df.iloc[self.index]

            # Compute the hp and hc using pycbc
            hp, hc = get_td_waveform(**parameters, **self.waveform_arguments)

            # Compute the F+ and Fx
            self.data_array = []
            for i in range(len(self.detectors)):

                if self.earth_rotation:
                    t_gps = np.arange(
                        parameters["geocent_time"] - hp.duration,
                        parameters["geocent_time"] + self.earth_rotation_timestep,
                        self.earth_rotation_timestep,
                    )

                    # Antenna pattern barely changes in 100 s
                    # More accuracy with inetrpolation but expensive
                    repeat_count = int(self.earth_rotation_timestep / self.waveform_arguments["delta_t"])

                    Fp, Fc = self.detectors[i].antenna_pattern(
                        right_ascension=parameters["right_ascension"],
                        declination=parameters["declination"],
                        polarization=parameters["polarization_angle"],
                        t_gps=t_gps,
                    )
                    Fp = np.repeat(Fp, repeat_count)[: len(hp)]
                    Fc = np.repeat(Fc, repeat_count)[: len(hp)]

                    if self.time_dependent_timedelay:
                        tdelayArr = self.detectors[i].time_delay_from_earth_center(
                            parameters["right_ascension"], parameters["declination"], t_gps
                        )
                        tdelayArr = np.repeat(tdelayArr, repeat_count)[: len(hp)]

                        # Now evaluate h at u = t + tau(t)
                        u = hp.sample_times.data + tdelayArr

                        Hp = interp1d(hp.sample_times.data, hp.data, kind="cubic", fill_value="extrapolate")
                        Hc = interp1d(hc.sample_times.data, hc.data, kind="cubic", fill_value="extrapolate")

                        hp.data = Hp(u)
                        hc.data = Hc(u)

                else:
                    Fp, Fc = self.detectors[i].antenna_pattern(
                        right_ascension=parameters["right_ascension"],
                        declination=parameters["declination"],
                        polarization=parameters["polarization_angle"],
                        t_gps=parameters["geocent_time"],
                    )

                ht = Fp * hp + Fc * hc
                # Set the geocent time
                ht.start_time += parameters["geocent_time"]

                ht = TimeSeries.from_pycbc(ht)
                self.data_array.append(ht)
                self.index += 1
            return self.data_array

        else:
            raise StopIteration
