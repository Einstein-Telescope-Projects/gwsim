from __future__ import annotations

from gwpy.timeseries import TimeSeries
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform

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

        self.waveform_arguments = waveform_arguments

    def next(self):
        if self.index < len(self.population_df):
            parameters = self.population_df.iloc[self.index]

            # Compute the hp and hc using pycbc
            hp, hc = get_td_waveform(**parameters, **self.waveform_arguments)

            # Compute the F+ and Fx
            self.data_array = []
            for i in range(len(self.detectors)):
                Fp, Fc = self.detectors[i].antenna_pattern(
                    right_ascension=parameters["right_ascension"],
                    declination=parameters["declination"],
                    polarization=parameters["polarization_angle"],
                    t_gps=parameters["geocent_time"],
                )

                if self.earth_rotation:
                    t_gps = np.arange(
                        parameters["geocent_time"] - hp.duration,
                        parameters["geocent_time"] + self.earth_rotation_timestep,
                        self.earth_rotation_timestep,
                    )
                    repeat_count = int(self.earth_rotation_timestep / self.waveform_arguments["delta_t"])

                    Fp, Fc = self.detectors[i].antenna_pattern(
                        right_ascension=parameters["right_ascension"],
                        declination=parameters["declination"],
                        polarization=parameters["polarization_angle"],
                        t_gps=t_gps,
                    )
                    Fp = np.repeat(Fp, repeat_count)[: len(hp)]
                    Fc = np.repeat(Fc, repeat_count)[: len(hp)]

                ht = Fp * hp + Fc * hc
                # Set the geocent time
                ht.start_time += parameters["geocent_time"]

                ht = TimeSeries.from_pycbc(ht)

                # ht.t0 += parameters['geocent_time']
                self.data_array.append(ht)
                self.index += 1
            return self.data_array

        else:
            raise StopIteration
