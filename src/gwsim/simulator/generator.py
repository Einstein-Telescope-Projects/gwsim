from __future__ import annotations

import numpy as np


class Generator:
    def __init__(self, verbose=True, *args, **kwargs):
        self.verbose = verbose
        self.data_array = []

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __lt__(self, generator):
        if len(self.data_array) != 0:
            return (self.start_time < generator.end_time).any()
        else:
            return False

    def is_left_disjoint(self, generator):
        if len(self.data_array) != 0:
            return (self.end_time < generator.start_time).all()
        else:
            return False

    def is_right_disjoint(self, generator):
        if len(self.data_array) != 0:
            return (self.start_time > generator.end_time).all()
        else:
            return False

    def is_left_overlapping(self, generator):
        if len(self.data_array) != 0:
            return np.logical_and(self.end_time >= generator.start_time, self.start_time < generator.start_time).any()
        else:
            return False

    def is_right_overlapping(self, generator):
        if len(self.data_array) != 0:
            return np.logical_and(self.start_time <= generator.end_time and self.end_time > generator.end_time).any()
        else:
            return False

    def is_containing(self, generator):
        if len(self.data_array) != 0:
            return np.logical_and(self.start_time <= generator.start_time and self.end_time >= generator.end_time).all()
        else:
            return False

    def is_contained_in(self, generator):
        if len(self.data_array) != 0:
            return np.logical_and(self.start_time >= generator.start_time, self.end_time <= generator.end_time).all()
        else:
            return False

    @property
    def start_time_array(self):
        return np.array([data.t0 for data in self.data_array])

    @property
    def end_time_array(self):
        return np.array([data.t0 + data.duration for data in self.data_array])

    @property
    def duration_array(self):
        return np.array([data.duration for data in self.data_array])

    def next(self):
        return self.data_array

    def save_state(self, fname):
        pass

    def load_state(self, fname):
        pass

    def save_data(self):
        pass
