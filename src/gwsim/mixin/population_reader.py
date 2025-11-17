"""Population reader mixin for simulators."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class PopulationReaderMixin:  # pylint: disable=too-few-public-methods
    """A mixin class to read population files for GW signal simulators."""

    population_counter = 0

    def __init__(self, population_file: str | Path, population_file_type: str = "pycbc", **kwargs):
        """Initialize the PopulationReaderMixin.

        Args:
            population_file: Path to the population file.
            population_file_type: Type of the population file. Default is 'pycbc'.
        """
        super().__init__(**kwargs)
        self.population_file = Path(population_file)
        self.population_file_type = population_file_type
        if not self.population_file.is_file():
            raise FileNotFoundError(f"Population file {self.population_file} does not exist.")

        if population_file_type == "pycbc":
            self.population_data = self._read_pycbc_population_file(self.population_file, **kwargs)
        else:
            raise ValueError(f"Unsupported population file type: {population_file_type}")

    def _read_pycbc_population_file(  # pylint: disable=unused-argument
        self, file_name: str | Path, **kwargs
    ) -> pd.DataFrame:
        """Read a pycbc population file in HDF5 format.

        Args:
            file_name: Path to the pycbc population file.
            **kwargs: Additional arguments (not used currently).

        Returns:
            A pandas DataFrame containing the population data.
        """
        # Load the pycbc population file and create a pandas DataFrame

        with h5py.File(file_name, "r") as f:
            data = {key: value[()] for key, value in f.items()}

            # Create a DataFrame
            population_data = pd.DataFrame(data)

            # Save the attributes to metadata
            attrs = dict(f.attrs.items())
            # If there is any numpy array in attrs, convert it to list
            for key, value in attrs.items():
                if isinstance(value, np.ndarray):
                    attrs[key] = value.tolist()
            self._population_metadata = attrs

        # Order the DataFrame by the coalescence time 'tc'
        return population_data.sort_values(by="tc").reset_index(drop=True)

    def get_next_injection_parameters(self) -> dict[str, float | int] | None:
        """Get the next set of injection parameters from the population.

        Returns:
            A dictionary of injection parameters for the next event,
                or None if all events have been used.
        """
        if self.population_counter < len(self.population_data):
            output = self.population_data.iloc[self.population_counter].to_dict()
            self.population_counter += 1
        else:
            output = None
        return output

    def get_injection_parameter_keys(self) -> list[str]:
        """Get the list of injection parameter keys from the population data.

        Returns:
            A list of strings representing the injection parameter keys.
        """
        if not self.population_data.empty:
            output = list(self.population_data.columns)
        else:
            output = []
        return output

    @property
    def metadata(self) -> dict:
        """Get metadata including population file information.

        Returns:
            Dictionary containing metadata.
        """
        metadata = {
            "population": {
                "population_file": str(self.population_file),
                "population_file_type": self.population_file_type,
            }
        }
        if hasattr(self, "_population_metadata"):
            metadata["population"].update(self._population_metadata)
        return metadata
