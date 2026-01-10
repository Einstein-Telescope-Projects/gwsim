"""Population reader mixin for simulators."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import h5py
import numpy as np
import pandas as pd
import yaml

from gwsim.utils.download import download_file

logger = logging.getLogger("gwsim")


class PopulationIterationState:  # pylint: disable=too-few-public-methods
    """Manages state for population file iteration with checkpoint support."""

    def __init__(self, checkpoint_file: str | Path | None = None, encoding: str = "utf-8") -> None:
        self.checkpoint_file = checkpoint_file
        self.encoding = encoding
        self.current_index = 0
        self.injected_indices: list[int] = []
        self.segment_map: dict[int, list[int]] = {}
        self._load_checkpoint()

    @property
    def checkpoint_file(self) -> Path | None:
        """Get the checkpoint file path.

        Returns:
            Path to the checkpoint file or None if not set.
        """
        return self._checkpoint_file

    @checkpoint_file.setter
    def checkpoint_file(self, value: str | Path | None) -> None:
        """Set the checkpoint file path.

        Args:
            value: Path to the checkpoint file or None to unset.
        """
        if value is None:
            self._checkpoint_file = None
        else:
            self._checkpoint_file = Path(value)

    def _load_checkpoint(self) -> None:
        if self.checkpoint_file and self.checkpoint_file.is_file():
            try:
                with open(self.checkpoint_file, encoding=self.encoding) as f:
                    data = yaml.safe_load(f)["population"]
                    self.current_index = data.get("current_index", 0)
                    self.injected_indices = data.get("injected_indices", [])
                    self.segment_map = data.get("segment_map", {})
                logger.info(
                    "Loaded checkpoint: current_index=%s, injected=%s",
                    self.current_index,
                    self.injected_indices,
                )
            except (OSError, yaml.YAMLError, KeyError) as e:
                logger.warning("Failed to load checkpoint %s: %s. Starting fresh.", self.checkpoint_file, e)


class PopulationReaderMixin:  # pylint: disable=too-many-instance-attributes
    """A mixin class to read population files for GW signal simulators."""

    population_counter = 0

    def __init__(
        self,
        population_file: str | Path,
        population_parameter_name_mapper: dict[str, str] | None = None,
        population_sort_by: str | None = None,
        population_cache_dir: str | Path | None = None,
        population_download_timeout: int = 300,
        **kwargs,
    ):
        """Initialize the PopulationReaderMixin.

        Args:
            population_file: Path to the population file.
            population_parameter_name_mapper: A dictionary to map original parameter names to standardized names.
            population_sort_by: Column name to sort the population data by. If None, no sorting is applied.
            population_cache_dir: Directory to cache downloaded population files. If None, uses default.
            population_download_timeout: Timeout in seconds for downloading population files.
            **kwargs: Additional arguments.

        Raises:
            FileNotFoundError: If the population file does not exist.
        """
        super().__init__(**kwargs)

        # Initialize the population data to None
        self._population_data = None

        # Store the population file information
        self.population_file = population_file

        # Set the cache directory
        self.population_cache_dir = population_cache_dir

        # Set the download timeout
        self.population_download_timeout = population_download_timeout

        # Get the default parameter mapper
        default_parameter_name_mapper = self._population_get_default_parameter_name_mapper()

        # Merge default mapper with instance override
        self.population_parameter_name_mapper = {
            **default_parameter_name_mapper,
            **(population_parameter_name_mapper or {}),
        }

        self.population_sort_by = population_sort_by

        # Apply the parameter name mapper after reading the population data
        self.population_data = self._population_apply_parameter_name_mapper(
            self.population_data, self.population_parameter_name_mapper
        )

        # Perform post-processing on the population data
        self.population_data = self._population_post_process_population_data(self.population_data)

        # Sort the population data if requested
        if self.population_sort_by is not None and self.population_sort_by in self.population_data.columns:
            self.population_data = self.population_data.sort_values(by=self.population_sort_by).reset_index(drop=True)

    @property
    def population_file(self) -> Path:
        """Get the population file path.

        Returns:
            Path to the population file.
        """
        return self._population_file

    @population_file.setter
    def population_file(self, value: str | Path) -> None:
        """Set the population file path.

        Args:
            value: Path to the population file.
        """
        # Check whether this is a URL or a local file path

        self._population_file_is_url = urlparse(str(value)).scheme in ("http", "https")
        if self._population_file_is_url:
            self._population_file_url = str(value)
        else:
            self._population_file_url = None
            self._population_file = Path(value)

        if not self._population_file_is_url and not self._population_file.is_file():
            raise FileNotFoundError(f"Population file {self._population_file} does not exist.")

    @property
    def population_cache_dir(self) -> Path:
        """Get the population cache directory.

        Returns:
            Path to the population cache directory.
        """
        return self._population_cache_dir

    @population_cache_dir.setter
    def population_cache_dir(self, value: str | Path | None) -> None:
        """Set the population cache directory.
        If None, uses the default directory at ~/.gwsim/population.

        Args:
            value: Path to the population cache directory or None to unset.
        """
        if value is None:
            self._population_cache_dir = Path.home() / ".gwsim" / "population"
            logger.info("population_cache_dir not set. Using default: %s", self._population_cache_dir)
        else:
            self._population_cache_dir = Path(value)
        if not self._population_cache_dir.exists():
            self._population_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created population cache directory: %s", self._population_cache_dir)

    @property
    def population_data(self) -> pd.DataFrame:
        """Get the population data as a pandas DataFrame.

        Returns:
            A pandas DataFrame containing the population data.
        """
        if self._population_data is None:
            if self._population_file_is_url:
                if self._population_file_url is not None:
                    self._population_data = self._population_download_population_file(
                        url=self._population_file_url,
                        outdir=self.population_cache_dir,
                        timeout=self.population_download_timeout,
                    )
                else:
                    raise ValueError("Unexpected Error. Population file URL is not set.")
            else:
                self._population_data = self._population_read_population_file(file_name=self._population_file)
        return self._population_data

    @population_data.setter
    def population_data(self, value: pd.DataFrame) -> None:
        """Set the population data.

        Args:
            value: A pandas DataFrame containing the population data.
        """
        self._population_data = value

    def _population_download_population_file(self, url: str, outdir: Path | str, timeout: int) -> pd.DataFrame:
        """Download the population file from a URL and read it.

        Args:
            url: The URL to download the population file from.
            outdir: The output directory to save the downloaded file.
            timeout: Timeout in seconds for the download operation.

        Returns:
            A pandas DataFrame containing the population data.
        """
        self._population_file = download_file(
            url=url,
            outdir=outdir,
            overwrite=False,
            allow_existing=True,
            dest_path_from_hashed_url=True,
            timeout=timeout,
        )

        return self._population_read_population_file(file_name=self._population_file)

    def _population_read_population_file(self, file_name: str | Path, **kwargs) -> pd.DataFrame:
        """Read the population file based on its type.
        Only supports '.hdf5', '.h5', '.hdf', and '.csv' formats for now.

        Args:
            file_name: Path to the population file.
            **kwargs: Additional arguments.

        Returns:
            A pandas DataFrame containing the population data.

        Raises:
            ValueError: If the population file format is unsupported.
        """
        # Check the suffix to determine file type
        file_name = Path(file_name)

        if file_name.suffix.lower() in [".hdf5", ".h5", ".hdf"]:
            return self._population_read_hdf5_population_file(file_name, **kwargs)
        if file_name.suffix.lower() == ".csv":
            return self._population_read_csv_population_file(file_name, **kwargs)
        raise ValueError(
            f"Unsupported population file format: {file_name.suffix}. "
            "Supported formats are .hdf5, .h5, .hdf, and .csv."
        )

    def _population_read_hdf5_population_file(self, file_name: str | Path, **kwargs) -> pd.DataFrame:
        """Read a generic HDF5 population file.

        Args:
            file_name: Path to the HDF5 population file.
            **kwargs: Additional arguments (not used currently).

        Returns:
            A pandas DataFrame containing the population data.
        """
        with h5py.File(file_name, "r", **kwargs) as f:
            data = {key: value[()] for key, value in f.items()}
            # Save attributes to metadata
            attrs = dict(f.attrs.items())
            # Convert numpy arrays to lists
            for key, value in attrs.items():
                if isinstance(value, np.ndarray):
                    attrs[key] = value.tolist()
            self._population_metadata = attrs
        return pd.DataFrame(data)

    def _population_read_csv_population_file(self, file_name: str | Path, **kwargs) -> pd.DataFrame:
        """Read a generic CSV population file.

        Args:
            file_name: Path to the CSV population file.
            **kwargs: Additional arguments (not used currently).

        Returns:
            A pandas DataFrame containing the population data.
        """
        return pd.read_csv(file_name, **kwargs)

    def _population_get_default_parameter_name_mapper(self) -> dict[str, str]:
        """Get the default parameter name mapper based on population_file_type.

        Returns:
            A dictionary mapping original parameter names to standardized names.
        """
        return {}

    def _population_apply_parameter_name_mapper(self, df: pd.DataFrame, parameter_name_mapper: dict) -> pd.DataFrame:
        """Apply the parameter name mapper to the DataFrame columns.

        Args:
            df: The original DataFrame.

        Returns:
            A DataFrame with renamed columns.
        """
        return df.rename(columns=parameter_name_mapper)

    def _population_post_process_population_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the population data DataFrame.

        Args:
            df: The original DataFrame.

        Returns:
            A post-processed DataFrame.
        """
        return df

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
        output = list(self.population_data.columns) if not self.population_data.empty else []
        return output

    @property
    def metadata(self) -> dict:
        """Get metadata including population file information.

        Returns:
            Dictionary containing metadata.
        """
        metadata = {
            "population_reader": {
                "arguments": {
                    "population_file": str(self.population_file),
                    "population_parameter_name_mapper": self.population_parameter_name_mapper,
                    "population_sort_by": self.population_sort_by,
                    "population_cache_dir": self.population_cache_dir,
                    "population_download_timeout": self.population_download_timeout,
                },
                "population_metadata": {
                    **getattr(self, "_population_metadata", {}),
                },
            }
        }
        return metadata
