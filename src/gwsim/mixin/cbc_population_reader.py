"""Mixin for reading compact binary coalescence (CBC) population data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gwsim.mixin.population_reader import PopulationReaderMixin

logger = logging.getLogger("gwsim")

CBC_COMMON_PARAMETER_NAME_MAPPER = {
    # Masses
    "m1": "mass1",
    "mass_1": "mass1",
    "m2": "mass2",
    "mass_2": "mass2",
    "m1_source": "srcmass1",
    "mass1_source": "srcmass1",
    "mass_1_source": "srcmass1",
    "m2_source": "srcmass2",
    "mass2_source": "srcmass2",
    "mass_2_source": "srcmass2",
    # Spins
    "chi1x": "spin1x",
    "chi1y": "spin1y",
    "chi1z": "spin1z",
    "chi2x": "spin2x",
    "chi2y": "spin2y",
    "chi2z": "spin2z",
    # Tidal deformabilities
    "Lambda1": "lambda1",
    "Lambda2": "lambda2",
    # Luminosity Distance
    "dL": "distance",
    "luminosity_distance": "distance",
    # Coalescence phase
    "Phicoal": "coa_phase",
    # Inclination angle
    "iota": "inclination",
    # Coalescence time
    "tGPS": "tc",
    # Sky position
    "ra": "right_ascension",
    "dec": "declination",
    # Polarization angle
    "polarization": "polarization_angle",
    "psi": "polarization_angle",
    # Redshift
    "z": "redshift",
}


class CBCPopulationReaderMixin(PopulationReaderMixin):
    """Mixin class for reading compact binary coalescence (CBC) population data."""

    def __init__(
        self,
        population_file: str | Path,
        population_parameter_name_mapper: dict[str, str] | None = None,
        population_cache_dir: str | Path | None = None,
        population_download_timeout: int = 300,
        **kwargs,
    ):
        """Initialize the CBC population reader mixin.

        Args:
            population_file: Path or URL to the population data file.
            population_parameter_name_mapper: Optional dictionary to map population parameter names to standard names.
            population_cache_dir: Optional directory to cache downloaded population files.
            population_download_timeout: Timeout in seconds for downloading population files. Default is 300 seconds.
            **kwargs: Additional arguments absorbed by parent classes.
        """
        super().__init__(
            population_file=population_file,
            population_parameter_name_mapper=population_parameter_name_mapper,
            population_sort_by="tc",
            population_cache_dir=population_cache_dir,
            population_download_timeout=population_download_timeout,
            **kwargs,
        )

    def _population_get_default_parameter_name_mapper(self) -> dict[str, str]:
        """Get the default parameter name mapper for CBC populations.

        Returns:
            A dictionary mapping common CBC population parameter names to standard names.
        """
        return CBC_COMMON_PARAMETER_NAME_MAPPER.copy()

    def _population_post_process_population_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the population data after reading.

        Compute mass1 and mass2 from srcmass1, srcmass2 and redshift if they are not present in the data.

        Args:
            df: DataFrame containing the population data.

        Returns:
            DataFrame with post-processed population data.
        """
        if "mass1" not in df.columns:
            if "srcmass1" in df.columns and "redshift" in df.columns:
                df["mass1"] = df["srcmass1"] * (1 + df["redshift"])
                logger.info("Computed mass1 from srcmass1 and redshift.")
            else:
                raise ValueError("mass1 is not in population data, and cannot be computed from srcmass1 and redshift.")
        if "mass2" not in df.columns:
            if "srcmass2" in df.columns and "redshift" in df.columns:
                df["mass2"] = df["srcmass2"] * (1 + df["redshift"])
                logger.info("Computed mass2 from srcmass2 and redshift.")
            else:
                raise ValueError("mass2 is not in population data, and cannot be computed from srcmass2 and redshift.")
        return df
