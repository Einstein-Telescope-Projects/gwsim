"""Colored noise simulator implementation."""

from __future__ import annotations

from gwsim.noise.base import NoiseSimulator


class StationaryGaussianNoiseSimulator(NoiseSimulator):  # pylint: disable=duplicate-code
    """Stationary Gaussian noise simulator.

    Generates noise from a specified power spectral density.
    """

    def __init__(
        self,
        sampling_frequency: float = 4096,
        duration: float = 4,
        start_time: float = 0,
        max_samples: int | None = None,
        seed: int | None = None,
        detectors: list[str] | None = None,
        **kwargs,
    ):
        """Initialize stationary Gaussian noise simulator.

        Args:
            psd: Path to PSD file or numpy array with PSD values, or label of PSD.
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            duration: Duration of each segment in seconds. Default is 4.
            start_time: Start time in GPS seconds. Default is 0.
            max_samples: Maximum number of samples. None means infinite.
            seed: Random seed. If None, RNG is not initialized.
            detectors: List of detector names. Default is None.
            **kwargs: Additional arguments.
        """
        super().__init__(  # pylint: disable=duplicate-code
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
            seed=seed,
            detectors=detectors,
            **kwargs,
        )
