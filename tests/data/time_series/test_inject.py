"""Unit tests for the inject module."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwsim.data.time_series.inject import inject


class TestInjectBasic:
    """Test basic injection functionality."""

    def test_inject_aligned_grids_simple(self):
        """Test injecting two time series with aligned time grids."""
        # Create target timeseries
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        # Create other timeseries with aligned grid
        other = GWpyTimeSeries(
            np.ones(50),
            t0=2.5,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Result should have the same length as timeseries
        assert len(result) == len(timeseries)
        # Result should have ones in the injected region and zeros elsewhere
        assert np.all(result.value[:25] == 0)  # Before injection
        assert np.all(result.value[25:75] == 1)  # Injected region
        assert np.all(result.value[75:] == 0)  # After injection

    def test_inject_with_addition(self):
        """Test that injection adds values correctly."""
        timeseries = GWpyTimeSeries(
            np.ones(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30) * 2,
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # In the injected region, values should be 1 + 2 = 3
        expected_injected_value = 3
        assert np.all(result.value[20:50] == expected_injected_value)
        # Outside injected region, values should remain 1
        assert np.all(result.value[:20] == 1)
        assert np.all(result.value[50:] == 1)

    def test_inject_preserves_metadata(self):
        """Test that injection preserves time series metadata."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=10.0,
            dt=0.01,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(20),
            t0=10.05,
            dt=0.01,
            unit="m",
        )

        result = inject(timeseries, other)

        # Check metadata preservation
        assert result.t0 == timeseries.t0
        assert result.dt == timeseries.dt
        assert result.unit == timeseries.unit
        assert len(result) == len(timeseries)

    def test_inject_sampling_frequency_mismatch(self):
        """Test that injection raises error for incompatible sampling frequencies."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(50),
            t0=2.5,
            dt=0.2,  # Different dt
            unit="m",
        )

        with pytest.raises((RuntimeError, ValueError)):  # GWpy's is_compatible raises RuntimeError or ValueError
            inject(timeseries, other)


class TestInjectOffset:
    """Test injection with time offset (non-aligned grids)."""

    def test_inject_with_fractional_offset_interpolate(self):
        """Test injection with fractional sample offset using interpolation."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        # Create other with a fractional offset (0.05s = 0.5 samples)
        other = GWpyTimeSeries(
            np.ones(30),
            t0=2.55,  # 2.55 = 2.5 + 0.05 (fractional offset)
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other, interpolate_if_offset=True)

        # Result should have been interpolated
        assert len(result) == len(timeseries)
        # The injected region should have non-zero values
        assert np.any(result.value[25:55] != 0)

    def test_inject_with_fractional_offset_no_interpolate(self):
        """Test injection with fractional offset without interpolation."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30),
            t0=2.55,  # Fractional offset
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other, interpolate_if_offset=False)

        # Without interpolation, should return original timeseries unchanged
        assert np.all(result.value == timeseries.value)


class TestInjectEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_inject_other_entirely_before_timeseries(self):
        """Test injection when other ends before timeseries starts."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=10.0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(20),
            t0=0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Should return result with same properties as timeseries
        assert len(result) == len(timeseries)
        assert result.t0 == timeseries.t0

    def test_inject_other_entirely_after_timeseries(self):
        """Test injection when other starts after timeseries ends."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(20),
            t0=20.0,  # Starts after timeseries ends at 10.0
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Should return result with same properties
        assert len(result) == len(timeseries)
        assert result.t0 == timeseries.t0

    def test_inject_other_at_start(self):
        """Test injection when other aligns with start of timeseries."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(20),
            t0=0,  # Starts at same time
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        assert np.all(result.value[:20] == 1)
        assert np.all(result.value[20:] == 0)

    def test_inject_other_at_end(self):
        """Test injection when other aligns with end of timeseries."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(20),
            t0=8.0,  # Ends at same time as timeseries
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        assert np.all(result.value[:80] == 0)
        assert np.all(result.value[80:] == 1)

    def test_inject_other_spans_entire_timeseries(self):
        """Test injection when other spans entire timeseries."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        assert np.all(result.value == 1)

    def test_inject_single_sample(self):
        """Test injection with single sample."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.array([5.0]),
            t0=2.5,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Single sample should be injected at index 25
        expected_value = 5.0
        assert result.value[25] == expected_value
        assert np.sum(result.value) == expected_value

    def test_inject_other_extends_beyond_timeseries(self):
        """Test injection when other extends beyond timeseries end."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(150),  # Longer than timeseries
            t0=0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # All of timeseries should be filled with ones
        assert len(result) == len(timeseries)
        assert np.all(result.value == 1)

    def test_inject_partial_overlap_start(self):
        """Test injection with partial overlap at start."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30),
            t0=-0.5,  # Starts before timeseries
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Overlap starts from beginning of timeseries
        overlap_len = 25  # 0.5 / 0.1 * 0.5 + remaining
        assert np.any(result.value[:overlap_len] != 0)

    def test_inject_partial_overlap_end(self):
        """Test injection with partial overlap at end."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30),
            t0=8.5,  # Extends beyond timeseries end
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Should inject what overlaps
        assert np.any(result.value[85:] != 0)


class TestInjectNumericalStability:
    """Test numerical stability and precision."""

    def test_inject_preserves_sampling_frequency_precision(self):
        """Test that injection preserves sampling frequency without precision loss."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.01,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(50),
            t0=0.25,
            dt=0.01,
            unit="m",
        )

        result = inject(timeseries, other)

        # Sampling frequency should be preserved exactly
        assert result.dt == timeseries.dt
        assert result.sample_rate == timeseries.sample_rate

    def test_inject_accumulation(self):
        """Test that multiple injections accumulate correctly."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        # Inject first signal (ones from t=0 to t=2.0, indices 0-20)
        other1 = GWpyTimeSeries(
            np.ones(20),
            t0=0,
            dt=0.1,
            unit="m",
        )
        result1 = inject(timeseries, other1)

        # Inject second signal (twos from t=1.0 to t=3.0, indices 10-30)
        # This overlaps with the first signal in indices 10-20
        other2 = GWpyTimeSeries(
            np.ones(20) * 2,
            t0=1.0,
            dt=0.1,
            unit="m",
        )
        result2 = inject(result1, other2)

        # Check accumulation
        # Indices 0-10: only other1 was injected, so value = 1
        assert np.allclose(result2.value[:10], 1)
        # Indices 10-20: both other1 and other2 overlap, so value = 1 + 2 = 3
        assert np.allclose(result2.value[10:20], 3)
        # Indices 20-30: only other2 was injected, so value = 2
        assert np.allclose(result2.value[20:30], 2)
        # Indices 30+: nothing injected, so value = 0
        assert np.allclose(result2.value[30:], 0)

    def test_inject_with_negative_values(self):
        """Test injection with negative signal values."""
        timeseries = GWpyTimeSeries(
            np.ones(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            -np.ones(30),
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Injected region should have 1 + (-1) = 0
        assert np.all(result.value[20:50] == 0)
        assert np.all(result.value[:20] == 1)

    def test_inject_with_float32_data(self):
        """Test injection preserves data type."""
        timeseries = GWpyTimeSeries(
            np.zeros(100, dtype=np.float32),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30, dtype=np.float32),
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Result should maintain consistency
        assert len(result) == len(timeseries)
        assert np.any(result.value[20:50] != 0)


class TestInjectDataTypes:
    """Test injection with different data types."""

    def test_inject_with_complex_data(self):
        """Test injection with complex-valued data."""
        timeseries = GWpyTimeSeries(
            np.zeros(100, dtype=complex),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30, dtype=complex),
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        assert len(result) == len(timeseries)
        assert np.any(result.value[20:50] != 0)


class TestInjectCropping:
    """Test injection with cropped data."""

    def test_inject_crops_other_to_fit(self):
        """Test that inject crops other to fit within timeseries bounds."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        # Other extends beyond timeseries
        other = GWpyTimeSeries(
            np.ones(200),
            t0=-2.0,  # Starts before timeseries
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Result should have same length as timeseries
        assert len(result) == len(timeseries)
        # All samples should be injected (since other covers entire range)
        assert np.all(result.value == 1)


class TestInjectZeroValues:
    """Test injection with zero values."""

    def test_inject_zeros_does_not_change_timeseries(self):
        """Test that injecting zeros doesn't change the timeseries."""
        timeseries = GWpyTimeSeries(
            np.ones(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.zeros(30),
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # All values should remain 1 (1 + 0 = 1)
        assert np.all(result.value == 1)

    def test_inject_into_zero_timeseries(self):
        """Test injection into zero-filled timeseries."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30),
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Injected region should have ones
        assert np.allclose(result.value[20:50], 1)
        # Rest should be zeros
        assert np.allclose(result.value[:20], 0)
        assert np.allclose(result.value[50:], 0)


class TestInjectTimescales:
    """Test injection with different time scales."""

    def test_inject_very_fine_resolution(self):
        """Test injection with very fine time resolution (high sampling rate)."""
        timeseries = GWpyTimeSeries(
            np.zeros(10000),
            t0=0,
            dt=1e-4,  # 10 kHz sampling
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(1000),
            t0=0.1,
            dt=1e-4,
            unit="m",
        )

        result = inject(timeseries, other)

        assert len(result) == len(timeseries)
        assert np.any(result.value != 0)

    def test_inject_very_coarse_resolution(self):
        """Test injection with very coarse time resolution (low sampling rate)."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=10.0,  # 0.1 Hz sampling
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30),
            t0=50.0,
            dt=10.0,
            unit="m",
        )

        result = inject(timeseries, other)

        assert len(result) == len(timeseries)
        assert np.any(result.value != 0)


class TestInjectMagnitudes:
    """Test injection with extreme magnitudes."""

    def test_inject_very_large_values(self):
        """Test injection with very large signal values."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30) * 1e10,
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Injected values should be preserved
        assert np.allclose(result.value[20:50], 1e10)

    def test_inject_very_small_values(self):
        """Test injection with very small signal values (close to machine epsilon)."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        other = GWpyTimeSeries(
            np.ones(30) * 1e-15,
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Small values should be detectable (above numerical noise)
        assert np.any(result.value[20:50] != 0)


class TestInjectOffsetEdgeCases:
    """Test edge cases related to time offset calculation."""

    def test_inject_very_small_offset(self):
        """Test injection with very small fractional offset."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        # Offset of 0.001 samples (0.0001 seconds)
        other = GWpyTimeSeries(
            np.ones(30),
            t0=2.50001,  # Very small offset
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other, interpolate_if_offset=True)

        assert len(result) == len(timeseries)
        assert np.any(result.value != 0)

    def test_inject_offset_near_rounding_boundary(self):
        """Test injection with offset very close to 0.5 samples."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        # Offset very close to 0.5 samples
        other = GWpyTimeSeries(
            np.ones(30),
            t0=2.5 + 0.0499,  # Almost 0.5 samples
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other, interpolate_if_offset=True)

        assert len(result) == len(timeseries)
        assert np.any(result.value != 0)


class TestInjectInterpolationQuality:
    """Test the quality of interpolation."""

    def test_inject_interpolation_preserves_signal_characteristics(self):
        """Test that interpolation preserves signal characteristics."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        # Create a smooth sinusoidal signal
        t = np.arange(20) * 0.1
        signal = np.sin(2 * np.pi * t / 10)  # 0.1 Hz sine wave

        other = GWpyTimeSeries(
            signal,
            t0=2.55,  # Fractional offset
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other, interpolate_if_offset=True)

        # Check that interpolation occurred and result is reasonable
        assert len(result) == len(timeseries)
        # Interpolated region should have some non-zero values
        assert np.any(result.value[25:45] != 0)
        # Values should be in reasonable range (approximately -1 to 1 for sine)
        upper_threshold = 1.5
        lower_threshold = -1.5
        assert np.all(result.value[25:45] <= upper_threshold)
        assert np.all(result.value[25:45] >= lower_threshold)


class TestInjectMixedSigns:
    """Test injection with mixed positive and negative values."""

    def test_inject_mixed_sign_signal(self):
        """Test injection of signal with mixed positive and negative values."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        # Alternating positive and negative values
        signal = np.array([1, -1] * 15)
        other = GWpyTimeSeries(
            signal,
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Check that mixed signs are preserved
        assert np.any(result.value[20:50] > 0)
        assert np.any(result.value[20:50] < 0)

    def test_inject_cancellation_with_negative(self):
        """Test that injection with negatives can cancel previous values."""
        timeseries = GWpyTimeSeries(
            np.ones(100),
            t0=0,
            dt=0.1,
            unit="m",
        )

        # Inject -0.5 to get partial cancellation
        other = GWpyTimeSeries(
            np.ones(30) * -0.5,
            t0=2.0,
            dt=0.1,
            unit="m",
        )

        result = inject(timeseries, other)

        # Injected region should have 1 + (-0.5) = 0.5
        assert np.allclose(result.value[20:50], 0.5)
        # Uninjected regions should remain 1
        assert np.allclose(result.value[:20], 1)
        assert np.allclose(result.value[50:], 1)


class TestInjectExactBoundaries:
    """Test injection at exact time boundaries."""

    def test_inject_starts_at_exact_sample(self):
        """Test injection that starts at exact sample boundary."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.01,
            unit="m",
        )

        # Start exactly at sample 50 (t=0.5)
        other = GWpyTimeSeries(
            np.ones(10),
            t0=0.5,
            dt=0.01,
            unit="m",
        )

        result = inject(timeseries, other)

        assert np.allclose(result.value[:50], 0)
        assert np.allclose(result.value[50:60], 1)
        assert np.allclose(result.value[60:], 0)

    def test_inject_ends_at_exact_sample(self):
        """Test injection that ends at exact sample boundary."""
        timeseries = GWpyTimeSeries(
            np.zeros(100),
            t0=0,
            dt=0.01,
            unit="m",
        )

        # End exactly at sample 75 (t=0.75)
        other = GWpyTimeSeries(
            np.ones(25),
            t0=0.25,
            dt=0.01,
            unit="m",
        )

        result = inject(timeseries, other)

        assert np.allclose(result.value[:25], 0)
        assert np.allclose(result.value[25:50], 1)
        assert np.allclose(result.value[50:], 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
