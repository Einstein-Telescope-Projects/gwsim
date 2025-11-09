"""Custom JSON encoder for TimeSeries objects."""

from __future__ import annotations

import json
from typing import Any

from gwsim.data.time_series import TimeSeries


class TimeSeriesEncoder(json.JSONEncoder):
    """Custom JSON encoder for TimeSeries objects."""

    def default(self, o: Any) -> Any:
        """Serialize TimeSeries objects to JSON.

        Args:
            o: Object to serialize.

        Returns:
            Serialized representation of the TimeSeries object.
        """
        if isinstance(o, TimeSeries):
            return {
                "__timeseries__": True,
                "data": [o[i].value.tolist() for i in range(o.num_channels)],
                "start_time": o.start_time.value,
                "sampling_frequency": o.sampling_frequency.value,
            }
        return super().default(o)
