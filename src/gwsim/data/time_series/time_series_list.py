"""Module defining TimeSeriesList, a list-like container for TimeSeries objects with validation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, cast, overload

from gwsim.data.time_series.time_series import TimeSeries


class TimeSeriesList(Iterable[TimeSeries]):
    """List of TimeSeries objects with validation."""

    def __init__(self, iterable: list[TimeSeries] | None = None):
        """List of TimeSeries objects with validation.

        Args:
            iterable: Optional list of TimeSeries objects to initialize the list.
        """
        self._data: list[TimeSeries] = []

        if iterable is not None:
            self._validate_items(iterable)
            self._data.extend(iterable)

    def _validate_items(self, items: Iterable[Any]) -> None:
        """Validate that all items are TimeSeries instances.

        Args:
            items: Iterable of items to validate.

        Raises:
            TypeError: If any item is not a TimeSeries instance.
        """
        for item in items:
            if not isinstance(item, TimeSeries):
                raise TypeError(f"All items must be TimeSeries instances, got {type(item)}")

    @overload
    def __setitem__(self, index: int, value: TimeSeries) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[TimeSeries]) -> None: ...

    def __setitem__(self, index: int | slice, value: TimeSeries | Iterable[TimeSeries]) -> None:
        """Set item with validation.

        Args:
            index: Index or slice to set.
            value: TimeSeries object or list of TimeSeries objects to set.
        """
        if isinstance(index, slice):
            items = list(value) if not isinstance(value, list) else value
            self._validate_items(items)
            self._data[index] = cast(list[TimeSeries], items)
        elif isinstance(index, int):
            if not isinstance(value, TimeSeries):
                raise TypeError(f"Value must be a TimeSeries instance, got {type(value)}")
            self._data[index] = value
        else:
            raise TypeError("Index must be an int or slice.")

    @overload
    def __getitem__(self, index: int) -> TimeSeries: ...

    @overload
    def __getitem__(self, index: slice) -> list[TimeSeries]: ...

    def __getitem__(self, index: int | slice) -> TimeSeries | list[TimeSeries]:
        """Get item.

        Args:
            index: Index or slice to get.

        Returns:
            TimeSeries object or list of TimeSeries objects.
        """
        return self._data[index]

    def __len__(self) -> int:
        """Get the number of TimeSeries objects in the list.

        Returns:
            Number of TimeSeries objects in the list.
        """
        return len(self._data)

    def __iter__(self) -> Iterator[TimeSeries]:
        """Iterate over the TimeSeries objects in the list.

        Returns:
            Iterator over the TimeSeries objects in the list.
        """
        return iter(self._data)

    def append(self, value: TimeSeries) -> None:
        """Append a TimeSeries object to the list.

        Args:
            value: TimeSeries object to append.
        """
        if not isinstance(value, TimeSeries):
            raise TypeError(f"Value must be a TimeSeries instance, got {type(value)}")
        self._data.append(value)

    def extend(self, iterable: Iterable[TimeSeries]) -> None:
        """Extend the list with TimeSeries objects from an iterable.

        Args:
            iterable: Iterable of TimeSeries objects to extend the list.
        """
        items = list(iterable)
        self._validate_items(items)
        self._data.extend(items)

    def insert(self, index: int, value: TimeSeries) -> None:
        """Insert a TimeSeries object at a specific index.

        Args:
            index: Index to insert at.
            value: TimeSeries object to insert.
        """
        if not isinstance(value, TimeSeries):
            raise TypeError(f"Value must be a TimeSeries instance, got {type(value)}")
        self._data.insert(index, value)

    def pop(self, index: int = -1) -> TimeSeries:
        """Pop item at index.

        Args:
            index: Index to pop. Defaults to -1 (last item).

        Returns:
            TimeSeries object that was popped.
        """
        return self._data.pop(index)

    def to_json_dict(self) -> dict[str, Any]:
        """Convert the TimeSeriesList to a JSON-serializable dictionary.

        Returns:
            JSON-serializable dictionary representation of the TimeSeriesList.
        """
        return {
            "__type__": "TimeSeriesList",
            "data": self._data,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> TimeSeriesList:
        """Reconstruct a TimeSeriesList instance from a JSON dictionary.

        Args:
            data: Dictionary with keys: 'data' containing list of TimeSeries dicts.

        Returns:
            Reconstructed TimeSeriesList instance.

        Raises:
            ValueError: If required keys are missing or data format is invalid.
        """
        try:
            items = data["data"]
        except KeyError as e:
            raise ValueError(f"Missing required key in JSON dict: {e}") from e

        ts_list: list[TimeSeries] = []
        for item in items:
            if isinstance(item, TimeSeries):
                ts_list.append(item)
            else:
                raise TypeError(f"Invalid item in TimeSeriesList JSON data: {type(item)}")

        return cls(ts_list)

    def __repr__(self) -> str:
        """Get the string representation of the TimeSeriesList.

        Returns:
            String representation of the TimeSeriesList.
        """
        return f"TimeSeriesList({self._data!r})"
