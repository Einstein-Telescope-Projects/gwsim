"""Population adapter bridging ``gwmock-pop`` batches to per-event dictionaries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from gwmock_pop import GWPopSimulator


class PopulationAdapter:
    """Adapt batched population outputs to deterministic per-event dictionaries."""

    def __init__(
        self,
        population_mapping: Mapping[str, Sequence[Any]],
        *,
        source_type: str,
        parameter_names: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the population adapter.

        Args:
            population_mapping: Batched parameter mapping keyed by canonical parameter names.
            source_type: Non-empty population routing key supplied by ``gwmock-pop``.
            parameter_names: Optional ordered parameter names. If omitted, the mapping order is used.
        """
        self._population_mapping = {name: tuple(values) for name, values in population_mapping.items()}
        self._source_type = self._validate_source_type(source_type)
        self._parameter_names = tuple(parameter_names or self._population_mapping.keys())
        self._metadata = dict(metadata or {})
        self._sample_count = self._validate_population_mapping(
            population_mapping=self._population_mapping,
            parameter_names=self._parameter_names,
        )

    @classmethod
    def from_backend(
        cls,
        backend: GWPopSimulator,
        *,
        n_samples: int | None,
        **kwargs: Any,
    ) -> PopulationAdapter:
        """Build an adapter from a ``gwmock-pop`` protocol backend.

        Args:
            backend: The backend to use.
            n_samples: The number of samples to generate, or ``None`` to load the full catalogue.
            **kwargs: Additional arguments to pass to the backend.

        Returns:
            A PopulationAdapter instance.

        Raises:
            TypeError: If the backend does not satisfy the GWPopSimulator protocol.
            ValueError: If n_samples is not a positive integer.
            ValueError: If backend.parameter_names is empty.
        """
        if not isinstance(backend, GWPopSimulator):
            raise TypeError("backend must satisfy the GWPopSimulator protocol.")
        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        parameter_names = tuple(backend.parameter_names)
        if not parameter_names:
            raise ValueError("backend.parameter_names must not be empty.")
        backend_metadata = getattr(backend, "metadata", None)

        return cls(
            backend.simulate(n_samples=n_samples, **kwargs),
            source_type=backend.source_type,
            parameter_names=parameter_names,
            metadata=backend_metadata if isinstance(backend_metadata, Mapping) else None,
        )

    @classmethod
    def from_mapping(
        cls,
        population_mapping: Mapping[str, Sequence[Any]],
        *,
        source_type: str,
        parameter_names: Sequence[str] | None = None,
    ) -> PopulationAdapter:
        """Build an adapter from an already-materialized population mapping.

        Args:
            population_mapping: The population mapping to use.
            source_type: The source type to use for the backend.
            parameter_names: The parameter names to use.

        Returns:
            A PopulationAdapter instance.
        """
        return cls(
            population_mapping,
            source_type=source_type,
            parameter_names=parameter_names,
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the deterministic parameter ordering.

        Returns:
            The deterministic parameter ordering.
        """
        return self._parameter_names

    @property
    def source_type(self) -> str:
        """Return the backend routing key.

        Returns:
            The backend routing key.
        """
        return self._source_type

    @property
    def population_mapping(self) -> Mapping[str, Sequence[Any]]:
        """Return the validated population mapping.

        Returns:
            The validated population mapping.
        """
        return MappingProxyType(self._population_mapping)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return backend metadata preserved across the gwmock-pop boundary.

        Returns:
            The backend metadata.
        """
        return dict(self._metadata)

    def __len__(self) -> int:
        """Return the number of events available in the adapter.

        Returns:
            The number of events available in the adapter.
        """
        return self._sample_count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over per-event parameter dictionaries.

        Returns:
            An iterator over the per-event parameter dictionaries.
        """
        return self.iter_event_parameters()

    def iter_event_parameters(self) -> Iterator[dict[str, Any]]:
        """Yield deterministic per-event parameter dictionaries.

        Returns:
            An iterator over the per-event parameter dictionaries.
        """
        for index in range(len(self)):
            yield self.get_event_parameters(index)

    def get_event_parameters(self, index: int) -> dict[str, Any]:
        """Return one event dictionary from the batched population mapping.

        Args:
            index: The index of the event to get.

        Returns:
            The event dictionary.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Population event index out of range.")

        return {
            parameter_name: self._coerce_event_value(self._population_mapping[parameter_name][index])
            for parameter_name in self.parameter_names
        }

    @staticmethod
    def _validate_source_type(source_type: str) -> str:
        """Validate the source type.

        Args:
            source_type: The source type to validate.

        Returns:
            The validated source type.
        """
        if not isinstance(source_type, str) or not source_type.strip():
            raise ValueError("source_type must be a non-empty string.")
        return source_type

    @classmethod
    def _validate_population_mapping(
        cls,
        *,
        population_mapping: Mapping[str, Sequence[Any]],
        parameter_names: Sequence[str],
    ) -> int:
        """Validate the population mapping.

        Args:
            population_mapping: The population mapping to validate.
            parameter_names: The parameter names to validate.

        Returns:
            The number of samples in the population mapping.
        """
        mapping_keys = tuple(population_mapping.keys())
        expected_keys = tuple(parameter_names)
        if mapping_keys != expected_keys:
            raise ValueError("Population mapping keys must match parameter_names in the same order.")

        sample_count: int | None = None
        for parameter_name in parameter_names:
            values = population_mapping[parameter_name]
            sample_count = cls._validate_parameter_values(
                parameter_name=parameter_name,
                values=values,
                expected_length=sample_count,
            )
        return sample_count or 0

    @staticmethod
    def _validate_parameter_values(
        *,
        parameter_name: str,
        values: Sequence[Any],
        expected_length: int | None,
    ) -> int:
        """Validate the parameter values.

        Args:
            parameter_name: The parameter name to validate.
            values: The values to validate.
            expected_length: The expected length of the values.

        Returns:
            The length of the values.
        """
        shape = getattr(values, "shape", None)
        if shape is not None and len(shape) != 1:
            raise ValueError(f"Population values for {parameter_name} must be one-dimensional.")

        try:
            length = len(values)
        except TypeError as exc:
            raise TypeError(f"Population values for {parameter_name} must be indexable sequences.") from exc

        if expected_length is not None and length != expected_length:
            raise ValueError("All population parameter arrays must have the same number of samples.")
        return length

    @staticmethod
    def _coerce_event_value(value: Any) -> Any:
        """Coerce the event value.

        Args:
            value: The value to coerce.

        Returns:
            The coerced value.
        """
        return value.item() if hasattr(value, "item") else value
