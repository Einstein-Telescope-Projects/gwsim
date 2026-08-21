"""The process-wide random number manager behind every seeded simulator.

``RandomnessMixin`` reaches for these five aliases and nothing else, so what they promise --
a seed reproduces a stream, a saved state resumes one, and spawned seeds do not overlap -- is
what a resumed run's reproducibility rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock.utils.random import (
    RandomManager,
    generate_seeds,
    get_rng,
    get_state,
    set_seed,
    set_state,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_the_process_wide_generator():
    """The manager is a singleton on the class, so a test that seeds it leaks into the next."""
    saved = RandomManager._rng
    yield
    RandomManager._rng = saved


class TestTheSeed:
    def test_the_same_seed_gives_the_same_stream(self) -> None:
        set_seed(1234)
        first = get_rng().random(5)
        set_seed(1234)
        assert np.array_equal(first, get_rng().random(5))

    def test_a_different_seed_gives_a_different_stream(self) -> None:
        set_seed(1234)
        first = get_rng().random(5)
        set_seed(1235)
        assert not np.array_equal(first, get_rng().random(5))

    def test_seeding_replaces_the_generator_rather_than_reusing_it(self) -> None:
        """Reseeding has to install a fresh generator, not reseed the one handed out earlier.

        A held reference must not keep producing the old stream, because the simulator asks for
        ``get_rng()`` once per seed change and keeps what it gets.
        """
        set_seed(7)
        before = get_rng()
        set_seed(7)
        assert get_rng() is not before

    def test_the_generator_is_the_one_the_manager_holds(self) -> None:
        set_seed(11)
        assert get_rng() is RandomManager._rng


class TestTheState:
    def test_a_restored_state_continues_the_same_stream(self) -> None:
        """The checkpoint contract: resume from a saved state and the draws carry on unchanged."""
        set_seed(99)
        rng = get_rng()
        rng.random(3)  # advance, so the state is not the freshly seeded one
        state = get_state()
        expected = rng.random(4)

        set_state(state)
        assert np.array_equal(get_rng().random(4), expected)

    def test_the_state_reflects_draws_already_taken(self) -> None:
        set_seed(99)
        before = get_state()
        get_rng().random(3)
        assert get_state() != before

    def test_restoring_does_not_hand_back_the_generator_that_saved_it(self) -> None:
        """``set_state`` builds a new generator on purpose: sharing one would let a later draw
        on the old reference move the restored stream."""
        set_seed(5)
        rng = get_rng()
        set_state(get_state())
        assert get_rng() is not rng

    def test_the_state_names_the_bit_generator(self) -> None:
        set_seed(5)
        assert get_state()["bit_generator"] == "PCG64"


class TestSpawnedSeeds:
    def test_it_yields_the_requested_count(self) -> None:
        set_seed(3)
        assert len(generate_seeds(4)) == 4

    def test_none_requested_yields_none(self) -> None:
        set_seed(3)
        assert generate_seeds(0) == []

    def test_the_spawned_seeds_are_distinct(self) -> None:
        """Independent, non-overlapping streams are the whole point of spawning."""
        set_seed(3)
        streams = [np.random.default_rng(seed).random(4) for seed in generate_seeds(5)]
        for index, stream in enumerate(streams):
            for other in streams[index + 1 :]:
                assert not np.array_equal(stream, other)

    def test_spawning_is_reproducible_from_the_seed(self) -> None:
        set_seed(3)
        first = [np.random.default_rng(seed).random(2) for seed in generate_seeds(3)]
        set_seed(3)
        second = [np.random.default_rng(seed).random(2) for seed in generate_seeds(3)]
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    def test_spawning_consumes_from_the_manager_stream(self) -> None:
        """The entropy comes from the managed generator, so two calls cannot repeat themselves."""
        set_seed(3)
        first = [np.random.default_rng(seed).random(2) for seed in generate_seeds(2)]
        second = [np.random.default_rng(seed).random(2) for seed in generate_seeds(2)]
        assert not any(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
