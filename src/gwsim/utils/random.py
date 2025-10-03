"""A random number manager."""

from __future__ import annotations

from numpy.random import Generator, SeedSequence, default_rng


class RandomManager:
    """Singleton manager for random number generation."""

    _rng = default_rng()

    @classmethod
    def get_rng(cls) -> Generator:
        """Get the random number generator.

        Returns:
            Generator: Random number generator.
        """
        return cls._rng

    @classmethod
    def seed(cls, seed_: int):
        """Set the seed of the random number generator.

        Args:
            seed_ (int): Seed.
        """
        cls._rng = default_rng(seed_)

    @classmethod
    def generate_seeds(cls, n_seeds: int) -> list:
        """Generate the seeds using the numpy SeedSequence class such that
        the BitGenerators are independent and very probably non-overlapping.

        Args:
            n_seeds (int): Number of seeds.

        Returns:
            list: A list of SeedSequence.
        """
        return SeedSequence(cls._rng.integers(0, 2**63 - 1, size=4)).spawn(n_seeds)

    @classmethod
    def get_state(cls) -> dict:
        """Get the current state of the random number generator.

        Returns:
            dict: The state of the RNG's bit generator.
        """
        return cls._rng.bit_generator.state

    @classmethod
    def set_state(cls, state: dict):
        """Set the state of the random number generator.

        Args:
            state (dict): The state of the RNG's bit generator.
        """
        cls._rng = default_rng()  # Create new generator to avoid state conflicts
        cls._rng.bit_generator.state = state


# Alias for easy  access
get_rng = RandomManager.get_rng
set_seed = RandomManager.seed
generate_seeds = RandomManager.generate_seeds
get_state = RandomManager.get_state
set_state = RandomManager.set_state
