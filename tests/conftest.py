"""Shared pytest configuration for the gwmock test suite.

The only thing here is the mutation-testing fork-safety hook. It is a no-op for an
ordinary ``pytest`` run and is installed only when a ``mutmut`` driver imports this
conftest; see ``tests/mutmut_fork_safety.py`` for what it does and why it has to be
installed from a conftest rather than from configuration.
"""

from __future__ import annotations

from tests.mutmut_fork_safety import install_if_mutmut_driver

install_if_mutmut_driver()
