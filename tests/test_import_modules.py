from __future__ import annotations

import importlib

import pytest

modules_to_test = [
    "gwsim",
    "gwsim.data",
    "gwsim.data.base",
    "gwsim.generator",
    "gwsim.generator.base",
    "gwsim.generator.state",
    "gwsim.noise",
    "gwsim.noise.base",
    "gwsim.noise.white_noise",
    "gwsim.population",
    "gwsim.population.base",
    "gwsim.tools",
    "gwsim.tools.config",
    "gwsim.tools.default_config",
    "gwsim.tools.main",
    "gwsim.tools.utils",
    "gwsim.utils",
    "gwsim.utils.io",
    "gwsim.utils.log",
    "gwsim.utils.random",
    "gwsim.version",
]


@pytest.mark.parametrize("module_name", modules_to_test)
def test_module_imports(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {e}")
