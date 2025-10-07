from __future__ import annotations

import importlib

import pytest

modules_to_test = [
    "gwsim",
]


@pytest.mark.parametrize("module_name", modules_to_test)
def test_module_imports(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {e}")
