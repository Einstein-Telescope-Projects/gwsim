"""Unit tests for discovery functions."""

from __future__ import annotations

from gwmock.cli.utils.discovery import (
    discover_geometries,
    discover_glitch_models,
    discover_population_presets,
    discover_psds,
    discover_source_types,
    discover_waveform_models,
)


def test_discover_geometries_returns_list():
    result = discover_geometries()
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(g, str) for g in result)


def test_discover_psds_returns_list():
    result = discover_psds()
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(p, str) for p in result)
    assert all(p.endswith("_psd") for p in result)


def test_discover_source_types_returns_list():
    result = discover_source_types()
    assert isinstance(result, list)
    assert len(result) > 0
    assert "bbh" in result


def test_discover_glitch_models_returns_list():
    result = discover_glitch_models()
    assert isinstance(result, list)
    assert len(result) > 0


def test_discover_population_presets_returns_list():
    result = discover_population_presets()
    assert isinstance(result, list)
    assert len(result) > 0


def test_discover_waveform_models_returns_list():
    result = discover_waveform_models()
    assert isinstance(result, list)
    # May be empty if LAL is not installed, but should not raise
    assert all(isinstance(w, str) for w in result)


def test_discover_geometries_fallback(monkeypatch):
    """Discovery returns empty list when the import fails."""
    from gwmock.cli.utils import discovery

    def patched():
        try:
            raise ImportError("mocked")
        except Exception:
            return []

    original = discovery.discover_geometries
    monkeypatch.setattr(discovery, "discover_geometries", patched)
    assert discovery.discover_geometries() == []
    monkeypatch.setattr(discovery, "discover_geometries", original)
