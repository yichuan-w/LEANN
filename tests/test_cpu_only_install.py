"""Packaging metadata checks for CPU-only installs."""

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    tomllib = pytest.importorskip("tomli")


def _load_leann_pyproject():
    pyproject_path = Path(__file__).resolve().parents[1] / "packages" / "leann" / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text())


def _load_leann_core_pyproject():
    pyproject_path = (
        Path(__file__).resolve().parents[1] / "packages" / "leann-core" / "pyproject.toml"
    )
    return tomllib.loads(pyproject_path.read_text())


def test_leann_base_dependencies_include_diskann():
    data = _load_leann_pyproject()
    deps = data["project"].get("dependencies", [])

    assert "leann-core>=0.1.0" in deps
    assert "leann-backend-hnsw>=0.1.0" in deps
    assert "leann-backend-diskann>=0.1.0" in deps


def test_leann_core_numpy_pinned_below_2():
    data = _load_leann_core_pyproject()
    deps = data["project"].get("dependencies", [])

    assert any(dep.startswith("numpy") and "<2" in dep for dep in deps)


def test_leann_core_cpu_extra_pins_cpu_torch():
    data = _load_leann_core_pyproject()
    extras = data["project"].get("optional-dependencies", {})

    cpu_deps = extras.get("cpu", [])
    assert cpu_deps, "cpu extra should be defined"
    assert any(
        dep.startswith("torch")
        and "==2.2.2" in dep
        and "platform_system == 'Linux'" in dep
        and "python_version < '3.13'" in dep
        for dep in cpu_deps
    )


def test_leann_cpu_extra_defined():
    data = _load_leann_pyproject()
    extras = data["project"].get("optional-dependencies", {})

    assert "cpu" in extras
    assert "leann-core[cpu]>=0.1.0" in extras["cpu"]
