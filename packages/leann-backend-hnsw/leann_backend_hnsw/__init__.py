import os
from pathlib import Path

_DLL_DIR_HANDLES: list[object] = []


def _configure_windows_dll_search_path() -> None:
    """Register vcpkg DLL directories so Windows can resolve native dependencies.

    Python 3.8+ no longer searches PATH for DLL dependencies of native
    extensions (see https://github.com/numpy/numpy/wiki/windows-dll-notes).
    The standard workaround is ``os.add_dll_directory``.

    This only matters for **CI builds and source installs** where C++ deps
    (OpenBLAS, ZeroMQ, protobuf, …) live in a vcpkg tree.  Pre-built wheels
    shipped via PyPI bundle all required DLLs inside the wheel (via
    delvewheel), so end-users installing with ``pip install`` are unaffected.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidate_dirs: list[Path] = []
    env_roots = [os.getenv("VCPKG_INSTALLATION_ROOT"), os.getenv("VCPKG_ROOT")]
    for root in env_roots:
        if not root:
            continue
        root_path = Path(root)
        candidate_dirs.extend(
            [
                root_path / "installed" / "x64-windows" / "bin",
                root_path / "installed" / "x64-windows" / "debug" / "bin",
                root_path / "installed" / "x64-windows" / "tools" / "protobuf",
            ]
        )

    for path_entry in os.environ.get("PATH", "").split(";"):
        entry = path_entry.strip()
        if not entry:
            continue
        if "vcpkg" in entry.lower():
            candidate_dirs.append(Path(entry))

    seen: set[str] = set()
    for dll_dir in candidate_dirs:
        resolved = str(dll_dir).lower()
        if resolved in seen or not dll_dir.exists():
            continue
        seen.add(resolved)
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            os.environ["PATH"] = f"{dll_dir};{os.environ.get('PATH', '')}"
        except OSError:
            continue


_configure_windows_dll_search_path()

from . import hnsw_backend as hnsw_backend  # noqa: E402
