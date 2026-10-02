from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional


# ============================================================
# GStreamer runtime detection
# ============================================================

def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _machine_triplets() -> list[str]:
    machine = platform.machine().lower()

    triplets = []

    if machine in ("x86_64", "amd64"):
        triplets.extend(["x86_64-linux-gnu", "amd64-linux-gnu"])
    elif machine in ("aarch64", "arm64"):
        triplets.extend(["aarch64-linux-gnu"])
    elif machine.startswith("arm"):
        triplets.extend(["arm-linux-gnueabihf", "arm-linux-gnueabi"])

    # Keep common values as fallbacks. Duplicates are removed by _dedupe_paths().
    triplets.extend(["x86_64-linux-gnu", "aarch64-linux-gnu"])
    return triplets


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        key = str(path).lower()
        if key not in seen:
            out.append(path)
            seen.add(key)

    return out


def _plugin_dirs_for_root(root: Path) -> list[Path]:
    paths = [
        root / "lib" / "gstreamer-1.0",
        root / "lib64" / "gstreamer-1.0",
    ]

    for triplet in _machine_triplets():
        paths.append(root / "lib" / triplet / "gstreamer-1.0")

    return _dedupe_paths(paths)


def _typelib_dirs_for_root(root: Path) -> list[Path]:
    paths = [
        root / "lib" / "girepository-1.0",
        root / "lib64" / "girepository-1.0",
        root / "share" / "gir-1.0",
    ]

    for triplet in _machine_triplets():
        paths.append(root / "lib" / triplet / "girepository-1.0")

    return _dedupe_paths(paths)


def _scanner_paths_for_root(root: Path) -> list[Path]:
    paths = [
        root / "libexec" / "gstreamer-1.0" / "gst-plugin-scanner.exe",
        root / "libexec" / "gstreamer-1.0" / "gst-plugin-scanner",
        root / "lib" / "gstreamer-1.0" / "gst-plugin-scanner",
        root / "lib64" / "gstreamer-1.0" / "gst-plugin-scanner",
    ]

    for triplet in _machine_triplets():
        paths.extend([
            root / "lib" / triplet / "gstreamer1.0" / "gstreamer-1.0" / "gst-plugin-scanner",
            root / "lib" / triplet / "gstreamer-1.0" / "gst-plugin-scanner",
        ])

    return _dedupe_paths(paths)


def _first_existing_dir(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        try:
            if path.is_dir():
                return path
        except OSError:
            pass
    return None


def _first_existing_file(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        try:
            if path.is_file():
                return path
        except OSError:
            pass
    return None


def _is_valid_gstreamer_root(path: Path) -> bool:
    """
    A valid root usually contains bin/gst-launch-1.0 and a plugin folder.

    On Windows/MSVC:
        <root>/bin/gst-launch-1.0.exe
        <root>/lib/gstreamer-1.0

    On Debian/Ubuntu Linux:
        /usr/bin/gst-launch-1.0
        /usr/lib/x86_64-linux-gnu/gstreamer-1.0
    """
    try:
        if not path.is_dir():
            return False

        bin_dir = path / "bin"
        gst_launch_win = bin_dir / "gst-launch-1.0.exe"
        gst_launch_unix = bin_dir / "gst-launch-1.0"
        plugin_dir = _first_existing_dir(_plugin_dirs_for_root(path))

        return bin_dir.is_dir() and plugin_dir is not None and (
            gst_launch_win.is_file() or gst_launch_unix.is_file()
        )
    except OSError:
        return False


def _project_root_from_this_file() -> Path:
    here = Path(__file__).resolve()
    try:
        return here.parents[2]
    except IndexError:
        return here.parent


def _candidate_roots_from_environment() -> list[Path]:
    candidates: list[Path] = []

    env_names = [
        "GST_ROOT",
        "GSTREAMER_ROOT",
        "GSTREAMER_ROOT_X86_64",
        "GSTREAMER_1_0_ROOT_X86_64",
        "GSTREAMER_1_0_ROOT_MSVC_X86_64",
        "GSTREAMER_1_0_ROOT_MINGW_X86_64",
    ]

    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            candidates.append(Path(value))

    return candidates


def _candidate_roots_from_program_files() -> list[Path]:
    candidates: list[Path] = []

    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root_text = os.environ.get(env_name, "").strip()
        if not root_text:
            continue

        root = Path(root_text)
        candidates.extend([
            root / "gstreamer" / "1.0" / "msvc_x86_64",
            root / "GStreamer" / "1.0" / "msvc_x86_64",
            root / "gstreamer" / "1.0" / "mingw_x86_64",
            root / "GStreamer" / "1.0" / "mingw_x86_64",
        ])

    return candidates


def _candidate_roots_from_unix_prefixes() -> list[Path]:
    if os.name == "nt":
        return []

    return [
        Path("/usr"),
        Path("/usr/local"),
        Path("/opt/gstreamer"),
    ]


def _candidate_roots_from_path() -> list[Path]:
    candidates: list[Path] = []

    for exe_name in ("gst-launch-1.0.exe", "gst-launch-1.0"):
        found = shutil.which(exe_name)
        if not found:
            continue

        exe_path = Path(found)
        if exe_path.parent.name.lower() == "bin":
            candidates.append(exe_path.parent.parent)

    return candidates


def _candidate_roots_from_project(project_root: Path) -> list[Path]:
    return [
        project_root / "Runtime" / "GStreamer",
        project_root / "runtime" / "gstreamer",
        project_root / "GStreamer",
        project_root / "gstreamer",
    ]


def find_gstreamer_root() -> Optional[Path]:
    project_root = _project_root_from_this_file()

    candidates: list[Path] = []
    candidates.extend(_candidate_roots_from_project(project_root))
    candidates.extend(_candidate_roots_from_environment())
    candidates.extend(_candidate_roots_from_path())
    candidates.extend(_candidate_roots_from_unix_prefixes())
    candidates.extend(_candidate_roots_from_program_files())

    seen: set[str] = set()

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate

        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)

        if _is_valid_gstreamer_root(resolved):
            return resolved

    return None


# ============================================================
# Environment modification helpers
# ============================================================

def prepend_path_if_possible(path_to_add: Path) -> None:
    if not _path_exists(path_to_add):
        return

    path_text = str(path_to_add)
    current_path = os.environ.get("PATH", "")

    parts = [part for part in current_path.split(os.pathsep) if part]
    lower_parts = {part.lower() for part in parts}

    if path_text.lower() not in lower_parts:
        os.environ["PATH"] = path_text + (os.pathsep + current_path if current_path else "")

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path_text)
        except OSError:
            pass


def prepend_env_path_if_possible(name: str, path_to_add: Path) -> None:
    if not _path_exists(path_to_add):
        return

    path_text = str(path_to_add)
    current = os.environ.get(name, "")

    parts = [part for part in current.split(os.pathsep) if part]
    lower_parts = {part.lower() for part in parts}

    if path_text.lower() not in lower_parts:
        os.environ[name] = path_text + (os.pathsep + current if current else "")


def set_env_if_missing(name: str, value: str | Path) -> None:
    if not os.environ.get(name):
        os.environ[name] = str(value)


def set_env(name: str, value: str | Path) -> None:
    os.environ[name] = str(value)


# ============================================================
# Optional Python binding path support
# ============================================================

def add_possible_python_binding_paths(gst_root: Path) -> None:
    if os.environ.get("GST_ADD_RUNTIME_PYTHON_BINDINGS", "").strip() != "1":
        return

    candidates = [
        gst_root / "lib" / "site-packages",
        gst_root / "lib" / "python3" / "site-packages",
    ]

    lib_dir = gst_root / "lib"

    if lib_dir.is_dir():
        try:
            for item in lib_dir.iterdir():
                if item.is_dir() and item.name.lower().startswith("python"):
                    candidates.append(item / "site-packages")
        except OSError:
            pass

    for candidate in candidates:
        if (candidate / "gi" / "__init__.py").is_file():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)


# ============================================================
# Main setup function
# ============================================================

def configure_gstreamer_environment() -> None:
    """
    Configure GStreamer before importing gi/Gst.

    Linux system packages usually work from PATH without forcing GST_ROOT.
    Windows package/PyPI-bundle mixes are avoided with QGXS_USE_PIP_GSTREAMER=1.
    """
    if os.environ.get("QGXS_USE_PIP_GSTREAMER", "").strip() == "1":
        set_env_if_missing("GST_DEBUG_NO_COLOR", "1")
        return

    gst_root_text = os.environ.get("GST_ROOT", "").strip()

    if gst_root_text:
        gst_root = Path(gst_root_text)
    else:
        detected = find_gstreamer_root()
        if detected is None:
            set_env_if_missing("GST_DEBUG_NO_COLOR", "1")
            return
        gst_root = detected
        set_env("GST_ROOT", gst_root)

    if not _is_valid_gstreamer_root(gst_root):
        set_env_if_missing("GST_DEBUG_NO_COLOR", "1")
        return

    bin_path = gst_root / "bin"
    plugin_path = _first_existing_dir(_plugin_dirs_for_root(gst_root))
    scanner_path = _first_existing_file(_scanner_paths_for_root(gst_root))
    typelib_path = _first_existing_dir(_typelib_dirs_for_root(gst_root))

    prepend_path_if_possible(bin_path)
    add_possible_python_binding_paths(gst_root)

    if plugin_path is not None:
        set_env_if_missing("GST_PLUGIN_PATH", plugin_path)
        set_env_if_missing("GST_PLUGIN_SYSTEM_PATH_1_0", plugin_path)

    if scanner_path is not None:
        set_env_if_missing("GST_PLUGIN_SCANNER", scanner_path)

    if typelib_path is not None:
        prepend_env_path_if_possible("GI_TYPELIB_PATH", typelib_path)

    set_env_if_missing("GST_DEBUG_NO_COLOR", "1")

    project_root = _project_root_from_this_file()
    set_env_if_missing("GST_REGISTRY", project_root / "gst-registry-python-sender.bin")
