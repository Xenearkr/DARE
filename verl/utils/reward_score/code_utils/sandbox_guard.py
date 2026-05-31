"""Sandbox helpers for untrusted code execution."""

from __future__ import annotations

import builtins
import faulthandler
import inspect
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

_GUARD_OS_ATTRS = (
    "kill",
    "system",
    "putenv",
    "remove",
    "removedirs",
    "rmdir",
    "fchdir",
    "setuid",
    "fork",
    "forkpty",
    "killpg",
    "rename",
    "renames",
    "truncate",
    "replace",
    "unlink",
    "fchmod",
    "fchown",
    "chmod",
    "chown",
    "chroot",
    "lchflags",
    "lchmod",
    "lchown",
    "getcwd",
    "chdir",
)
_GUARD_SHUTIL_ATTRS = ("rmtree", "move", "chown")
_GUARD_MODULE_KEYS = ("ipdb", "joblib", "resource", "psutil", "tkinter")
_MISSING = object()


def _get_os_attr(name: str) -> Any:
    if hasattr(os, name):
        return getattr(os, name)
    return _MISSING


def _capture_guard_state() -> dict[str, Any]:
    return {
        "builtins.exit": builtins.exit,
        "builtins.quit": builtins.quit,
        "builtins.help": builtins.help,
        "subprocess.Popen": subprocess.Popen,
        "faulthandler_enabled": faulthandler.is_enabled(),
        "os.environ.OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        **{f"os.{name}": _get_os_attr(name) for name in _GUARD_OS_ATTRS},
        **{f"shutil.{name}": getattr(shutil, name) for name in _GUARD_SHUTIL_ATTRS},
        **{f"sys.modules.{name}": sys.modules.get(name) for name in _GUARD_MODULE_KEYS},
    }


_ORIGINAL_GUARD_STATE = _capture_guard_state()


def reliability_guard(maximum_memory_bytes: Optional[int] = None) -> None:
    """Disable destructive builtins in the current process (call only in isolated workers)."""
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None  # type: ignore[assignment]

    builtins.help = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def snapshot_guard_state() -> dict[str, Any]:
    """Capture current process globals before applying reliability_guard()."""
    return _capture_guard_state()


def restore_guard_state(state: dict[str, Any] | None = None) -> None:
    """Restore process globals mutated by reliability_guard()."""
    saved = _ORIGINAL_GUARD_STATE if state is None else state

    builtins.exit = saved["builtins.exit"]
    builtins.quit = saved["builtins.quit"]
    builtins.help = saved["builtins.help"]
    subprocess.Popen = saved["subprocess.Popen"]

    for name in _GUARD_OS_ATTRS:
        original = saved[f"os.{name}"]
        if original is _MISSING:
            if hasattr(os, name):
                delattr(os, name)
        else:
            setattr(os, name, original)
    for name in _GUARD_SHUTIL_ATTRS:
        setattr(shutil, name, saved[f"shutil.{name}"])

    omp_threads = saved["os.environ.OMP_NUM_THREADS"]
    if omp_threads is None:
        os.environ.pop("OMP_NUM_THREADS", None)
    else:
        os.environ["OMP_NUM_THREADS"] = omp_threads

    for name in _GUARD_MODULE_KEYS:
        module_value = saved[f"sys.modules.{name}"]
        if module_value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module_value

    if saved["faulthandler_enabled"]:
        faulthandler.enable()
    else:
        faulthandler.disable()


@contextmanager
def guard_context() -> Iterator[None]:
    """Apply reliability_guard() and always restore on exit."""
    saved = snapshot_guard_state()
    try:
        reliability_guard()
        yield
    finally:
        restore_guard_state(saved)


def snapshot_shutil_rmtree() -> Callable[..., None]:
    return _ORIGINAL_GUARD_STATE["shutil.rmtree"]


def restore_shutil_rmtree(saved_rmtree: Callable[..., None] | None = None) -> None:
    restore_guard_state()


def standalone_guard_module_source() -> str:
    """Minimal guard module for subprocess execution (no verl/torch imports)."""
    return (
        "from typing import Optional\n"
        "import builtins\n"
        "import faulthandler\n"
        "import os\n"
        "import platform\n"
        "import shutil\n"
        "import subprocess\n"
        "import sys\n\n"
        + inspect.getsource(reliability_guard)
    )


def guarded_solution_preamble() -> str:
    return "import _sandbox_guard\n_sandbox_guard.reliability_guard()\n\n"


def write_standalone_guard_module(tmpdir: str) -> str:
    path = os.path.join(tmpdir, "_sandbox_guard.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(standalone_guard_module_source())
    return path


def repo_root_for_imports() -> str:
    return str(Path(__file__).resolve().parents[4])


def build_guarded_runner_script(solution_path: str) -> str:
    """Run *solution_path* in a fresh interpreter after applying reliability_guard."""
    return f"""import runpy
import sys
sys.path.insert(0, {repo_root_for_imports()!r})
from verl.utils.reward_score.code_utils.sandbox_guard import reliability_guard
reliability_guard()
runpy.run_path({solution_path!r}, run_name='__main__')
"""
