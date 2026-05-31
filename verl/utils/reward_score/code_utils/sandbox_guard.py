"""Sandbox helpers for untrusted code execution."""

from __future__ import annotations

import faulthandler
import inspect
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional


def reliability_guard(maximum_memory_bytes: Optional[int] = None) -> None:
    """Disable destructive builtins in the current process (call only in isolated workers)."""
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

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
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None  # type: ignore[assignment]

    __builtins__["help"] = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def snapshot_shutil_rmtree() -> Callable[..., None]:
    return shutil.rmtree


def restore_shutil_rmtree(saved_rmtree: Callable[..., None]) -> None:
    if shutil.rmtree is None:
        shutil.rmtree = saved_rmtree


def standalone_guard_module_source() -> str:
    """Minimal guard module for subprocess execution (no verl/torch imports)."""
    return (
        "from typing import Optional\n"
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
