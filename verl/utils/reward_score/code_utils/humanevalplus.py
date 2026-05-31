import ast
import os
import subprocess
import sys
from tempfile import TemporaryDirectory

from .sandbox_guard import (
    build_guarded_runner_script,
    repo_root_for_imports,
    restore_shutil_rmtree,
    snapshot_shutil_rmtree,
)
from .utils import BASE_IMPORTS

CLI_ARG_SIZE_LIMIT = 1024 * 3

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 60


def get_num_test_cases(test_code):
    # Parse the code into an AST
    parsed = ast.parse(test_code)

    # Find the assignment node for 'inputs'
    inputs_node = None
    results_node = None

    for node in ast.walk(parsed):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "inputs":
                        inputs_node = node.value

    if inputs_node is None:
        return "Could not find inputs or results in the code"

    # Count number of test cases
    if isinstance(inputs_node, ast.List):
        input_count = len(inputs_node.elts)
    else:
        input_count = "Unknown (not a direct list)"
    return input_count


def run_test(code, test: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS):
    if not test:
        raise ValueError("No test provided.")

    code_to_run = f"""
{BASE_IMPORTS}

{code}

{test}

"""
    saved_rmtree = snapshot_shutil_rmtree()
    try:
        with TemporaryDirectory() as tmpdir:
            solution_path = os.path.join(tmpdir, "solution.py")
            with open(solution_path, "w") as f:
                f.write(code_to_run)

            env = os.environ.copy()
            repo_root = repo_root_for_imports()
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

            command = [
                sys.executable,
                "-c",
                build_guarded_runner_script(solution_path),
            ]
            try:
                result = subprocess.run(
                    command,
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                    timeout=timeout,
                )

                stderr = result.stderr.decode().strip()
                stdout = result.stdout.decode()
                if result.returncode == 0:
                    return True, stdout
                return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

            except subprocess.TimeoutExpired:
                return False, _ERROR_MSG_PREFIX + f"Execution timed out after {timeout} seconds."
            except Exception as e:
                return False, _ERROR_MSG_PREFIX + f"An Exception occurred in the code: {str(e)}"
    finally:
        restore_shutil_rmtree(saved_rmtree)


# Backward-compatible re-export for callers/tests that import from humanevalplus.
from .sandbox_guard import reliability_guard  # noqa: E402,F401