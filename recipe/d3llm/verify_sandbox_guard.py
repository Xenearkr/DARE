#!/usr/bin/env python3
"""Verify sandbox guard + code reward paths used in BGPO training."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
from tempfile import TemporaryDirectory

import pandas as pd

from verl.utils.reward_score.code_reward import (
    _temp_run,
    humaneval_check_correctness,
    lcb_check_correctness_v2,
)
from verl.utils.reward_score.code_utils.sandbox_guard import (
    guard_context,
    reliability_guard,
    restore_guard_state,
    snapshot_guard_state,
)


def _assert_os_healthy() -> None:
    assert os.putenv is not None
    assert os.unlink is not None
    assert shutil.rmtree is not None
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe")
        with open(path, "w", encoding="utf-8") as f:
            f.write("ok")


def test_guard_restore_cycle() -> None:
    saved = snapshot_guard_state()
    reliability_guard()
    restore_guard_state(saved)
    _assert_os_healthy()
    print("[PASS] guard/restore cycle")


def test_guard_context() -> None:
    with guard_context():
        assert os.putenv is None
    _assert_os_healthy()
    print("[PASS] guard_context")


def test_humaneval_after_lcb_processes() -> None:
    df = pd.read_parquet("data/preprocessed/rl/test/humaneval_evalplus_1.parquet")
    row = df.iloc[0]
    gt = row["reward_model"]["ground_truth"]
    pred = "def has_close_elements(numbers, threshold):\n    return False\n"

    train_row = pd.read_parquet("data/preprocessed/rl/train/primeintellect-K8_1.parquet").iloc[0]
    tests = train_row["reward_model"]["ground_truth"]
    metadata = train_row.get("extra_info", {}) or {}
    code = "def solve():\n    return 0\n"

    for _ in range(5):
        lcb_check_correctness_v2(tests, code, metadata, timeout=2, debug=False)
        _assert_os_healthy()

    ok, _ = humaneval_check_correctness(gt, pred, timeout_per_test=1)
    assert ok is False
    _assert_os_healthy()
    print("[PASS] humaneval after repeated lcb Process scoring")


def test_temp_run_process() -> None:
    train_row = pd.read_parquet("data/preprocessed/rl/train/primeintellect-K8_1.parquet").iloc[0]
    tests = train_row["reward_model"]["ground_truth"]
    metadata = train_row.get("extra_info", {}) or {}
    from verl.utils.reward_score.code_reward import postprocess_lcb_sample

    sample = postprocess_lcb_sample(tests, metadata)
    manager = mp.Manager()
    result = manager.list()
    metadata_list = manager.list()
    code = "def solve():\n    return 0\n"
    p = mp.Process(
        target=_temp_run,
        args=(sample, code, False, result, metadata_list, 2),
    )
    p.start()
    p.join(timeout=30)
    if p.is_alive():
        p.kill()
    assert p.exitcode is not None
    _assert_os_healthy()
    print(f"[PASS] _temp_run subprocess exitcode={p.exitcode}")


def main() -> None:
    test_guard_restore_cycle()
    test_guard_context()
    test_humaneval_after_lcb_processes()
    test_temp_run_process()
    print("[PASS] all sandbox guard checks")


if __name__ == "__main__":
    main()
