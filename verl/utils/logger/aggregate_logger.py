# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A Ray logger will receive logging info from different processes.
"""

import logging
import math
import numbers
from typing import Dict


def format_metric_value(v: numbers.Number) -> str:
    """Format scalars for console logging without collapsing small values to 0.000."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, numbers.Integral) and not isinstance(v, bool):
        return str(v)
    fv = float(v)
    if not math.isfinite(fv):
        return str(fv)
    if fv == 0.0:
        return "0"
    av = abs(fv)
    # Scientific notation for very small/large magnitudes (e.g. lr=5e-7, grad_norm=1e-5).
    if av < 1e-3 or av >= 1e4:
        return f"{fv:.6e}"
    s = f"{fv:.8f}".rstrip("0").rstrip(".")
    return s if s else "0"


def concat_dict_to_str(dict: Dict, step):
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{format_metric_value(v)}")
    output_str = " - ".join(output)
    return output_str


class LocalLogger:
    def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
        self.print_to_console = print_to_console
        if print_to_console:
            print("Using LocalLogger is deprecated. The constructor API will change ")

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class DecoratorLoggerBase:
    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")
