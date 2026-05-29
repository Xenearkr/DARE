"""TPF (tokens per forward) helpers for code RL efficiency rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def normalize_rollout_nfe(nfe: Any) -> int:
    """Collapse SGLang multi-round ``nfe`` lists to a single forward count."""
    if nfe is None:
        return 0
    if isinstance(nfe, (list, tuple)):
        if not nfe:
            return 0
        return int(sum(int(x) for x in nfe))
    return int(nfe)


def compute_tpf(gen_tokens: int, nfe: int) -> float:
    if nfe <= 0 or gen_tokens <= 0:
        return 0.0
    return float(gen_tokens) / float(nfe)


@dataclass
class TpfEfficiencyConfig:
    enable: bool = False
    coef: float = 0.1
    initial_baseline: float = 2.0
    max_bonus: float = 0.25
    max_penalty: float = 0.25


class TpfBaselineTracker:
    """Arithmetic-mean baseline over **passed** samples seen so far in training."""

    def __init__(self, cfg: TpfEfficiencyConfig):
        self.cfg = cfg
        self._sum: float = 0.0
        self._count: int = 0

    @property
    def baseline(self) -> float:
        if self._count > 0:
            return self._sum / self._count
        return max(self.cfg.initial_baseline, 1e-6)

    @property
    def num_observations(self) -> int:
        return self._count

    def observe_passed(self, tpf: float) -> None:
        if tpf <= 0:
            return
        self._sum += tpf
        self._count += 1

    def efficiency_reward(self, tpf: float, *, passed: bool) -> float:
        if not self.cfg.enable or not passed or tpf <= 0:
            return 0.0
        base = self.baseline
        raw = self.cfg.coef * (tpf / base - 1.0)
        return float(max(-self.cfg.max_penalty, min(self.cfg.max_bonus, raw)))
