"""TPF (tokens per forward) helpers for code RL efficiency rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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
    ema_alpha: float = 0.1
    initial_baseline: float = 2.0
    max_bonus: float = 0.25
    max_penalty: float = 0.25


class TpfBaselineTracker:
    """EMA baseline updated only from **passed** samples."""

    def __init__(self, cfg: TpfEfficiencyConfig):
        self.cfg = cfg
        self._ema: Optional[float] = None

    @property
    def baseline(self) -> float:
        if self._ema is not None and self._ema > 0:
            return self._ema
        return max(self.cfg.initial_baseline, 1e-6)

    def observe_passed(self, tpf: float) -> None:
        if tpf <= 0:
            return
        if self._ema is None:
            self._ema = tpf
        else:
            a = self.cfg.ema_alpha
            self._ema = a * tpf + (1.0 - a) * self._ema

    def efficiency_reward(self, tpf: float, *, passed: bool) -> float:
        if not self.cfg.enable or not passed or tpf <= 0:
            return 0.0
        base = self.baseline
        raw = self.cfg.coef * (tpf / base - 1.0)
        return float(max(-self.cfg.max_penalty, min(self.cfg.max_bonus, raw)))
