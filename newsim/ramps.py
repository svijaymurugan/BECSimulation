"""Time profiles for switching a coupling on, from 0 to 1.

A Ramp is a callable of time. Inside the simulation it is called with
UNITLESS time (t = step * dtau, in units of 1/omega), but it works in any
consistent unit - it only ever compares t against start and duration.

Every profile is 0 before `start`. All except "exponential" reach exactly 1
at start + duration; "exponential" approaches 1 with time constant `duration`.

This module imports nothing from the package, so config.py can import
RAMP_KINDS for validation without creating an import cycle.
"""
from __future__ import annotations

import math


# Profiles over u in [0, 1], the fraction of the ramp elapsed.
def _linear(u):   return u
def _smooth(u):   return u * u * (3.0 - 2.0 * u)                # C1: zero rate at both ends
def _smoother(u): return u**3 * (u * (6.0 * u - 15.0) + 10.0)   # C2: zero rate and acceleration
def _cosine(u):   return 0.5 * (1.0 - math.cos(math.pi * u))    # C1: raised cosine

PROFILES = {
    "linear":   _linear,
    "smooth":   _smooth,
    "smoother": _smoother,
    "cosine":   _cosine,
}

# The registry of valid names. "none" = no ramp; "quench" and "exponential"
# are handled separately because they are not profiles over [0, 1].
RAMP_KINDS = ("none", "quench", "exponential", *PROFILES)


class Ramp:
    """Scale factor s(t) in [0, 1] for a coupling being switched on."""

    def __init__(self, kind: str, start: float = 0.0, duration: float = 0.0):
        if kind not in RAMP_KINDS:
            raise ValueError(f"Unknown ramp {kind!r}. Choose from {RAMP_KINDS}.")
        if kind not in ("none", "quench") and duration <= 0:
            raise ValueError(f"ramp {kind!r} needs a positive duration")
        self.kind, self.start, self.duration = kind, start, duration

    @classmethod
    def from_config(cls, cfg) -> "Ramp":
        """Convert the config's seconds into the loop's unitless time."""
        return cls(cfg.g2_ramp,
                   start=cfg.g2_ramp_start / cfg.tau,
                   duration=cfg.g2_ramp_time / cfg.tau)

    def __call__(self, t: float) -> float:
        if self.kind == "none":
            return 1.0
        if t < self.start:
            return 0.0
        if self.kind == "quench":
            return 1.0
        x = (t - self.start) / self.duration
        if self.kind == "exponential":
            return 1.0 - math.exp(-x)
        return PROFILES[self.kind](min(x, 1.0))

    def __repr__(self):
        return f"Ramp({self.kind!r}, start={self.start:.4g}, duration={self.duration:.4g})"