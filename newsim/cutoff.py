"""Momentum-space filters for the g2 term.

Each variant precomputes whatever it needs once, then exposes a single method.
Callers never learn which one they have.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch


class Cutoff(ABC):
    """Applies a filter to a field already in momentum space."""

    @abstractmethod
    def apply(self, field_k: torch.Tensor) -> torch.Tensor:
        ...


class NoCutoff(Cutoff):
    def __init__(self, grid, kc_u):
        pass

    def apply(self, field_k):
        return field_k                # no multiply at all: exact, and free


class HardCutoff(Cutoff):
    """Keep |k| below kc."""

    def __init__(self, grid, kc_u):
        self.mask = grid.K2 < kc_u**2

    def apply(self, field_k):
        return field_k * self.mask


class CylinderCutoff(Cutoff):
    """Independent limits in the xy-plane and along z."""

    def __init__(self, grid, kc_u):
        rho2 = grid.kx3**2 + grid.ky3**2
        self.mask = (rho2 < kc_u**2) & (grid.kz3**2 < kc_u**2)

    def apply(self, field_k):
        return field_k * self.mask


class SoftCutoff(Cutoff):
    """A sigmoid roll-off instead of a hard edge, to avoid ringing."""

    def __init__(self, grid, kc_u, width_fraction=0.1):
        k_mag = torch.sqrt(grid.K2)
        self.sigma = torch.sigmoid(-(k_mag - kc_u) / (width_fraction * kc_u))

    def apply(self, field_k):
        return field_k * self.sigma


# The registry. This dict is the single source of truth for what a valid
# cutoff name is - config.py imports it to validate, and make_cutoff uses it
# to build. Adding a variant means adding one class and one line here.
CUTOFFS = {
    "No Cutoff":       NoCutoff,
    "Hard Cutoff":     HardCutoff,
    "Cylinder Cutoff": CylinderCutoff,
    "Soft Cutoff":     SoftCutoff,
}


def make_cutoff(cfg, grid) -> Cutoff:
    kc   = math.pi / grid.dx * cfg.cutoff_coeff     # m^-1
    kc_u = kc * cfg.l                               # unitless
    try:
        cls = CUTOFFS[cfg.cutoff]
    except KeyError:
        raise ValueError(
            f"Unknown cutoff {cfg.cutoff!r}. Choose from {sorted(CUTOFFS)}."
        ) from None
    return cls(grid, kc_u)