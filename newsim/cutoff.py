"""Momentum-space filters for the g2 term.

Each variant precomputes whatever it needs once, then exposes a single method.
Callers never learn which one they have.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import warnings


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
    """Build the momentum-space filter for the g2 term.

    The cutoff is PHYSICS, so specify it as a physical momentum with
    cfg.cutoff_k (m^-1). Then runs at different N or box size simulate the
    same Hamiltonian, and only the resolution changes.

    cfg.cutoff_coeff - a fraction of the grid's Nyquist momentum - is kept as
    a fallback for old configs, but beware: it moves whenever N or the box
    changes, which silently changes the physics.
    """
    try:
        cls = CUTOFFS[cfg.cutoff]
    except KeyError:
        raise ValueError(
            f"Unknown cutoff {cfg.cutoff!r}. Choose from {sorted(CUTOFFS)}."
        ) from None

    # The grid can only represent momenta up to pi/dx along each axis. With
    # Nz != N the z spacing differs, so the binding limit is the smaller one.
    k_nyq_xy = math.pi / grid.dx                        # m^-1
    k_nyq_z  = math.pi / grid.dz                        # m^-1
    k_grid   = min(k_nyq_xy, k_nyq_z)

    if cfg.cutoff_kc is not None:
        kc = cfg.cutoff_kc                               # physical, m^-1
    else:
        kc = k_nyq_xy * cfg.cutoff_coeff                # legacy: tied to the grid

    if cls is not NoCutoff:
        if kc >= k_grid:
            raise ValueError(
                f"cutoff {kc:.3e} m^-1 is at or above the grid's Nyquist momentum "
                f"{k_grid:.3e} m^-1, so the filter removes nothing and the g2 "
                f"product aliases. Increase N (need dx < {math.pi/kc*1e6:.3f} um).")
        if kc > k_grid / 1.5:
            warnings.warn(
                f"cutoff {kc:.3e} m^-1 is within 1.5x of the grid's Nyquist "
                f"momentum {k_grid:.3e} m^-1; momenta near the cutoff are poorly "
                f"resolved. Consider N >= "
                f"{math.ceil(2 * cfg.up * 1.5 * kc / math.pi)}.",
                RuntimeWarning)

    return cls(grid, kc * cfg.l)                        # unitless, for the masks