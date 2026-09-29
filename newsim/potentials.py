"""Trap potentials, in oscillator units.

A potential is a callable of unitless time. It knows whether it depends on
time at all, which is what lets the evolution loop skip 9,999 rebuilds.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from ramps import Ramp


class Potential(ABC):
    @abstractmethod
    def __call__(self, t: float) -> torch.Tensor:
        """The potential at unitless time t."""

    @property
    def is_static(self) -> bool:
        return True


class HarmonicTrap(Potential):
    """V = 1/2 (x^2 + y^2 + gamma^2 z^2), built once on the device."""

    def __init__(self, grid, gamma):
        self.gamma = gamma
        self._V = 0.5 * (grid.ux3**2 + grid.uy3**2 + gamma**2 * grid.uz3**2)

    def __call__(self, t):
        return self._V


class ModulatedHarmonicTrap(Potential):
    """Scales any potential by (1 + A sin(w t))^2.

    It never asks what it wraps. That is the point: a lattice, a box trap or
    anything you write later can be modulated without modifying this class,
    and this class can be swapped out without modifying the trap.
    """

    def __init__(self, inner: Potential, amp: float, freq: float):
        self.inner, self.amp, self.freq = inner, amp, freq

    @property
    def is_static(self) -> bool:
        return self.amp == 0.0 and self.inner.is_static

    def __call__(self, t):
        return self.inner(t) * (1.0 + self.amp * math.sin(self.freq * t))**2

class BoxPotential(Potential):
    """Smooth-walled rectangular box, optionally squeezed in time.

        V = height * [ s(|x| - L_perp) + s(|y| - L_perp) + s(|z| - L_z) ]
        with s(u) = sigmoid(u / width)

    A hard wall would need infinite height (exp(-inf) -> NaN) and would put a
    step on the grid; the Gibbs ringing from a step lives at high k, which is
    exactly where g2 is most sensitive. A sigmoid wall a couple of grid points
    wide is finite, resolved, and still steep compared with the healing length.
    """

    def __init__(self, grid, L_perp, L_z, height=200.0, width=0.0,
                 ramp=None, squeeze_to=1.0):
        self.grid = grid
        self.L_perp, self.L_z, self.height = L_perp, L_z, height
        self.width = width if width > 0 else 2 * grid.dxu
        self.ramp, self.squeeze_to = ramp, squeeze_to
        self._static = None if ramp is not None else self._build(1.0)

    def _build(self, s):
        g, w = self.grid, self.width
        def wall(u, L):
            return torch.sigmoid((torch.abs(u) - s * L) / w)
        return self.height * (wall(g.ux3, self.L_perp) + wall(g.uy3, self.L_perp)
                              + wall(g.uz3, self.L_z))

    @property
    def is_static(self) -> bool:
        return self.ramp is None

    def __call__(self, t):
        if self._static is not None:
            return self._static
        return self._build(1.0 + (self.squeeze_to - 1.0) * self.ramp(t))

class GammaModulatedTrap(Potential):
    """Harmonic trap whose z anisotropy is modulated:
        gamma(t) = gamma0 * (1 + A sin(w t)),  V = 1/2 (x^2 + y^2 + gamma(t)^2 z^2)

    Unlike ModulatedHarmonicTrap, which scales the WHOLE trap (isotropic, so it
    parametrically drives only the monopole at 2*sqrt(5) omega), this changes z
    relative to x and y. The perturbation contains a quadrupole piece, so it
    drives the l=2 mode DIRECTLY - a linear drive, resonant at sqrt(2) omega.
    """

    def __init__(self, grid, gamma, amp, freq):
        self.gamma, self.amp, self.freq = gamma, amp, freq
        self._Vxy = 0.5 * (grid.ux3**2 + grid.uy3**2)
        self._Vz = 0.5 * grid.uz3**2

    @property
    def is_static(self) -> bool:
        return self.amp == 0.0

    def __call__(self, t):
        g = self.gamma * (1.0 + self.amp * math.sin(self.freq * t))
        return self._Vxy + (g * g) * self._Vz

class ZeroPotential(Potential):
    """No external potential: a periodic cell. With V = 0 a uniform psi is an
    EXACT eigenstate, so a uniform gas needs no ground-state search at all -
    and the Bogoliubov dispersion applies exactly, with no Thomas-Fermi or
    local-density approximation anywhere.
    """

    def __init__(self, grid):
        self._V = torch.zeros((1, 1, 1), dtype=grid.real_dtype, device=grid.device)

    def __call__(self, t):
        return self._V


def make_potential(cfg, grid, gamma) -> Potential:
    """Build the trap for one evolution."""
    if cfg.trap_kind == "none":
        return ZeroPotential(grid)    
    if cfg.trap_kind == "box":
        ramp = None
        if cfg.box_squeeze != "none":
            ramp = Ramp(cfg.box_squeeze, start=cfg.box_squeeze_start / cfg.tau,
                        duration=cfg.box_squeeze_time / cfg.tau)
        return BoxPotential(grid, cfg.box_L_perp, cfg.box_L_z, cfg.box_height,
                            cfg.box_width, ramp=ramp, squeeze_to=cfg.box_squeeze_to)

    if cfg.gamma_mod_amp != 0.0:
        trap = GammaModulatedTrap(grid, gamma, cfg.gamma_mod_amp, cfg.gamma_mod_freq)
    else:
        trap = HarmonicTrap(grid, gamma)
    if cfg.modulation_amp == 0.0:
        return trap
    return ModulatedHarmonicTrap(trap, cfg.modulation_amp, cfg.modulation_freq)