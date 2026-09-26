"""Trap potentials, in oscillator units.

A potential is a callable of unitless time. It knows whether it depends on
time at all, which is what lets the evolution loop skip 9,999 rebuilds.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch


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
    """A box potential with hard walls at |x|, |y|, |z| = L/2.
    
    TODO: BoxPotential is broken: self._V[grid.ux3.abs() > 0.5] = inf indexes a (N,N,nz) tensor with an (N,1,1) mask, 
    and inf in the exponent gives NaN. Build it by broadcasting and use a large finite wall, or delete it — nothing constructs it.
    
    """

    def __init__(self, grid):
        self._V = torch.zeros(grid.shape, dtype=grid.real_dtype, device=grid.device)
        self._V[grid.ux3.abs() > 0.5] = float('inf')
        self._V[grid.uy3.abs() > 0.5] = float('inf')
        self._V[grid.uz3.abs() > 0.5] = float('inf')

    def __call__(self, t):
        return self._V

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


def make_potential(cfg, grid, gamma) -> Potential:
    """Build the trap for one evolution. `gamma` is passed explicitly so that
    imaginary time can use a different anisotropy."""
    if cfg.gamma_mod_amp != 0.0:
        trap = GammaModulatedTrap(grid, gamma, cfg.gamma_mod_amp, cfg.gamma_mod_freq)
    else:
        trap = HarmonicTrap(grid, gamma)
    if cfg.modulation_amp == 0.0:
        return trap
    return ModulatedHarmonicTrap(trap, cfg.modulation_amp, cfg.modulation_freq)