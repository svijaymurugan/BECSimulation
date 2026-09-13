"""Nonlinear interaction terms.

Every term answers the same three questions:
  what is your name, what field do you add to the exponent, what energy do you carry.
Adding a new interaction means writing one class - nothing else changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class InteractionTerm(ABC):
    name: str = "unnamed"

    @abstractmethod
    def field(self, psi: torch.Tensor) -> torch.Tensor:
        """The real nonlinear potential this term contributes, in position space."""

    def energy(self, field, density, dV):
        """Mean-field energy. The 1/2 avoids double counting the pair interaction."""
        return 0.5 * torch.sum(field * density) * dV


class ContactTerm(InteractionTerm):
    """The usual s-wave contact interaction, G0 |psi|^2."""

    name = "g0 (contact)"

    def __init__(self, G0):
        self.G0 = G0

    def field(self, psi):
        return self.G0 * torch.abs(psi)**2


class QuadrupoleTerm(InteractionTerm):
    """The anisotropic g2 term, evaluated in momentum space with a cutoff."""

    name = "g2 (quadrupole)"

    def __init__(self, G2, grid, cutoff):
        self.G2 = G2
        self.cutoff = cutoff
        self.multiplier = (1/3) * grid.K2 - grid.kz3**2

    def field(self, psi):
        rho_k = torch.fft.fftn(torch.abs(psi)**2)
        op = self.cutoff.apply(rho_k * self.multiplier)
        return self.G2 * torch.real(torch.fft.ifftn(op))


def make_terms(cfg, grid, cutoff) -> list[InteractionTerm]:
    """Build the active terms.

    ORDER MATTERS for bit-exact reproduction: the original computed
    `first_g0 + q1` (line 469), so contact must come before quadrupole.
    Floating-point addition is not associative; see Part 0.
    """
    terms: list[InteractionTerm] = []
    if cfg.include_g0:
        terms.append(ContactTerm(cfg.G0))
    if cfg.include_g2 and cfg.a02 != 0.0:
        terms.append(QuadrupoleTerm(cfg.G2, grid, cutoff))
    return terms