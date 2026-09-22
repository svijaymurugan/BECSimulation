"""Nonlinear interaction terms.

Every term answers the same questions: what is your name, what field do you
add to the exponent at time t, what energy do you carry, and does your
coupling depend on time. Adding a new interaction means writing one class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ramps import Ramp


class InteractionTerm(ABC):
    name: str = "unnamed"

    @abstractmethod
    def field(self, psi: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        """The real nonlinear potential at unitless time t, in position space."""

    def energy(self, field, density, dV):
        """Mean-field energy. The 1/2 avoids double counting the pair interaction."""
        return 0.5 * torch.sum(field * density) * dV

    @property
    def is_static(self) -> bool:
        """False if the coupling itself changes with time."""
        return True


class ContactTerm(InteractionTerm):
    """The usual s-wave contact interaction, G0 |psi|^2."""

    name = "g0 (contact)"

    def __init__(self, G0):
        self.G0 = G0

    def field(self, psi, t=0.0):
        return self.G0 * torch.abs(psi)**2


class QuadrupoleTerm(InteractionTerm):
    """The anisotropic g2 term, evaluated in momentum space with a cutoff."""

    name = "g2 (quadrupole)"

    def __init__(self, G2, grid, cutoff):
        self.G2 = G2
        self.cutoff = cutoff
        self.multiplier = (1/3) * grid.K2 - grid.kz3**2

    def field(self, psi, t=0.0):
        rho_k = torch.fft.fftn(torch.abs(psi)**2)
        op = self.cutoff.apply(rho_k * self.multiplier)
        return self.G2 * torch.real(torch.fft.ifftn(op))


class RampedTerm(InteractionTerm):
    """Scales another term by  initial + (1 - initial) * ramp(t).

    `initial` is the starting fraction of the full coupling (a02_initial / a02).
    With ramp=None the term is held at `initial` for good - that is what the
    ground-state search uses. Composition: it never asks what it wraps, and it
    keeps the inner term's name so recorder columns and legends are unchanged.
    """

    def __init__(self, inner: InteractionTerm, ramp: Ramp | None = None,
                 initial: float = 0.0):
        self.inner, self.ramp, self.initial = inner, ramp, initial
        self.name = inner.name

    @property
    def is_static(self) -> bool:
        return self.ramp is None

    def scale(self, t: float) -> float:
        if self.ramp is None:
            return self.initial
        return self.initial + (1.0 - self.initial) * self.ramp(t)

    def field(self, psi, t=0.0):
        s = self.scale(t)
        if s == 0.0:
            # skip the inner computation entirely - for g2 that saves an FFT pair
            return torch.zeros_like(psi.real)
        return s * self.inner.field(psi, t)

    def energy(self, field, density, dV):
        return self.inner.energy(field, density, dV)


def make_terms(cfg, grid, cutoff, *, stage="real") -> list[InteractionTerm]:
    """Build the active terms for one stage of the simulation.

    stage="imag": the ground-state search. A ramped g2 is held at its INITIAL
        value (a02_initial), and left out entirely if that is zero.
    stage="real": real-time evolution, with the ramp applied if configured.

    ORDER MATTERS for bit-exact reproduction: contact before quadrupole.
    """
    if stage not in ("real", "imag"):
        raise ValueError(f"stage must be 'real' or 'imag', got {stage!r}")

    terms: list[InteractionTerm] = []
    if cfg.include_g0:
        terms.append(ContactTerm(cfg.G0))

    if cfg.include_g2 and cfg.a02 != 0.0:
        g2 = QuadrupoleTerm(cfg.G2, grid, cutoff)
        if cfg.g2_ramp == "none":
            terms.append(g2)
        else:
            f0 = cfg.a02_initial / cfg.a02          # G2 is linear in a02
            if stage == "real":
                terms.append(RampedTerm(g2, Ramp.from_config(cfg), initial=f0))
            elif f0 != 0.0:
                terms.append(RampedTerm(g2, None, initial=f0))
    return terms