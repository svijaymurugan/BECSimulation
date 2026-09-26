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
    """Scales another term from `initial` to `final` along a ramp profile.

    Both are fractions of the inner term's full strength. Ramping g2 up is
    initial=a02_initial/a02, final=1; quenching g0 off is initial=1, final=0.
    With ramp=None the term is held at `initial` for good - that is what the
    ground-state search uses.

    It never asks what it wraps, and it keeps the inner term's name, so
    recorder columns and plot legends are unchanged.
    """

    def __init__(self, inner: InteractionTerm, ramp: Ramp | None = None,
                 initial: float = 0.0, final: float = 1.0):
        self.inner, self.ramp = inner, ramp
        self.initial, self.final = initial, final
        self.name = inner.name

    @property
    def is_static(self) -> bool:
        return self.ramp is None

    def scale(self, t: float) -> float:
        if self.ramp is None:
            return self.initial
        return self.initial + (self.final - self.initial) * self.ramp(t)

    def field(self, psi, t=0.0):
        s = self.scale(t)
        if s == 0.0:
            # skip the inner computation entirely - after a g0 quench, or
            # before a g2 ramp that starts from zero
            return torch.zeros_like(psi.real)
        return s * self.inner.field(psi, t)

    def energy(self, field, density, dV):
        return self.inner.energy(field, density, dV)


def make_terms(cfg, grid, cutoff, *, stage="real") -> list[InteractionTerm]:
    """Build the active terms for one stage of the simulation.

    stage="imag": the ground-state search. g0 is always at full strength (the
        quench happens in real time); a ramped g2 is held at its INITIAL value,
        and left out entirely if that is zero.
    stage="real": ramps and quenches applied.

    ORDER MATTERS for bit-exact reproduction: contact before quadrupole.
    """
    if stage not in ("real", "imag"):
        raise ValueError(f"stage must be 'real' or 'imag', got {stage!r}")

    terms: list[InteractionTerm] = []

    if cfg.include_g0:
        contact = ContactTerm(cfg.G0)
        if stage == "real" and cfg.g0_ramp != "none":
            terms.append(RampedTerm(
                contact,
                Ramp(cfg.g0_ramp, start=cfg.g0_ramp_start / cfg.tau,
                     duration=cfg.g0_ramp_time / cfg.tau),
                initial=1.0, final=cfg.g0_final))
        else:
            terms.append(contact)              # ITE always sees the full g0

    if cfg.include_g2 and cfg.a02 != 0.0:
        g2 = QuadrupoleTerm(cfg.G2, grid, cutoff)
        f0 = cfg.a02_initial / cfg.a02
        if cfg.g2_ramp == "none":
            terms.append(g2)
        elif stage == "real":
            terms.append(RampedTerm(g2, Ramp.from_config(cfg),
                                    initial=f0, final=1.0))
        elif f0 != 0.0:
            terms.append(RampedTerm(g2, None, initial=f0))

    return terms