"""Time evolution.

The split-step algorithm is written once, here. Real and imaginary time differ
in five specific ways, and each one is a method a subclass may override.

Do the following to improve error scaling in imaginary time: This averages the operator over each half-step, rather than using the field at the start of the step. 
This is a predictor-corrector method, and is second-order accurate in dtau. It is also more stable and we can use larger dtau. Claude says 10x larger.

You can do this for g2 as well but it'll cost more FFTs. So choose wisely.

def _apply_terms(self, psi, half):
    W0 = self._total_field(psi)                    # predictor: field now
    psi_pred = psi * torch.exp(self.clip(self.phase * W0 * half))
    W1 = self._total_field(psi_pred)               # field after a trial kick
    W = 0.5 * (W0 + W1)                            # trapezoid
    return psi * torch.exp(self.clip(self.phase * W * half))
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Evolution(ABC):
    # ---- the one thing every subclass MUST supply ----
    @property
    @abstractmethod
    def phase(self) -> complex:
        """-1j for real time, -1.0 for imaginary time."""

    def __init__(self, grid, potential, terms, dtau, max_steps,
                 recorder, measure_every = 1,monitors=()):
        self.grid = grid
        self.potential = potential
        self.terms = terms
        self.dtau = dtau
        self.max_steps = max_steps
        self.recorder = recorder
        self.monitors = monitors
        self.ke_divisor = grid.n_points
        self.KE = 0.5 * grid.K2
        self.steps_taken = 0
        self.measure_every = measure_every

    # ---- the other four differences, with harmless defaults ----
    def clip(self, exponent):
        """Imaginary time overrides this to stop exp() overflowing."""
        return exponent

    def after_step(self, psi, i):
        """Imaginary time overrides this to renormalise."""
        return psi

    def should_stop(self, i) -> bool:
        """Imaginary time overrides this to test convergence."""
        return False

    # ---- the algorithm: written ONCE, never subclassed ----
    def run(self, psi):
        dtau = self.dtau
        exp_K = torch.exp(self.phase * self.KE * dtau)

        static = self.potential.is_static
        V = self.potential(0.0)
        exp_V = torch.exp(self.phase * V * (dtau / 2))

        i = -1
        for i in range(self.max_steps):
            if not static:                       # finding 12: skipped entirely
                V = self.potential(i * dtau)     #   when the trap is static
                exp_V = torch.exp(self.phase * V * (dtau / 2))

            measure = (i % self.measure_every == 0)
            psi, energies = self.step(psi, V, exp_V, exp_K, dtau, measure=measure)
            psi = self.after_step(psi, i)

            if measure:
                self.recorder.record_energies(energies)
                self.recorder.record_waist(psi)
                self.recorder.record_frames(psi)
            

            for monitor in self.monitors:
                monitor.check(psi, i, self)

            if self.should_stop(i):
                break

        self.steps_taken = i + 1
        return psi

    def step(self, psi, V, exp_V, exp_K, dtau, measure=True):
        """One symmetric split step. No branch on real vs imaginary anywhere:
        the difference is carried entirely by `self.phase`."""
        grid = self.grid
        half = dtau / 2
        energies = None

        if measure:

            fields = [term.field(psi) for term in self.terms]

            # --- measure (see finding 06 for why it happens HERE) ---
            density = torch.abs(psi)**2
            energies = {t.name: t.energy(f, density, grid.dV)
                        for t, f in zip(self.terms, fields)}
            energies["potential"] = torch.sum(V * density) * grid.dV

            psi_k_meas = torch.fft.fftn(psi)
            energies["kinetic"] = (torch.sum(self.KE * torch.abs(psi_k_meas)**2)
                                * grid.dV / self.ke_divisor)  # Phase B seam - see finding 04

            # Sum in the original's order: ((KE + pot) + g0) + g2.
            # Float addition is not associative; see Part 0.
            total = energies["kinetic"] + energies["potential"]
            for term in self.terms:
                total = total + energies[term.name]
            energies["total"] = total

        # --- first half step in position space ---
        psi = psi * exp_V
        psi = self._apply_terms(psi, fields, half)

        # --- full step in momentum space ---
        psi_k = torch.fft.fftn(psi)

        if measure:
            self.recorder.record_modes(psi_k)

        psi_k = psi_k * exp_K
        #energies["kinetic"] = (torch.sum(self.KE * torch.abs(psi_k)**2)
        #                       * grid.dV / grid.n_points) / norm  # Phase B seam - see finding 04
        psi = torch.fft.ifftn(psi_k)

        # --- second half step in position space ---
        psi = psi * exp_V
        fields = [term.field(psi) for term in self.terms]
        psi = self._apply_terms(psi, fields, half)

        return psi, energies

    def _apply_terms(self, psi, fields, half):
        if not fields:
            return psi
        total = fields[0]
        for f in fields[1:]:
            total = total + f
        return psi * torch.exp(self.clip(self.phase * total * half))

class RealTimeEvolution(Evolution):
    """Unitary evolution. Everything the base class does by default is correct."""

    phase = -1j          # a plain class attribute satisfies the abstract property


class ImaginaryTimeEvolution(Evolution):
    """Gradient descent towards the ground state, in imaginary time."""

    phase = -1.0

    def __init__(self, *args, tolerance=1e-6, check_every=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.tolerance = tolerance
        self.check_every = check_every
        self._previous_energy = float("inf")
        self.converged = False

    # --- difference 1: the exponent is a real decay, so it can overflow ---
    def clip(self, exponent):
        return torch.clip(exponent, min=-50, max=50)

    # --- difference 2: imaginary time is not norm-preserving ---
    def after_step(self, psi, i):
        mag = self.grid.unitless_norm(psi).item()
        return psi * (1.0 / mag)**0.5

    # --- difference 3: stop on convergence, not on a step count ---
    def should_stop(self, i) -> bool:
        if i % self.check_every or i == 0:
            return False
        current = self.recorder.last_total()
        change = abs(current - self._previous_energy) / self._previous_energy
        if change < self.tolerance:
            self.converged = True
            print(f"Converged at step {i}: E = {current:.6f}, change = {change:.3e}")
            return True
        self._previous_energy = current
        return False