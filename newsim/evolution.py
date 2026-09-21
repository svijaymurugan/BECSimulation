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


NOTE ON ADAPTIVE IMAGINARY TIME AT THE BOTTOM OF THIS SCRIPT
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import time


class Evolution(ABC):
    # ---- the one thing every subclass MUST supply ----
    @property
    @abstractmethod
    def phase(self) -> complex:
        """-1j for real time, -1.0 for imaginary time."""

    def __init__(self, grid, potential, terms, dtau, max_steps,
                 recorder, measure_every = 10,monitors=()):
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

        #if len(self.recorder._rows)>0:
        #    raise RuntimeError(
        #        f"This Recorder already holds {len(self.recorder)} rows from a "
        #        f"previous run. Build a fresh Recorder for each run - re-run the "
        #        f"cell that constructs it, not just the cell that calls run().")

        dtau = self.dtau
        exp_K = torch.exp(self.phase * self.KE * dtau)

        static = self.potential.is_static
        V = self.potential(0.0)
        exp_V = torch.exp(self.phase * V * (dtau / 2))

        i = -1
        start_time = time.perf_counter()
        mid_time = start_time
        for i in range(self.max_steps):
            if not static:                       # finding 12: skipped entirely
                V = self.potential(i * dtau)     #   when the trap is static
                exp_V = torch.exp(self.phase * V * (dtau / 2))

            measure = (i % self.measure_every == 0)
            psi, energies = self.step(psi, V, exp_V, exp_K, dtau, measure=measure)
            psi = self.after_step(psi, i)

            if measure:
                self.recorder.record_energies(energies)
            

            for monitor in self.monitors:
                monitor.check(psi, i, self)

            if self.should_stop(i):
                break

            if i % (self.max_steps//10) == 0:
                print(f"{i/self.max_steps * 100} % Completed: {time.perf_counter() - mid_time:.1f} s elapsed, Total time: {time.perf_counter() - start_time:.1f} s")
                mid_time = time.perf_counter()

        self.steps_taken = i + 1
        print(f"Total time: {time.perf_counter() - start_time:.1f} s for {self.steps_taken} steps")
        return psi

    #@torch.compile
    def step(self, psi, V, exp_V, exp_K, dtau, measure=True):
        """One symmetric split step. No branch on real vs imaginary anywhere:
        the difference is carried entirely by `self.phase`."""
        grid = self.grid
        half = dtau / 2
        energies = None

        if measure:

            self.recorder.record_waist(psi)
            self.recorder.record_frames(psi) #NOTE if this is too frequent, introduce another bool plot which is controlled by cfg.plot_every (just like measure_every)

            meas_fields = [term.field(psi) for term in self.terms]

            # --- measure (see finding 06 for why it happens HERE) ---
            density = torch.abs(psi)**2
            energies = {t.name: t.energy(f, density, grid.dV)
                        for t, f in zip(self.terms, meas_fields)}
            energies["potential"] = torch.sum(V * density) * grid.dV

            psi_k_meas = torch.fft.fftn(psi)
            energies["kinetic"] = (torch.sum(self.KE * torch.abs(psi_k_meas)**2)
                                * grid.dV / self.ke_divisor)  

            # Sum in the original's order: ((KE + pot) + g0) + g2.
            # Float addition is not associative; see Part 0.
            total = energies["kinetic"] + energies["potential"]
            for term in self.terms:
                total = total + energies[term.name]
            energies["total"] = total

        # --- first half step in position space ---
        psi = psi * exp_V

        fields = [term.field(psi) for term in self.terms]
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

        assert self.check_every % self.measure_every == 0, "check_every must be a multiple of measure_every"

    # --- difference 1: the exponent is a real decay, so it can overflow ---
    def clip(self, exponent):
        return torch.clip(exponent, min=-50, max=50)

    # --- difference 2: imaginary time is not norm-preserving ---
    def after_step(self, psi, i):
        mag = self.grid.unitless_norm(psi).item()
        return psi * (1.0 / mag)**0.5

    # --- difference 3: stop on convergence, not on a step count ---
    def should_stop(self, i) -> bool:
        if i % self.check_every != 0 or i == 0:
            return False
        current = self.recorder.last_total()
        change = abs(current - self._previous_energy) / self._previous_energy
        print(f"Step {i}: E = {current:.6f}, change = {change:.3e}")
        if change < self.tolerance:
            self.converged = True
            print(f"Converged at step {i}: E = {current:.6f}, change = {change:.3e}")
            return True
        self._previous_energy = current
        return False


"""
Note on imaginary-time step size
--------------------------------
`imag_dtau` is fixed for the whole ITE run. Two known limitations, both
understood and deliberately not addressed:

1. First-order accuracy in the nonlinear term. The interaction field
   G0|psi|^2 is frozen at the start of each half-kick. In real time the kick
   is a pure phase, so |psi|^2 is genuinely constant during the sub-step and
   the scheme stays second order. In imaginary time the kick is a real decay,
   |psi|^2 changes during the sub-step, and the frozen field is wrong by
   O(dtau). Measured: virial residual/V scales as 5.78e-2 -> 2.87e-2 ->
   1.43e-2 -> 7.14e-3 as dtau halves from 5e-3 (ratio 2.0, first order),
   against ratios of 4.0 for real-time energy drift.

   Remedy if needed: predictor-corrector on the field only, in imaginary
   time only --
       W0 = total_field(psi)
       psi_pred = psi * exp(phase * W0 * half)
       W  = 0.5 * (W0 + total_field(psi_pred))
       psi = psi * exp(phase * W * half)
   Cost: the interaction field is evaluated twice per half-kick. Cheap for
   the contact term, two extra FFT pairs for the quadrupole term. Buys
   second-order convergence, so a ~10x larger dtau for the same accuracy --
   usually a net win once step count dominates.

2. No adaptation. High-energy components decay fast; the slow part is
   separating the ground state from the lowest excited states, so a step
   size that is right at the start is conservative later.

   Simplest useful scheme, decided at convergence checks (never per step):
   monitor the energy at each check; if it RISES, dtau was too large --
   halve it and rebuild exp_K and exp_V. If the relative change has been
   below some threshold for several consecutive checks, multiply dtau by
   ~1.5 up to a ceiling. Both actions require rebuilding the cached
   exponentials, which is why they belong at check boundaries and not in
   the hot loop.

   Two caveats before implementing:
     - `imag_dtau` is part of GROUND_STATE_FIELDS, so it feeds the cache
       hash. With an adaptive schedule the stored dtau no longer identifies
       the computation; hash the initial dtau AND the adaptation rule, or
       the cache will report matches that are not matches.
     - Cheaper alternative that needs no loop changes at all: run ITE twice,
       coarse dtau to near-convergence then fine dtau to refine. Two fixed-
       dtau runs chained. Try this before building adaptivity.

   Not currently worth it: ITE converges in seconds at test grid sizes and
   runs once per unique config thanks to the ground-state cache.
"""