"""Measurement. Everything accumulates on the GPU; one transfer at the end."""
from __future__ import annotations

import warnings
import numpy as np
import torch


class Recorder:
    """Collects per-step diagnostics without stalling the GPU.

    Your original did `[e.real.item() for e in energy_tuple]` every step. Each
    .item() blocks the CPU until the entire GPU queue drains - five stalls per
    step, 50,000 over a run. Keeping the tensors on the device and stacking
    once at the end costs 560 kB of VRAM and zero stalls.

    You already solved this problem for the mode amplitudes (line 604 stacks
    them at the end). This applies the same fix to the energies.
    """

    def __init__(self, term_names, measure_every=1, dt = None, track_modes=True, radius_squared=None, dV=None, track_frames=True):
        self.columns = ["kinetic", "potential", *term_names, "total"]
        self.track_modes = track_modes
        self.measure_every, self.track_frames = measure_every, track_frames
        self.dt = dt
        self._rows = []
        self._frames_xy, self._frames_xz = [], []
        self._kx, self._kz = [], []
        self._waist = []
        self._r2, self._dV = radius_squared, dV

    @classmethod
    def energies_only(cls, term_names, **kwargs):
        """Energies and nothing else. The right Recorder for imaginary time:
        ITE only needs the total for its convergence check, and a movie of a
        ground-state search is not something anyone wants."""
        return cls(term_names, track_modes=False, track_frames=False,
                   radius_squared=None, **kwargs)

    def record_energies(self, energies):
        self._rows.append(
            torch.stack([energies[c] for c in self.columns]).detach())

    def record_modes(self, psi_k):
        if not self.track_modes:
            return
        self._kx.append(torch.fft.fftshift(torch.abs(psi_k[:, 0, 0])).detach())
        self._kz.append(torch.fft.fftshift(torch.abs(psi_k[0, 0, :])).detach())

    def record_waist(self, psi):
        if self._r2 is None:
            return
        self._waist.append(
            (torch.sum(torch.abs(psi)**2 * self._r2) * self._dV).detach())

    # ---- readers: these DO transfer, and are called once, at the end ----
    def last_total(self) -> float:
        """The only per-step transfer, and only imaginary time asks for it."""
        return self._rows[-1][-1].item()

    def energies(self):
        return torch.stack(self._rows).cpu().numpy()

    def modes(self):
        if not self._kx:
            return None, None
        return (torch.stack(self._kx).cpu().numpy(),
                torch.stack(self._kz).cpu().numpy())

    def waist(self):
        if not self._waist:
            return None
        return torch.stack(self._waist).cpu().numpy()

    def times(self):
        """Physical time for each recorded row. Never reconstruct this by hand."""
        if self.dt is None:
            raise ValueError("Recorder was built without dt; cannot give times.")
        return np.arange(len(self._rows)) * self.dt * self.measure_every

    # Recorder
    def record_frames(self, psi):
        if not self.track_frames:
            return
        rho = torch.abs(psi)**2
        self._frames_xy.append(torch.sum(rho, dim=2).detach().to(torch.float32))   # integrate out z
        self._frames_xz.append(torch.sum(rho, dim=1).detach().to(torch.float32))   # integrate out y

    def frames(self):
        if not self._frames_xy:
            return None, None
        return (torch.stack(self._frames_xy).cpu().numpy(),
                torch.stack(self._frames_xz).cpu().numpy())

class NormMonitor:
    """Real-time evolution is unitary. Warn if it stops being so."""

    def __init__(self, grid, every=100, tol=1e-8):
        self.grid, self.every, self.tol = grid, every, tol

    def check(self, psi, i, evolution):
        if i % self.every:
            return
        n = self.grid.unitless_norm(psi).item()
        if abs(n - 1.0) > self.tol:
            warnings.warn(f"norm drifted to {n:.12f} at step {i}", RuntimeWarning)


class EnergyMonitor:
    """With a static trap, total energy should be constant."""

    def __init__(self, every=500, rel_tol=1e-4):
        self.every, self.rel_tol = every, rel_tol
        self._reference = None

    def check(self, psi, i, evolution):
        if i % self.every:
            return
        e = evolution.recorder.last_total()
        if self._reference is None:
            self._reference = e
            return
        drift = abs(e - self._reference) / abs(self._reference)
        if drift > self.rel_tol:
            warnings.warn(f"energy drifted {drift:.2e} by step {i} "
                          f"- consider a smaller dt", RuntimeWarning)