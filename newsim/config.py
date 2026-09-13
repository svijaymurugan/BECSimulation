"""Every parameter, in one immutable, validated object."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict

import scipy.constants as const

from cutoff import CUTOFFS       # the registry IS the list of valid names

HBAR = const.hbar


@dataclass(frozen=True)
class SimulationConfig:
    # --- physical inputs (no defaults: you must state these) ---
    Np: float                      # particle number
    m: float                       # atomic mass, kg
    omega: float                   # trap frequency, rad/s
    a0: float                      # s-wave scattering length, m
    a02: float                     # scattering volume for the g2 term, m^3
    up: float                      # half box length, m
    N: int                         # grid points along x and y

    # --- geometry ---
    Nz: int | None = None          # grid points along z; None means "same as N"
    sigma: float = 10e-6           # width of the Gaussian ansatz, m
    y_scale: float = 1.0
    z_scale: float = 1.0

    # --- trap ---
    gamma: float = 1.0             # z anisotropy, real time
    imag_gamma: float = 1.0        # z anisotropy, imaginary time
    modulation_amp: float = 0.0
    modulation_freq: float = 0.0

    # --- real-time evolution ---
    ev_time: float = 0.015
    time_steps: int = 10_000

    # --- imaginary-time evolution ---
    imag_dtau: float = 1e-4
    imag_max_steps: int = 10_000
    tolerance: float = 1e-6

    # --- numerics ---
    cutoff: str = "Hard Cutoff"
    cutoff_coeff: float = 0.9
    include_g0: bool = True
    include_g2: bool = True
    high_precision: bool = True

    # --- run control ---
    init_type: str = "Ground State"
    plot_fraction: float = 0.01
    track_waist: bool = False
    track_modes: bool = False

    @property
    def nz(self) -> int:
        return self.N if self.Nz is None else self.Nz

    @property
    def l(self) -> float:
        """Harmonic-oscillator length: the natural unit of length (m)."""
        return math.sqrt(HBAR / (self.m * self.omega))

    @property
    def tau(self) -> float:
        return 1.0 / self.omega

    @property
    def g0(self) -> float:
        return 4 * math.pi * HBAR**2 * self.a0 * self.Np / self.m

    @property
    def g2(self) -> float:
        return 6 * math.sqrt(5) * math.pi * HBAR**2 * self.a02 * self.Np / self.m

    @property
    def G0(self) -> float:
        return self.g0 / (self.l**3 * HBAR * self.omega)

    @property
    def G2(self) -> float:
        return self.g2 / (self.l**5 * HBAR * self.omega)

    @property
    def dt(self) -> float:
        return self.ev_time / self.time_steps

    @property
    def dtau(self) -> float:
        return self.dt / self.tau

    @property
    def plot_every(self) -> int:
        """Whole number of steps between frames. Approximate by design (see Q3)."""
        return max(1, round(self.time_steps * self.plot_fraction))

    # ---------- validation: runs at construction, before anything is used ----------
    def __post_init__(self):
        bad = []

        for name in ("N", "time_steps", "imag_max_steps"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                bad.append(f"{name} must be a positive int, got {v!r}")

        if self.Nz is not None and (not isinstance(self.Nz, int) or self.Nz <= 0):
            bad.append(f"Nz must be a positive int or None, got {self.Nz!r}")

        for name in ("Np", "m", "omega", "up", "sigma", "ev_time",
                     "imag_dtau", "tolerance"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v <= 0:     # int is fine!
                bad.append(f"{name} must be a positive number, got {v!r}")

        if self.cutoff not in CUTOFFS:
            bad.append(f"cutoff must be one of {sorted(CUTOFFS)}, got {self.cutoff!r}")

        if self.init_type not in ("Gaussian", "Ground State"):
            bad.append(f"init_type must be 'Gaussian' or 'Ground State', "
                       f"got {self.init_type!r}")

        if not 0.0 < self.plot_fraction <= 1.0:
            bad.append(f"plot_fraction must be in (0, 1], got {self.plot_fraction}")

        # physical sanity, now that l is computable
        if not 1e-9 < self.l < 1e-3:
            bad.append(f"oscillator length l = {self.l:.3e} m is outside "
                       f"any plausible range - check m and omega")
        if not 0.1 < self.sigma / self.l < 10:
            bad.append(f"sigma/l = {self.sigma/self.l:.2f}; the ansatz is not on "
                       f"the scale of the trap")

        if bad:
            raise ValueError("Invalid SimulationConfig:\n  - " + "\n  - ".join(bad))

    # ---------- identity ----------
    GROUND_STATE_FIELDS = (          # no annotation => a class attr, NOT a field
        "Np", "m", "omega", "a0", "a02", "up", "sigma", "y_scale", "z_scale",
        "imag_gamma", "include_g0", "include_g2", "high_precision",
        "cutoff", "cutoff_coeff", "imag_dtau", "tolerance",
    )

    def ground_state_key(self) -> str:
        """A short hash of everything the ground state depends on."""
        payload = {k: getattr(self, k) for k in self.GROUND_STATE_FIELDS}
        payload["N"]  = self.N
        payload["nz"] = self.nz          # RESOLVED, not the Nz shorthand
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Everything, for writing beside a run. Derived values included."""
        d = asdict(self)
        d["_derived"] = {k: getattr(self, k) for k in
                         ("nz", "l", "tau", "g0", "g2", "G0", "G2",
                          "dt", "dtau", "plot_every")}
        return d