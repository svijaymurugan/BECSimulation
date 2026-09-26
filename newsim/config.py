"""Every parameter, in one immutable, validated object."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from ramps import RAMP_KINDS

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
    cutoff_kc: float | None = None        # optional override of kc, m^-1
    include_g0: bool = True
    include_g2: bool = True
    high_precision: bool = True

    # --- g2 ramp (real time only; the ground state is computed with g2 OFF) ---
    g2_ramp: str = "none"          # one of ramps.RAMP_KINDS
    g2_ramp_start: float = 0.0     # s, hold time before the ramp begins
    g2_ramp_time: float = 0.0      # s, duration (time constant for "exponential")
    a02_initial: float = 0.0       # m^3, a02 before the ramp (0 = g2 off)

    # --- g0 quench/ramp (real time only; the ground state always uses full g0) ---
    g0_ramp: str = "none"          # one of ramps.RAMP_KINDS
    g0_ramp_start: float = 0.0     # s, when the quench/ramp happens
    g0_ramp_time: float = 0.0      # s, duration (0 for a quench)
    g0_final: float = 0.0          # fraction of g0 remaining afterwards

    gamma_mod_amp: float = 0.0     # modulates gamma itself: drives the l=2 mode
    gamma_mod_freq: float = 0.0    # unitless (units of omega); sqrt(2) is quadrupole resonance

    # --- run control ---
    init_type: str = "Ground State"
    plot_fraction: float = 1 # NOTE: at the moment this has no functionality. We are just saving frames when we measure energy. If we want distinct recording frequencies for energy and plotting, we can use this.
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

    @property
    def unitless_up(self) -> float:
        """Half box length in units of the oscillator length."""
        return self.up / self.l

    @property
    def elastic_ratio(self) -> float:
        """The ratio of the g2 term to the g0 term, in oscillator units."""
        if self.G0 == 0:
            return float("inf")          # non-interacting: no interaction scale at all
        if self.G0 < 0:
            return float("nan")          # attractive: elastic ratio is not defined
        return self.G2 / self.G0

    @property #NOTE: I haven't incorporated this as a check. The function is just here for now. Maybe also add the unitless gas parameter.
    def collapse_parameter(self) -> float:
        """N|a0|/l. Attractive condensates collapse above ~0.5 (Bradley et al.).
        Meaningless for a0 > 0."""
        return self.Np * abs(self.a0) / self.l

    @property
    def tf_radius(self) -> float:
        """Thomas-Fermi radius estimate (m): R/l = (15 G0 / 4 pi)^(1/5).
        Isotropic trap, contact term only - an estimate for sizing the box."""
        return self.l * (15 * self.G0 / (4 * math.pi))**0.2

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

        if self.cutoff_kc is not None and self.cutoff_kc <= 0:
            bad.append(f"cutoff_k must be positive or None, got {self.cutoff_kc}")

        if self.g2_ramp not in RAMP_KINDS:
            bad.append(f"g2_ramp must be one of {RAMP_KINDS}, got {self.g2_ramp!r}")
        elif self.g2_ramp not in ("none", "quench") and self.g2_ramp_time <= 0:
            bad.append(f"g2_ramp={self.g2_ramp!r} needs g2_ramp_time > 0")
        if self.g2_ramp_start < 0:
            bad.append(f"g2_ramp_start must be >= 0, got {self.g2_ramp_start}")
        if self.g2_ramp != "none" and not (self.include_g2 and self.a02 != 0):
            bad.append("g2_ramp is set but the g2 term is off "
                       "(include_g2=False or a02=0) - there is nothing to ramp")
        if self.g2_ramp == "none" and self.a02_initial != 0.0:
            bad.append("a02_initial only has meaning with a g2 ramp")
        if self.g2_ramp != "none" and not (self.a02_initial * self.a02 >= 0
                                           and abs(self.a02_initial) < abs(self.a02)):
            bad.append(f"a02_initial must lie between 0 and a02 "
                       f"(got {self.a02_initial:g}, a02 = {self.a02:g})")

        if self.g0_ramp not in RAMP_KINDS:
            bad.append(f"g0_ramp must be one of {RAMP_KINDS}, got {self.g0_ramp!r}")
        elif self.g0_ramp not in ("none", "quench") and self.g0_ramp_time <= 0:
            bad.append(f"g0_ramp={self.g0_ramp!r} needs g0_ramp_time > 0")
        if not 0.0 <= self.g0_final <= 1.0:
            bad.append(f"g0_final must be in [0, 1], got {self.g0_final}")
        if self.g0_ramp != "none" and not self.include_g0:
            bad.append("g0_ramp is set but include_g0 is False - nothing to quench")

        if self.gamma_mod_amp < 0:
            bad.append(f"gamma_mod_amp must be >= 0, got {self.gamma_mod_amp}")
        if self.gamma_mod_amp != 0.0 and self.gamma_mod_freq <= 0:
            bad.append("gamma_mod_amp is set but gamma_mod_freq is not")

        if bad:
            raise ValueError("Invalid SimulationConfig:\n  - " + "\n  - ".join(bad))

    # ---------- identity ----------
    GROUND_STATE_FIELDS = (          # no annotation => a class attr, NOT a field
        "Np", "m", "omega", "a0", "a02", "up", "sigma", "y_scale", "z_scale",
        "imag_gamma", "include_g0", "include_g2", "high_precision",
        "cutoff", "cutoff_coeff", "imag_dtau", "tolerance", "cutoff_kc"
    )

    def ground_state_key(self) -> str:
        """A short hash of everything the ground state depends on."""
        payload = {k: getattr(self, k) for k in self.GROUND_STATE_FIELDS}
        payload["N"]  = self.N
        payload["nz"] = self.nz          # RESOLVED, not the Nz shorthand
        if self.g2_ramp != "none":
            payload["a02_ground_state"] = self.a02_initial
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Everything, for writing beside a run. Derived values included."""
        d = asdict(self)
        d["_derived"] = {k: getattr(self, k) for k in
                         ("nz", "l", "tau", "g0", "g2", "G0", "G2","elastic_ratio",
                          "dt", "dtau", "plot_every", "unitless_up" )}
        return d

    def summary(self) -> str:
        """A readable report of the configuration and what it implies.

        Grouped, with units and the derived quantities placed next to the
        inputs they come from. Print this at the top of every run.
        """
        d = self.to_dict()
        w = 22   # label column width

        def row(label, value, unit="", note=""):
            v = f"{value:.6g}" if isinstance(value, float) else str(value)
            line = f"  {label:<{w}} {v:>14} {unit:<8}"
            return f"{line} {note}".rstrip()

        L = []
        L.append("=" * 72)
        L.append("SimulationConfig")
        L.append("=" * 72)

        L.append("\nPhysical")
        L.append(row("particle number", self.Np))
        L.append(row("atomic mass", self.m, "kg"))
        L.append(row("trap frequency", self.omega / (2 * math.pi), "Hz",
                        f"(omega = {self.omega:.4g} rad/s)"))
        L.append(row("scattering length", self.a0 * 1e9, "nm"))
        L.append(row("scattering volume", self.a02, "m^3"))

        L.append("\nScales")
        L.append(row("oscillator length l", self.l * 1e6, "um"))
        L.append(row("trap period tau", self.tau * 1e3, "ms"))
        L.append(row("G0", self.G0, "", "contact, unitless"))
        L.append(row("G2", self.G2, "", "quadrupole, unitless"))

        L.append("\nGrid")
        L.append(row("points (N, N, nz)", f"{self.N} x {self.N} x {self.nz}"))
        L.append(row("half box", self.up * 1e6, "um",
                        f"= {self.up / self.l:.2f} l"))
        dx = 2 * self.up / self.N / self.l
        L.append(row("spacing dx", dx, "l",
                        "OK" if dx < 0.5 else "COARSE - resolves poorly"))
        L.append(row("ansatz sigma", self.sigma * 1e6, "um",
                        f"= {self.sigma / self.l:.2f} l"))

        L.append("\nReal time")
        L.append(row("duration", self.ev_time * 1e3, "ms"))
        L.append(row("steps", self.time_steps))
        L.append(row("dt", self.dt * 1e6, "us", f"= {self.dtau:.4g} tau"))
        L.append(row("z anisotropy gamma", self.gamma))
        if self.modulation_amp:
            L.append(row("modulation", self.modulation_amp, "",
                            f"at {self.modulation_freq:.4g} (unitless)"))
        else:
            L.append(row("modulation", "off", "", "-> static trap, V built once"))

        L.append("\nImaginary time")
        L.append(row("dtau", self.imag_dtau))
        L.append(row("max steps", self.imag_max_steps))
        L.append(row("tolerance", self.tolerance))
        L.append(row("z anisotropy", self.imag_gamma))

        L.append("\nNumerics")
        terms = [n for n, on in (("g0", self.include_g0),
                                    ("g2", self.include_g2 and self.a02 != 0)) if on]
        L.append(row("interactions", ", ".join(terms) or "none"))
        if self.g2_ramp == "none":
            L.append(row("g2 ramp", "none", "", "g2 on from t = 0"))
        else:
            L.append(row("g2 ramp", self.g2_ramp, "",
                         f"a02 {self.a02_initial:.3g} -> {self.a02:.3g} m^3, "
                         f"start {self.g2_ramp_start * 1e3:.3g} ms, "
                         f"duration {self.g2_ramp_time * 1e3:.3g} ms"))
        if self.g0_ramp != "none":
            L.append(row("g0 quench", self.g0_ramp, "",
                         f"1 -> {self.g0_final:g} at {self.g0_ramp_start * 1e3:g} ms"))
        L.append(row("precision", "float64" if self.high_precision else "float32"))
        if self.cutoff_kc is not None:
            L.append(row("cutoff", self.cutoff, "",
                         f"k_c = {self.cutoff_kc:.3g} m^-1 = {self.cutoff_kc * self.l:.3g} / l"))
        else:
            L.append(row("cutoff", self.cutoff, "",
                         f"{self.cutoff_coeff} x Nyquist (moves with N!)"))

        L.append("\nRecording")
        L.append(row("waist", "on" if self.track_waist else "off"))
        L.append(row("modes", "on" if self.track_modes else "off"))

        L.append("\n" + "=" * 72)
        L.append(f"  cache key: {self.ground_state_key()}")
        L.append("=" * 72)
        return "\n".join(L)

    def print_summary(self):
        print(self.summary())


def parameters_markdown(cfg) -> str:
    """A Markdown table of every field and every derived quantity.

    Generated from the dataclass, so it cannot drift out of date. Regenerate
    after any change to SimulationConfig.
    """
    d = cfg.to_dict()
    derived = d.pop("_derived")

    # field -> the comment you wrote beside it isn't accessible at runtime,
    # so keep descriptions here, keyed by name. Missing keys render blank.
    DESCRIPTIONS = {
        "Np":     "particle number",
        "m":      "atomic mass (kg)",
        "omega":  "trap frequency (rad/s)",
        "a0":     "s-wave scattering length (m)",
        "a02":    "scattering volume for the g2 term (m^3)",
        "up":     "half box length (m)",
        "N":      "grid points along x and y",
        "Nz":     "grid points along z; None means same as N",
        "sigma" : "width of the Gaussian ansatz (m)",
        "y_scale": "scale factor for y axis",
        "z_scale": "scale factor for z axis",
        "ev_time": "real-time evolution duration (s)",
        "time_steps": "real-time evolution steps",
        "imag_dtau": "imaginary-time evolution step size (s)",
        "imag_max_steps": "imaginary-time evolution max steps",
        "tolerance": "imaginary-time convergence tolerance",
        "cutoff": "type of cutoff function",
        "cutoff_coeff": "fraction of max k at which cutoff is applied",
        "include_g0": "boolean flag to include the g0 term in the Hamiltonian",
        "include_g2": "boolean flag to include the g2 term in the Hamiltonian",
        "gamma":  "z anisotropy of harmonic trap potential, gamma^2*omega^2*z in potential, real time",
        "imag_gamma": "z anisotropy of harmonic trap potential, gamma^2*omega^2*z in potential, imaginary time",
        "modulation_amp": "amplitude of trap modulation with driven harmonic trap potential",
        "modulation_freq": "frequency of trap modulation with driven harmonic trap potential",
        "high_precision": "boolean flag to use high-precision numerics",
        "init_type": "initial state type: Gaussian or Ground State",
        "plot_fraction": "fraction of steps to plot",
        "track_waist": "boolean flag to track the waist of the wavefunction",
        "track_modes": "boolean flag to track the modes of the wavefunction",
        "elastic_ratio": "ratio of the g2 term to the g0 term, in oscillator units",
        # ... fill in as you go
    }

    lines = ["<!-- GENERATED by parameters_markdown(). Do not edit by hand. -->",
             "", "## Inputs", "",
             "| Parameter | Value | Description |", "|---|---|---|"]
    for f in fields(cfg):
        v = d[f.name]
        lines.append(f"| `{f.name}` | `{v!r}` | {DESCRIPTIONS.get(f.name, '')} |")

    lines += ["", "## Derived", "",
              "| Quantity | Value |", "|---|---|"]
    for k, v in derived.items():
        val = f"{v:.6g}" if isinstance(v, float) else repr(v)
        lines.append(f"| `{k}` | `{val}` |")

    return "\n".join(lines) + "\n"


def write_parameters(cfg, path="PARAMETERS.md"):
    Path(path).write_text(parameters_markdown(cfg))
    return path