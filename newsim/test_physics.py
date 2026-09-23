"""Conservation laws the simulation must obey, whatever the parameters."""
import math

import numpy as np
import torch
from pathlib import Path

import torch
from config import SimulationConfig 
from grid import Grid
from cutoff import make_cutoff 
from interactions import make_terms
from potentials import make_potential 
from evolution import RealTimeEvolution, ImaginaryTimeEvolution
from diagnostics import Recorder, NormMonitor
import storage, plotting
from dataclasses import replace
from ramps import Ramp, RAMP_KINDS
from interactions import RampedTerm, ContactTerm

HBAR = 1.054571817e-34


def small(**overrides):
    """A grid that is tiny but still resolves the physics.

    l = 1.486 um for these constants, so:
        up = 7.5e-6  ->  box = +/-5.0 l   (tails fit)
        N  = 32      ->  dx  = 0.32 l     (Gaussian resolved)
    Do NOT shrink `up` and `N` together without rechecking both ratios.
    """
    params = dict(Np=1e5, m=1.9e-25, omega=2 * math.pi * 40,
                  a0=2e-9, a02=0.0,
                  up=7.5e-6, N=32, sigma=2 * 1.486e-6,
                  ev_time=3e-4, time_steps=300*2,
                  include_g0=False, include_g2=False,
                  cutoff="No Cutoff", init_type="Gaussian",
                  imag_dtau=5e-3/2, imag_max_steps=20_000, tolerance=1e-12)
    params.update(overrides)
    return SimulationConfig(**params)


def build(cfg, kind="real", device="cpu"):
    grid   = Grid(cfg, device)
    cutoff = make_cutoff(cfg, grid)
    terms  = make_terms(cfg, grid, cutoff)
    rec    = Recorder([t.name for t in terms], track_modes=False)
    if kind == "real":
        ev = RealTimeEvolution(grid, make_potential(cfg, grid, cfg.gamma), terms,
                               dtau=cfg.dtau, max_steps=cfg.time_steps, recorder=rec)
    else:
        ev = ImaginaryTimeEvolution(grid, make_potential(cfg, grid, cfg.imag_gamma),
                                    terms, dtau=cfg.imag_dtau,
                                    max_steps=cfg.imag_max_steps,
                                    tolerance=cfg.tolerance, recorder=rec)
    return grid, ev, rec


# ---------------- 1. norm conservation ----------------

def test_real_time_conserves_norm():
    """Real-time evolution is unitary. Nothing else needs to be true for this."""
    cfg = small(include_g0=True, a02=5e-23, include_g2=True, cutoff="Hard Cutoff")
    grid, ev, _ = build(cfg, "real")
    psi = ev.run(grid.gaussian().to(grid.device))
    assert np.isclose(grid.unitless_norm(psi).item(), 1.0, rtol=1e-10)


# ---------------- 2. energy conservation ----------------

def test_static_trap_conserves_energy():
    """A time-independent Hamiltonian conserves its own expectation value.

    Strang splitting is second order, so drift scales as O(dtau^2).
    Measured: 2.02e-8 at dtau=2.51e-4, with ratios 3.99/4.00 under halving.
    The 1e-7 bound is ~5x headroom at the test's dtau; if you change
    time_steps or ev_time, rescale it as (dtau_new/2.51e-4)^2 * 2e-8.
    """
    cfg = small(include_g0=True, modulation_amp=0.0)
    _, ev, rec = build(cfg, "real")
    grid = ev.grid
    ev.run(grid.gaussian().to(grid.device))
    total = rec.energies()[:, -1]
    drift = abs(total[-1] - total[0]) / abs(total[0])
    assert drift < 1e-7, f"energy drifted by {drift:.2e}"

def test_static_trap_conserves_energy_with_g2():
    """Same as above but with the quadrupole term active - exercises the
    cutoff and the k-space multiplier, which nothing else checks."""
    cfg = small(include_g0=True, include_g2=True, a02=5e-23,
                cutoff="Hard Cutoff", modulation_amp=0.0)
    _, ev, rec = build(cfg, "real")
    grid = ev.grid
    ev.run(grid.gaussian().to(grid.device))
    total = rec.energies()[:, -1]
    drift = abs(total[-1] - total[0]) / abs(total[0])
    assert drift < 1e-6, f"energy drifted by {drift:.2e}"


# ---------------- 3. the analytic ground state ----------------

def test_noninteracting_ground_state_energy():
    """With no interactions, the 3-D isotropic HO ground state has E = 3/2 hw."""
    cfg = small(include_g0=False, include_g2=False, a02=0.0, gamma=1.0,
                imag_gamma=1.0)
    grid, ev, rec = build(cfg, "imag")
    ev.run(grid.gaussian().to(grid.device))
    assert ev.converged, "ITE never converged - raise imag_max_steps"
    print(rec.last_total())
    assert np.isclose(rec.last_total(), 1.5, rtol=1e-3)


def test_noninteracting_ground_state_shape():
    """...and its density is exp(-r^2) in oscillator units."""
    cfg = small(include_g0=False, include_g2=False, a02=0.0)
    grid, ev, _ = build(cfg, "imag")
    psi = ev.run(grid.gaussian().to(grid.device))

    r2 = (grid.ux3**2 + grid.uy3**2 + grid.uz3**2).expand(grid.shape)
    expected = torch.exp(-r2)
    expected = expected / (torch.sum(expected) * grid.dV)
    actual = torch.abs(psi)**2 / (torch.sum(torch.abs(psi)**2) * grid.dV)
    assert torch.max(torch.abs(actual - expected)).item() < 1e-4


# ---------------- 4. the virial relation (contact only) ----------------

def test_virial_relation():
    """For a stationary contact-interacting state: 2E_kin - 2E_trap + 3E_int = 0.

    Only holds with g2 switched off - the quadrupole term scales differently
    under the dilation this theorem is built on.
    """
    cfg = small(include_g0=True, include_g2=False, a02=0.0,
                up=2.0e-5, N=64, imag_dtau=6.25e-4)  
    grid, ev, rec = build(cfg, "imag")
    ev.run(grid.gaussian().to(grid.device))
    e = rec.energies()[-1]
    cols = rec.columns
    kin  = e[cols.index("kinetic")]
    trap = e[cols.index("potential")]
    inter = e[cols.index("g0 (contact)")]
    residual = 2 * kin - 2 * trap + 3 * inter
    assert abs(residual) / abs(trap) < 1e-2

def test_virial_relation_noninteracting():
    """2K - 2V = 0 exactly in the continuum. Tight tolerance: this checks the
    virial machinery itself, not the nonlinear scheme."""
    cfg = small(include_g0=False, include_g2=False, a02=0.0)
    grid, ev, rec = build(cfg, "imag")
    ev.run(grid.gaussian().to(grid.device))
    e = rec.energies()[-1]
    cols = rec.columns
    kin  = e[cols.index("kinetic")]
    trap = e[cols.index("potential")]
    assert abs(2*kin - 2*trap) / abs(trap) < 1e-5



def test_ramp_endpoints():
    """Every ramp profile is exactly 0 before start and reaches 1 by the end
    (exponential only asymptotically)."""
    for kind in RAMP_KINDS:
        if kind == "none":
            continue
        r = Ramp(kind, start=1.0, duration=2.0)
        assert r(0.5) == 0.0, kind
        if kind == "exponential":
            assert math.isclose(r(1.0 + 20 * 2.0), 1.0, rel_tol=1e-8)
        else:
            assert math.isclose(r(3.0), 1.0), kind


def test_ramped_term_scales_from_initial_to_one():
    term = RampedTerm(ContactTerm(1.0), Ramp("linear", start=1.0, duration=1.0),
                      initial=0.25)
    assert term.scale(0.0) == 0.25
    assert math.isclose(term.scale(1.5), 0.625)
    assert term.scale(3.0) == 1.0


def test_imag_stage_holds_g2_at_initial_value():
    base = small(include_g2=True, a02=5e-23, g2_ramp="linear", g2_ramp_time=1e-3)
    grid = Grid(base, "cpu")
    cutoff = make_cutoff(base, grid)
    names = lambda c, s: [t.name for t in make_terms(c, grid, cutoff, stage=s)]
    assert "g2 (quadrupole)" not in names(base, "imag")                 # starts at 0
    assert "g2 (quadrupole)" in names(replace(base, a02_initial=5e-28), "imag")
    assert "g2 (quadrupole)" in names(base, "real")


def test_ramped_g2_is_off_before_start():
    cfg = small(include_g0=True, include_g2=True, a02=5e-23, cutoff="Hard Cutoff",
                g2_ramp="smoother", g2_ramp_start=1.0, g2_ramp_time=1e-3)
    _, ev, rec = build(cfg, "real")
    ev.run(ev.grid.gaussian().to(ev.grid.device))
    col = rec.columns.index("g2 (quadrupole)")
    assert np.all(rec.energies()[:, col] == 0.0)


def test_ground_state_key_with_ramps():
    """Ramp shape and timing share a ground state; the starting a02 does not."""
    a = small(include_g2=True, a02=5e-23, g2_ramp="linear", g2_ramp_time=1e-3)
    b = replace(a, g2_ramp="smooth", g2_ramp_time=5e-3, g2_ramp_start=2e-3)
    c = replace(a, g2_ramp="none")
    d = replace(a, a02_initial=5e-28)
    assert a.ground_state_key() == b.ground_state_key()
    assert a.ground_state_key() != c.ground_state_key()
    assert a.ground_state_key() != d.ground_state_key()

def test_ramped_g2_acts_on_the_wavefunction():
    """After the ramp, g2 must change psi itself - not just the reported energy.
    Catches the propagation path calling field(psi) without the time.
    """
    cfg = small(include_g0=True, include_g2=True, a02=5e-23, cutoff="Hard Cutoff",
                g2_ramp="quench", g2_ramp_start=0.0)
    _, ev_on, _ = build(cfg, "real")
    psi_on = ev_on.run(ev_on.grid.gaussian().to(ev_on.grid.device))

    cfg_off = replace(cfg, include_g2=False, a02=0.0, g2_ramp="none")
    _, ev_off, _ = build(cfg_off, "real")
    psi_off = ev_off.run(ev_off.grid.gaussian().to(ev_off.grid.device))

    diff = (torch.abs(psi_on - psi_off).max()
            / torch.abs(psi_off).max()).item()
    assert diff > 1e-6, f"g2 changed psi by only {diff:.1e} - is it in the propagation?"

'''
cfg = small(include_g0=True, include_g2=False, a02=0.0, up=2.0e-5, N=64)
grid, ev, rec = build(cfg, "imag")
psi = ev.run(grid.gaussian().to(grid.device))

rho = torch.abs(psi)**2
V_trap = ev.potential(0.0)

K_direct = (torch.sum(0.5*grid.K2 * torch.abs(torch.fft.fftn(psi))**2)
            * grid.dV / grid.n_points).item()
V_direct = (torch.sum(V_trap * rho) * grid.dV).item()
I_direct = (0.5 * cfg.G0 * torch.sum(rho**2) * grid.dV).item()   # (G0/2)∫|psi|^4

print(f"norm     = {grid.unitless_norm(psi).item():.12f}")
print(f"K: direct={K_direct:.6f}  recorder={rec.energies()[-1][rec.columns.index('kinetic')]:.6f}")
print(f"V: direct={V_direct:.6f}  recorder={rec.energies()[-1][rec.columns.index('potential')]:.6f}")
print(f"I: direct={I_direct:.6f}  recorder={rec.energies()[-1][rec.columns.index('g0 (contact)')]:.6f}")
print(f"residual (direct) = {2*K_direct - 2*V_direct + 3*I_direct:+.6f}")

for dtau in (5e-3, 2.5e-3, 1.25e-3, 6.25e-4):
    cfg = small(include_g0=True, include_g2=False, a02=0.0, up=2.0e-5, N=64,
                imag_dtau=dtau, imag_max_steps=200_000)
    grid, ev, rec = build(cfg, "imag")
    ev.run(grid.gaussian().to(grid.device))
    e, c = rec.energies()[-1], rec.columns
    K, V, I = (e[c.index(n)] for n in ("kinetic", "potential", "g0 (contact)"))
    print(f"dtau={dtau:.2e}  K={K:.6f}  residual/V={abs(2*K-2*V+3*I)/V:.3e}")'''