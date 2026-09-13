
Split-step GPE simulation for a trapped BEC with contact (g0) and quadrupole (g2)
interactions. Real-time evolution, imaginary-time ground states, and a diagnostic
recorder that keeps everything on the GPU until the run ends.

## Layout

| Module | Owns | Never does |
|---|---|---|
| `config.py` | every user-supplied value and everything derivable from it | touch a tensor |
| `grid.py` | axes, k-space, volume elements, norms | know about physics |
| `cutoff.py` | momentum-space filters for the g2 term | know about the loop |
| `interactions.py` | one nonlinear term per class | know about the loop |
| `potentials.py` | trap shapes and modulation | know about the loop |
| `evolution.py` | the time loop, and nothing else | plot, save, build operators |
| `diagnostics.py` | measurement, kept off the CPU | know about files |
| `storage.py` | bytes on disk | know about figures |
| `plotting.py` | arrays to figures | know where they came from |

Dependencies point one way: config → grid → operators → evolution. Nothing points back.

## Quick start

Five cells. Only the first one changes between runs.

### Cell 1 — the knobs

Everything you tweak lives here. Nothing below this cell needs editing.

```python
import torch
from config import SimulationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = SimulationConfig(
    # --- required: no defaults, you must state these ---
    Np=1e5,                       # particle number
    m=1.9e-25,                    # atomic mass (kg)
    omega=2 * torch.pi * 40,      # trap frequency (rad/s)
    a0=2e-9,                      # s-wave scattering length (m)
    a02=5e-23,                    # scattering volume, g2 term (m^3)
    up=2.5e-5,                    # half box length (m)
    N=256,                        # grid points in x and y

    # --- geometry ---
    Nz=None,                      # None means same as N
    sigma=10e-6,                  # width of the Gaussian ansatz (m)

    # --- trap ---
    gamma=1.0,                    # z anisotropy, real time
    imag_gamma=1.0,               # z anisotropy, imaginary time
    modulation_amp=0.1,
    modulation_freq=2 * 5**0.5,

    # --- real time ---
    ev_time=0.015,
    time_steps=10_000,

    # --- imaginary time ---
    imag_dtau=1e-4,
    imag_max_steps=10_000,
    tolerance=1e-6,

    # --- numerics ---
    cutoff="Hard Cutoff",         # see the registry in cutoff.py
    cutoff_coeff=0.9,
    include_g0=True,
    include_g2=True,
    high_precision=True,

    # --- what to record (all opt-in; see "Diagnostics" below) ---
    track_waist=True,
    track_modes=False,
)

print(cfg)                        # dataclass prints every field, labelled
print(cfg.to_dict()["_derived"])  # l, dt, G0, G2, ...
```

The config is **frozen**: nothing can change it after construction, so the config
you print is provably the config that ran. To vary one parameter, make a copy:

```python
from dataclasses import replace
cfg2 = replace(cfg, gamma=1.3)         # validation re-runs; derived values follow
scan = [replace(cfg, omega=w) for w in omegas]
```

Validation runs at construction and reports **every** problem at once, not just
the first.

### Cell 2 — build the machinery

This cell is the same for every run. Read it once, then leave it alone.

```python
from grid import Grid
from cutoff import make_cutoff
from interactions import make_terms
from potentials import make_potential
from evolution import RealTimeEvolution, ImaginaryTimeEvolution
from diagnostics import Recorder, NormMonitor
import storage, plotting

grid   = Grid(cfg, device)
cutoff = make_cutoff(cfg, grid)
terms  = make_terms(cfg, grid, cutoff)
names  = [t.name for t in terms]
```

Four objects, each fully constructed before use. Nothing reaches into anything
else afterwards — if you ever find yourself writing `something.attribute = x`
from outside the class that defines it, that value belongs in a constructor.

### Cell 3 — the ground state (cached)

```python
path = storage.ground_state_path(cfg)

if path.exists():
    psi, meta = storage.load_ground_state(path, cfg, device)
    print(f"Loaded ground state, E = {meta['final_energy']:.6f}, "
          f"{meta['steps']} steps, created {meta['created']}")
else:
    rec_ite = Recorder.energies_only(names, dt=cfg.imag_dtau)
    ite = ImaginaryTimeEvolution(
        grid, make_potential(cfg, grid, cfg.imag_gamma), terms,
        dtau=cfg.imag_dtau, max_steps=cfg.imag_max_steps,
        recorder=rec_ite, tolerance=cfg.tolerance)
    psi = ite.run(grid.gaussian().to(device))
    storage.save_ground_state(path, psi, cfg,
                              converged=ite.converged,
                              final_energy=rec_ite.last_total(),
                              steps=ite.steps_taken)
    print(f"Computed and cached: {path}")
```

The filename is a 16-character hash of every parameter the ground state depends
on, so changing `a02` gives a clean cache miss rather than silently wrong physics.
The same key is stored *inside* the file and verified on load — a renamed or
mis-copied file raises rather than lying to you. A state that never converged is
saved with `converged=False` and refuses to load.

Note `Recorder.energies_only` — imaginary time needs the total energy for its
convergence check and nothing else.

### Cell 4 — the real-time run

```python
rec = Recorder(names,
               measure_every=10,          # energies every 10th step
               dt=cfg.dt,
               track_modes=cfg.track_modes,
               track_frames=True,         # for the GIF
               radius_squared=grid.radius_squared() if cfg.track_waist else None,
               dV=grid.dV)

rte = RealTimeEvolution(
    grid, make_potential(cfg, grid, cfg.gamma), terms,
    dtau=cfg.dtau, max_steps=cfg.time_steps,
    recorder=rec, measure_every=10,
    monitors=[NormMonitor(grid)])

psi_final = rte.run(psi)

out = "storage/run_2026_09_13"
storage.save_run(out, cfg, grid, rec,
                 description="modulation at 2*sqrt(5), watching the k=2 mode",
                 psi_final=psi_final.cpu().numpy())
```

`measure_every` must be passed to **both** the Recorder (so `times()` is right)
and the Evolution (so it knows when to measure). Keep them equal.

### Cell 5 — plots

```python
cfg_dict, arrays = storage.load_run(f"{out}/run.npz")

figs, gifs = plotting.standard_set(arrays, out_dir=f"{out}/plots")
figs["energies"]                                  # renders inline
figs["energies"].axes[0].set_yscale("log")        # adjust after the fact
plotting.save_set(figs, f"{out}/plots")
```

Every plotting function **returns** a figure and saves nothing, which is what
makes them work both inline and in a batch script. Re-plotting a months-old run
needs no GPU and no config — the npz carries its own time axis, k-axes and
spatial axes.

## Diagnostics

Energies are always recorded. Everything else is opt-in, because everything else
is expensive.

| Diagnostic | Enabled by | Cost at N=256, 10k steps |
|---|---|---|
| energies | always | 0.56 MB |
| waist | `radius_squared=` not None | 0.08 MB |
| mode amplitudes | `track_modes=True` | 41 MB |
| density frames | `track_frames=True` | 0.26 MB each |
| full 3-D psi | pass to `save_run` | 268 MB each |

`measure_every=10` cuts all per-step costs tenfold and still gives 1,000 samples
across a run. The measurement takes its own FFT (it cannot share the propagation
one — see the note in `evolution.py`), so sampling is also where the speed comes
from.

**Never enable frames for imaginary time.** A 20,000-step ITE at N=64 would
accumulate ~650 MB of GPU memory for a movie of a ground-state search.

## Monitors

Monitors sample during a run and **warn** rather than raise — you don't throw
away forty minutes of GPU time because a diagnostic wobbled.

```python
monitors=[NormMonitor(grid, every=100, tol=1e-8),
          EnergyMonitor(every=500, rel_tol=1e-4)]   # static traps only
```

## Tests

```bash
pip install pytest
pytest test_physics.py -v
```

Seven physics facts, all on a tiny CPU grid, all in under a minute. Keep them
that fast — a suite that takes five minutes is a suite you stop running.

| Test | The fact |
|---|---|
| norm conservation | real-time evolution is unitary |
| energy conservation | static trap ⇒ total energy constant, O(dtau²) |
| same, with g2 | exercises the cutoff and k-space multiplier |
| ground-state energy | with no interactions, E = 3/2 |
| ground-state shape | density is exp(−r²) in oscillator units |
| virial, non-interacting | 2K − 2V = 0, tight tolerance |
| virial, interacting | 2K − 2V + 3I = 0, tolerance set by imaginary-time dtau |

## Extending it

**A new cutoff:** write a class with an `apply(field_k)` method in `cutoff.py`,
add one line to `CUTOFFS`. Validation picks it up automatically — the registry
*is* the list of valid names.

**A new interaction:** write a class with a `name` and a `field(psi)` in
`interactions.py`, add one line to `make_terms`. Energy legends, recorder
columns, saved arrays and plot labels all follow, because every one of them
reads `term.name`.

**A new trap:** subclass `Potential` with a `__call__(t)` and an `is_static`
property. A static trap gets its exponential built once and reused for the whole
run.

If adding any of these requires editing anything beyond the new class and one
registry line, the abstraction isn't finished — that's the test.

## Colab

```python
!git clone -q https://github.com/<you>/BECSimulation.git
%cd BECSimulation
%load_ext autoreload
%autoreload 2
```

`autoreload 2` re-reads changed `.py` files before each cell, so you edit in an
editor and re-run the cell without restarting the kernel. After changing a
constructor, rebuild your objects.

Colab's disk is wiped on disconnect. Mount Drive and point outputs there:

```python
from google.colab import drive; drive.mount('/content/drive')
OUT = "/content/drive/MyDrive/BECSimulation/storage"
storage.save_run(f"{OUT}/run_x", cfg, grid, rec, ...)
storage.ground_state_path(cfg, root=OUT)
```

## Known limitations

Imaginary-time evolution is **first order** in `imag_dtau` for the nonlinear
term — the interaction field is frozen at the start of each half-kick, and in
imaginary time the kick is a real decay, so `|psi|²` changes during the step.
Real time is unaffected (the kick is a pure phase). See the note at the bottom
of `evolution.py` for the measured scaling and the predictor-corrector remedy.