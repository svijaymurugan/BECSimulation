import math
from scipy.fft import next_fast_len
import numpy as np

def grid_requirements(cfg, *, extent=None, box_sigmas=3.0,
                      k_margin=1.5, tail_sigmas=4.5):
    """Estimate the box and N a run needs. An estimate - confirm by convergence.

    extent: the largest cloud width (m) over the run; defaults to cfg.sigma.
    """
    l = cfg.l
    extent = extent or cfg.sigma

    # 1. box
    up_min = box_sigmas * extent

    # 2. momentum: the larger of the cutoff (with headroom) and the focus
    E0 = 0.75 * (cfg.sigma / l)**2                 # initial <V>, hbar*omega
    k_rms_axis = math.sqrt(2 * E0 / 3) / l         # m^-1, if all of E0 -> kinetic
    k_focus = tail_sigmas * k_rms_axis
    k_cut = k_margin * cfg.cutoff_kc if cfg.cutoff_kc else 0.0
    k_need = max(k_focus, k_cut)

    # 3. N
    up = max(cfg.up, up_min)
    dx_max = math.pi / k_need
    N_min = next_fast_len(math.ceil(2 * up / dx_max))
    gb = N_min**3 * 16 / 1e9

    too_small = cfg.up < up_min * (1 - 1e-9)

    print(f"box:      up >= {up_min*1e6:.1f} um   (you have {cfg.up*1e6:.1f} um)"
          f"{'  <-- TOO SMALL' if too_small else ''}")
    print(f"momentum: focus needs {k_focus:.3e} m^-1, cutoff needs {k_cut:.3e} m^-1")
    print(f"grid:     dx <= {dx_max*1e6:.3f} um  ->  N >= {N_min}"
          f"   (you have {cfg.N})  ~{gb:.2f} GB per complex array")
    return N_min

def ramp_residual(kind, duration, mode_freq, *, n=20001):
    """Predicted leftover oscillation of a mode, as a fraction of a quench's.

    Linear response: switching a coupling on along s(t) leaves a mode of
    angular frequency Omega oscillating with amplitude |∫ s'(t) e^{i Omega t} dt|,
    which is 1 for an instantaneous switch and falls toward 0 for slow ramps.

    duration in seconds; mode_freq in rad/s, e.g. sqrt(2) * cfg.omega for the
    quadrupole mode of an isotropic Thomas-Fermi condensate.
    """
    from ramps import Ramp
    if kind == "quench":
        return 1.0
    trap = getattr(np, "trapezoid", None) or np.trapz     # numpy 2 renamed it
    tmax = 30 * duration if kind == "exponential" else duration
    tt = np.linspace(0.0, tmax, n)
    r = Ramp(kind, start=0.0, duration=duration)
    s = np.array([r(x) for x in tt])
    return abs(trap(np.gradient(s, tt) * np.exp(1j * mode_freq * tt), tt))