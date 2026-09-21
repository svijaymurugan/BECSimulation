import math
from scipy.fft import next_fast_len

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

    print(f"box:      up >= {up_min*1e6:.1f} um   (you have {cfg.up*1e6:.1f} um)"
          f"{'  <-- TOO SMALL' if cfg.up < up_min else ''}")
    print(f"momentum: focus needs {k_focus:.3e} m^-1, cutoff needs {k_cut:.3e} m^-1")
    print(f"grid:     dx <= {dx_max*1e6:.3f} um  ->  N >= {N_min}"
          f"   (you have {cfg.N})  ~{gb:.2f} GB per complex array")
    return N_min