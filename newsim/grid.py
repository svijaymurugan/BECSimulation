"""All geometry. Nothing else in the package builds a meshgrid or an FFT frequency."""
from __future__ import annotations

import math
import torch


class Grid:
    """Real- and momentum-space grids for one simulation.

    The 3-D coordinate arrays are kept as *broadcast views* of 1-D axes rather
    than materialised meshgrids. `ux3` has shape (N, 1, 1), `uy3` is (1, N, 1),
    `uz3` is (1, 1, nz); any arithmetic among them broadcasts to the full
    (N, N, nz) block with exactly the same values a meshgrid would give -- but
    without storing three N^3 arrays you never actually need.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.real_dtype    = torch.float64   if cfg.high_precision else torch.float32
        self.complex_dtype = torch.complex128 if cfg.high_precision else torch.complex64

        N, nz, up, l = cfg.N, cfg.nz, cfg.up, cfg.l
        self.shape = (N, N, nz)
        self.n_points = N * N * nz           # the honest count (see finding 04)

        # --- real space, in metres (kept on the CPU; only used to build psi_0) ---
        self.x = torch.linspace(-up, up, N  + 1, dtype=self.real_dtype)[:-1]
        self.z = torch.linspace(-up, up, nz + 1, dtype=self.real_dtype)[:-1]
        self.dx = (self.x[1] - self.x[0]).item()
        self.dz = (self.z[1] - self.z[0]).item()

        # --- real space, unitless, on the device, as broadcast views ---
        self.dxu = self.dx / l
        self.dzu = self.dz / l
        self.ux = (self.x / l).to(self.device)
        self.uz = (self.z / l).to(self.device)
        self.ux3 = self.ux.view(-1, 1, 1)
        self.uy3 = self.ux.view(1, -1, 1)
        self.uz3 = self.uz.view(1, 1, -1)

        # --- momentum space, unitless ---
        kx = 2 * math.pi * torch.fft.fftfreq(N,  d=self.dxu, dtype=self.real_dtype)
        kz = 2 * math.pi * torch.fft.fftfreq(nz, d=self.dzu, dtype=self.real_dtype)
        self.kx3 = kx.to(self.device).view(-1, 1, 1)
        self.ky3 = kx.to(self.device).view(1, -1, 1)
        self.kz3 = kz.to(self.device).view(1, 1, -1)
        self.K2  = self.kx3**2 + self.ky3**2 + self.kz3**2      # one N^3 array
        self.kmax = torch.pi / self.dx

        # --- volume elements ---
        self.dV    = self.dxu**2 * self.dzu    # unitless
        self.dV_si = self.dx**2  * self.dz     # metres^3

    def healing_length(self, psi, reduce="peak"):
        """Healing length in oscillator units.

        xi/l = 1/sqrt(2 G0 rho), with rho = |psi|^2 normalised to 1. The particle
        number enters through G0 = 4 pi hbar^2 a0 Np / (m l^3 hbar omega), so it
        does NOT appear separately here - putting Np in again would double-count.

        Resolution rule of thumb: dx/xi < 1, or the condensate edge and any
        vortex cores are unresolved.

        Legacy calculation:
        self.n0 = torch.max(torch.abs(self.init_wf*self.l**(-3/2))**2).item() * self.Np
        self.healing_length = 1/np.sqrt(8*pi*self.n0*self.a0)
        self.N_0 = self.n0 * self.l**3
        """
        rho = torch.abs(psi)**2
        n = (torch.max(rho) if reduce == "peak"
            else torch.sum(rho**2) * self.dV)
        if self.cfg.G0 == 0:
            return float("inf")          # non-interacting: no interaction scale at all
        if self.cfg.G0 < 0:
            return float("nan")          # attractive: healing length is not defined      # no interactions, no healing length
        return (1.0 / torch.sqrt(2 * self.cfg.G0 * n)).item()
        

    # ---------- measures ----------
    def norm(self, psi):
        """Norm with the SI volume element."""
        return torch.sum(torch.abs(psi)**2) * self.dV_si

    def unitless_norm(self, psi):
        return torch.sum(torch.abs(psi)**2) * self.dV

    def radius_squared(self):
        """r^2 in oscillator units, for the waist diagnostic."""
        return self.ux3**2 + self.uy3**2 + self.uz3**2

    # ---------- initial states ----------
    def gaussian(self):
        """The Gaussian ansatz: normalised, then nondimensionalised.

        Built on the CPU and returned on the CPU, exactly as the original did,
        so that torch.exp is evaluated by the same implementation. Move it to
        the device at the call site.
        """
        cfg, l = self.cfg, self.cfg.l
        x = self.x.view(-1, 1, 1)
        y = self.x.view(1, -1, 1)
        z = self.z.view(1, 1, -1)
        g = torch.exp(-1 * (x**2 + (cfg.y_scale * y)**2 + (cfg.z_scale * z)**2)
                      / (2 * cfg.sigma**2)).to(self.complex_dtype)
        g = g * (1.0 / self.norm(g).item())**0.5
        return (g * l**(3/2)).to(self.complex_dtype)

    def second_moments(self, psi):
        """(<x^2>, <y^2>, <z^2>) in oscillator units, via 1-D marginals of |psi|^2."""
        rho = torch.abs(psi)**2
        x2 = self.ux3.flatten()**2
        z2 = self.uz3.flatten()**2
        return torch.stack([torch.sum(rho.sum(dim=(1, 2)) * x2),
                            torch.sum(rho.sum(dim=(0, 2)) * x2),
                            torch.sum(rho.sum(dim=(0, 1)) * z2)]) * self.dV

    def mean_r2(self, psi):
        return self.second_moments(psi).sum()