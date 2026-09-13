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
        ux = (self.x / l).to(self.device)
        uz = (self.z / l).to(self.device)
        self.ux3 = ux.view(-1, 1, 1)
        self.uy3 = ux.view(1, -1, 1)
        self.uz3 = uz.view(1, 1, -1)

        # --- momentum space, unitless ---
        kx = 2 * math.pi * torch.fft.fftfreq(N,  d=self.dxu, dtype=self.real_dtype)
        kz = 2 * math.pi * torch.fft.fftfreq(nz, d=self.dzu, dtype=self.real_dtype)
        self.kx3 = kx.to(self.device).view(-1, 1, 1)
        self.ky3 = kx.to(self.device).view(1, -1, 1)
        self.kz3 = kz.to(self.device).view(1, 1, -1)
        self.K2  = self.kx3**2 + self.ky3**2 + self.kz3**2      # one N^3 array

        # --- volume elements ---
        self.dV    = self.dxu**2 * self.dzu    # unitless
        self.dV_si = self.dx**2  * self.dz     # metres^3

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