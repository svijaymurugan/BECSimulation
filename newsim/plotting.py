"""Figures. Every function RETURNS a figure and saves nothing.

That one decision is what makes these usable both in a notebook (Jupyter
displays a returned figure) and in a batch script (call .savefig yourself).
Your original fused create-and-save-and-close, which is why interactive use
was impossible.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import numpy as np
from pathlib import Path
import torch


def density_map(field, xmesh, ymesh, *, title="", xlabel="", ylabel="",
                cbar_label="Density", ax=None, **pcolormesh_kw):
    """One 2-D density panel. This replaces six near-identical blocks."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.pcolormesh(xmesh, ymesh, field, **pcolormesh_kw)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    return fig


def energy_history(times, energies, columns, *, title="Energy contributions"):
    """The legend comes from the data. Add an interaction term and this
    updates itself - which is the whole point of finding 3.4."""
    fig, ax = plt.subplots()
    for k, name in enumerate(columns):
        ax.plot(times, energies[:, k], label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (unitless)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def mode_spectrogram(k_axis, times, amplitudes, *, title="", noise_floor=1e-10):
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots()
    vmax = max(float(np.max(amplitudes)), noise_floor * 10)
    im = ax.pcolormesh(k_axis, times, amplitudes, shading="auto", cmap="magma",
                       norm=LogNorm(vmin=noise_floor, vmax=vmax))
    ax.set_xlabel("Unitless momentum")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Amplitude")
    fig.tight_layout()
    return fig

def waist_history(times, waist, *, title="RMS waist"):
    """<r^2> in oscillator units against time."""
    fig, ax = plt.subplots()
    ax.plot(times, waist)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\langle r^2 \rangle / \ell^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig

# --- plotting.py ---
def standard_set(arrays, *, out_dir=None, gif_fps=2, scale="linear") -> dict:
    """Every routine figure for a run, keyed by name. Static figures are
    returned unsaved; GIFs must be written to disk, so they need out_dir.
    Takes only arrays - works identically on a live run or a loaded npz.
    """
    t = arrays["times"]
    figs = {"energies": energy_history(t, arrays["energies"], list(arrays["columns"]))}

    if "waist" in arrays:
        figs["waist"] = waist_history(t, arrays["waist"])

    if "kx_amplitudes" in arrays:
        figs["modes_kx"] = mode_spectrogram(arrays["kx_axis"], t,
                                            arrays["kx_amplitudes"], title="kx spectrum")
        figs["modes_kz"] = mode_spectrogram(arrays["kz_axis"], t,
                                            arrays["kz_amplitudes"], title="kz spectrum")

    gif_paths = dict()
    if "frames_xy" in arrays and out_dir is not None:
        gif_paths = dict()
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        X, Y = np.meshgrid(arrays["x_axis"], arrays["x_axis"], indexing="ij")
        X2, Z = np.meshgrid(arrays["x_axis"], arrays["z_axis"], indexing="ij")
        gif_paths["gif_xy"] = animate_density(arrays["frames_xy"], X, Y, t,
                                         path=out / "density_xy.gif",
                                         title="XY column density",
                                         x_label="x/l", y_label="y/l", fps=gif_fps, scale=scale)
        gif_paths["gif_xz"] = animate_density(arrays["frames_xz"], X2, Z, t,
                                         path=out / "density_xz.gif",
                                         title="XZ column density",
                                         x_label="x/l", y_label="z/l", fps=gif_fps, scale=scale)
    return figs, gif_paths


def save_set(figs: dict, directory, *, dpi=300, close=True):
    """Write every Figure in the dict. Non-Figure entries (GIF paths) are
    already on disk, so skip them."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, value in figs.items():
        if not isinstance(value, Figure):
            written.append(value)       # already a path; nothing to do
            continue
        p = directory / f"{name}.png"
        value.savefig(p, dpi=dpi)
        if close:
            plt.close(value)
        written.append(p)
    return written


def animate_density(frames, xmesh, ymesh, times, *, path, title="",
                    x_label="x/l", y_label="y/l", fps=20, cmap="magma",
                    scale="linear", decades=3, clip_percentile=None,
                    per_frame=False):
    """Write a GIF from recorded density frames.

    scale: "linear" | "log" | "power"
    decades: for log, how many decades below vmax to show (3 is usually right)
    clip_percentile: e.g. 99.5, to stop a thin peak eating the dynamic range
    per_frame: rescale each frame independently. Makes every frame well-exposed
        but DESTROYS comparability between frames - an expanding, thinning cloud
        will look like it is not changing. The title is marked when this is on.
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LogNorm, Normalize, PowerNorm

    def limits(data):
        vmax = (float(np.percentile(data, clip_percentile)) if clip_percentile
                else float(np.max(data)))
        vmax = max(vmax, 1e-300)
        if scale == "log":
            return LogNorm(vmin=vmax * 10.0**(-decades), vmax=vmax)
        if scale == "power":
            return PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)
        return Normalize(vmin=0.0, vmax=vmax)

    norm = limits(frames)          # from the whole stack, unless per_frame

    fig, ax = plt.subplots()
    im = ax.pcolormesh(xmesh, ymesh, frames[0], norm=norm, cmap=cmap,
                       shading="auto")
    ax.set_aspect("equal", "box")
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    title_artist = ax.set_title("")
    fig.colorbar(im, ax=ax, label="Column density")
    if title:
        fig.suptitle(title + ("  [per-frame scale]" if per_frame else ""))

    def update(k):
        im.set_array(frames[k].ravel())
        if per_frame:
            im.set_norm(limits(frames[k]))
        title_artist.set_text(f"t = {times[k]*1e3:.2f} ms")
        return im, title_artist

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    return path