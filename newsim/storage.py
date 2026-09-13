"""Bytes on disk. Every file describes itself."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1


# ---------------- ground-state cache ----------------

def ground_state_path(cfg, root="storage") -> Path:
    """Short, unique, and meaningless to a human - which is fine, because the
    file carries its own description inside."""
    return Path(root) / f"ground_state_{cfg.ground_state_key()}.pt"


def save_ground_state(path, psi, cfg, *, converged, final_energy, steps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": FORMAT_VERSION,
        "psi":            psi.cpu(),
        "config":         cfg.to_dict(),
        "config_key":     cfg.ground_state_key(),
        "converged":      bool(converged),
        "final_energy":   float(final_energy),
        "steps":          int(steps),
        "created":        datetime.now().isoformat(timespec="seconds"),
    }, path)


def load_ground_state(path, cfg, device, *, require_converged=True):
    """Load and VERIFY. A filename is a hint; the payload is the truth."""
    payload = torch.load(path, map_location=device, weights_only=False)

    if payload.get("config_key") != cfg.ground_state_key():
        raise ValueError(
            f"{path} was computed for a different configuration.\n"
            f"  file:   {payload.get('config_key')}\n"
            f"  wanted: {cfg.ground_state_key()}\n"
            f"Delete it or change your parameters back.")

    if require_converged and not payload.get("converged", False):
        raise ValueError(
            f"{path} holds a ground state that never converged "
            f"({payload.get('steps')} steps, E = {payload.get('final_energy')}). "
            f"Delete it and rerun with a larger imag_max_steps.")

    return payload["psi"].to(device), payload


# ---------------- run output ----------------

def save_run(directory, cfg, grid, recorder, *, description="",psi_initial=None, psi_final=None,
            extras=None):
    """Write the numbers. Plots are derived from these later, not instead of them.

    Sizes at N=256, 10,000 steps, for calibration:
      energies + waist + modes   ~42 MB   <- always worth it
      one 2-D density frame f32  0.26 MB  <- cheaper than its own PNG
      one full 3-D psi c128       268 MB  <- checkpoints only
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    arrays = {
        "config_json": np.array(json.dumps(cfg.to_dict(), default=str)),
        "description": np.array(description),
        "columns":     np.array(recorder.columns),
        "times":    recorder.times().astype(np.float32),
        "energies":    recorder.energies().astype(np.float32),
        "x_axis":  (grid.x / cfg.l).numpy().astype(np.float32),
        "z_axis":  (grid.z / cfg.l).numpy().astype(np.float32),
        "kx_axis": torch.fft.fftshift(grid.kx3.flatten()).cpu().numpy().astype(np.float32),
        "kz_axis": torch.fft.fftshift(grid.kz3.flatten()).cpu().numpy().astype(np.float32),
    }

    fxy, fxz = recorder.frames()
    if fxy is not None:
        arrays["frames_xy"] = fxy.astype(np.float32)
        arrays["frames_xz"] = fxz.astype(np.float32)

    kx, kz = recorder.modes()
    if kx is not None:
        arrays["kx_amplitudes"] = kx.astype(np.float32)
        arrays["kz_amplitudes"] = kz.astype(np.float32)
        # in save_run
        arrays["kx_axis"] = torch.fft.fftshift(grid.kx3.flatten()).cpu().numpy()
        arrays["kz_axis"] = torch.fft.fftshift(grid.kz3.flatten()).cpu().numpy()

    waist = recorder.waist()
    if waist is not None:
        arrays["waist"] = waist.astype(np.float32)

    if psi_initial is not None:
        arrays["psi_initial"] = np.asarray(psi_initial)
    if psi_final is not None:
        arrays["psi_final"] = np.asarray(psi_final)

    frames = recorder.frames()
    if frames is not None:
        arrays["frames"] = np.array(frames, dtype=np.float32)
    if extras:
        arrays.update(extras)

    path = directory / "run.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_run(path):
    """Returns (config_dict, arrays_dict). Re-plot anything, any time, no GPU."""
    data = np.load(path, allow_pickle=False)
    cfg = json.loads(str(data["config_json"]))
    arrays = {k: data[k] for k in data.files if k != "config_json"}
    return cfg, arrays