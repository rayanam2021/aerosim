"""
Generate a placeholder ``mlp_model.pt`` for aerodynamics_sm_fmu.

The user's contract is that Luminary's ``aero_sm`` model is available locally as
a TorchScript file that maps [mach, alpha_deg, elevator_deg] -> [force_x_n,
force_z_n, moment_y_nm]. Until the real weights are dropped in, this script
trains a small MLP to reproduce the analytic aero function used as the FMU
fallback, then exports it via ``torch.jit.script`` so the FMU can load it with
``torch.jit.load`` exactly like the production model.

Usage:
    python create_placeholder_mlp_model.py [--out mlp_model.pt]
"""

from __future__ import annotations

import argparse
import os

# Avoid an OpenMP double-initialization crash when torch and numpy (MKL) load
# different OpenMP runtimes in one process (common on Windows).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

# Keep these consistent with aerodynamics_sm_fmu defaults / analytic_aero().
AIR_DENSITY = 0.7
REF_AREA = 0.03
SPEED_OF_SOUND = 340.0

MACH_RANGE = (0.3, 3.0)
ALPHA_RANGE = (-10.0, 10.0)
ELEVATOR_RANGE = (-20.0, 20.0)


def analytic_aero(mach, alpha_deg, elevator_deg):
    v = np.maximum(mach * SPEED_OF_SOUND, 1.0)
    qbar_s = 0.5 * AIR_DENSITY * v * v * REF_AREA
    cd = 0.30 + 0.015 * alpha_deg * alpha_deg
    cz = -(0.11 * alpha_deg + 0.045 * elevator_deg)
    cm = -(0.020 * alpha_deg + 0.060 * elevator_deg)
    force_x = -cd * qbar_s
    force_z = cz * qbar_s
    moment_y = cm * qbar_s
    return np.stack([force_x, force_z, moment_y], axis=-1)


class AeroMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )
        # Input/output normalization buffers (baked into the scripted model).
        self.register_buffer("in_mean", torch.zeros(3))
        self.register_buffer("in_std", torch.ones(3))
        self.register_buffer("out_mean", torch.zeros(3))
        self.register_buffer("out_std", torch.ones(3))

    def forward(self, x):
        z = (x - self.in_mean) / self.in_std
        y = self.net(z)
        return y * self.out_std + self.out_mean


def sample_dataset(n: int, rng: np.random.Generator):
    mach = rng.uniform(*MACH_RANGE, size=n)
    alpha = rng.uniform(*ALPHA_RANGE, size=n)
    elevator = rng.uniform(*ELEVATOR_RANGE, size=n)
    features = np.stack([mach, alpha, elevator], axis=-1)
    targets = analytic_aero(mach, alpha, elevator)
    return features.astype(np.float32), targets.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlp_model.pt")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--samples", type=int, default=8192)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    x, y = sample_dataset(args.samples, rng)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    model = AeroMLP()
    model.in_mean.copy_(x_t.mean(0))
    model.in_std.copy_(x_t.std(0).clamp_min(1e-6))
    model.out_mean.copy_(y_t.mean(0))
    model.out_std.copy_(y_t.std(0).clamp_min(1e-6))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1:4d}  mse={loss.item():.3e}")

    model.eval()
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, args.out)
    print(f"Saved placeholder surrogate to {args.out}")

    # Quick sanity check against the analytic function.
    with torch.no_grad():
        test = torch.tensor([[2.0, 3.0, -5.0]], dtype=torch.float32)
        print("model(mach=2, alpha=3, elev=-5) =", scripted(test).squeeze().tolist())
        print("analytic                        =", analytic_aero(2.0, 3.0, -5.0).tolist())


if __name__ == "__main__":
    main()
