"""Bounded single-band degree-2 SH residual, with no DC or color offset."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from utils.sh_utils import C1, C2


def sh2_basis(directions):
    """Canonical 3DGS real SH ordering, degrees 1 and 2 only (8 terms)."""
    if directions.shape[-1] != 3:
        raise ValueError("SH directions must have three components")
    d = F.normalize(directions, dim=-1, eps=1e-12)
    x, y, z = d.unbind(-1)
    return torch.stack((-C1 * y, C1 * z, -C1 * x,
        C2[0] * x * y, C2[1] * y * z,
        C2[2] * (2 * z.square() - x.square() - y.square()),
        C2[3] * x * z, C2[4] * (x.square() - y.square())), dim=-1)


class ThermalSHResidual(nn.Module):
    degree = 2
    coefficient_count = 8
    # Addition theorem: sum of squared degree-1/2 bases on the unit sphere.
    basis_energy = 8.0 / (4.0 * math.pi)

    def __init__(self, xyz, fit_support, signal_bound, trainable=False):
        super().__init__()
        support = torch.as_tensor(fit_support, device=xyz.device, dtype=torch.bool).reshape(-1)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(support) or not len(xyz):
            raise ValueError("SH support must match nonempty Gaussian xyz")
        if not math.isfinite(signal_bound) or signal_bound <= 0:
            raise ValueError("SH signal bound must be finite and positive")
        if trainable and not bool(support.any()):
            raise ValueError("No fit support for trainable SH residual")
        self.signal_bound = float(signal_bound)
        self.register_buffer("fit_support", support.clone())
        self.raw = nn.Parameter(xyz.new_zeros((len(xyz), 8)), requires_grad=trainable)

    @property
    def coefficients(self):
        # Smooth L2 ball, preserving a pure non-DC SH function. Unlike tanh of
        # the evaluated residual, this does not introduce a DC/high-order term.
        denominator = torch.sqrt(self.basis_energy * (1 + self.raw.square().sum(-1, keepdim=True)))
        return self.signal_bound * self.raw / denominator * self.fit_support[:, None].to(self.raw)

    def forward(self, xyz, camera_center, detach_geometry=True):
        if xyz.shape != (len(self.raw), 3):
            raise ValueError("SH residual and geometry have different Gaussian counts")
        # Match canonical 3DGS camera -> Gaussian direction; geometry is fixed.
        positions = xyz.detach() if detach_geometry else xyz
        directions = positions - camera_center.detach().reshape(1, 3)
        return (self.coefficients * sh2_basis(directions)).sum(-1, keepdim=True)

    def regularizer(self):
        if not bool(self.fit_support.any()):
            return self.raw.sum() * 0
        squared = self.raw[self.fit_support].square().sum(-1)
        return (squared / (1 + squared)).mean()

    @torch.no_grad()
    def diagnostics(self):
        values = self.raw[self.fit_support].square().sum(-1)
        usage = torch.sqrt(values / (1 + values))
        return {"sh_degree": 2, "sh_coefficients_per_gaussian": 8,
                "sh_signal_bound": self.signal_bound,
                "sh_fit_support_fraction": float(self.fit_support.float().mean()),
                "sh_budget_usage_mean": float(usage.mean()) if usage.numel() else 0.0,
                "sh_budget_usage_p95": float(torch.quantile(usage, 0.95)) if usage.numel() else 0.0,
                "sh_budget_saturated_fraction": float((usage >= 0.95).float().mean()) if usage.numel() else 0.0}

    def export_state(self):
        return {"format_version": 1, "kind": "single_band_sh2_no_dc",
                "degree": 2, "signal_bound": self.signal_bound,
                "state_dict": {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint, xyz, trainable=False):
        saved = checkpoint.get("sh_residual")
        if saved is None:
            return None
        if (saved.get("format_version") != 1 or saved.get("kind") != "single_band_sh2_no_dc"
                or saved.get("degree") != 2):
            raise ValueError("Unsupported thermal SH residual checkpoint")
        model = cls(xyz, saved["state_dict"]["fit_support"], saved["signal_bound"], trainable).to(xyz.device)
        model.load_state_dict({key: value.to(xyz.device) for key, value in saved["state_dict"].items()}, strict=True)
        if not bool(torch.isfinite(model.raw).all()):
            raise ValueError("Non-finite SH residual checkpoint")
        return model


def thermal_view_radiance(physical, xyz, camera_center, residual=None):
    """Return scalar signal, signed residual, and fraction clipped before blending."""
    if residual is None:
        return physical, torch.zeros_like(physical), physical.new_zeros(())
    delta = residual(xyz, camera_center)
    unbounded = physical + delta
    clipped_fraction = ((unbounded < 0) | (unbounded > 1)).to(physical).mean()
    return unbounded.clamp(0, 1), delta, clipped_fraction
