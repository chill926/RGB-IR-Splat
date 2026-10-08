"""Bounded, view-independent IR opacity; RGB geometry/opacity stay immutable."""
import math

import torch
from torch import nn


class IROpacityCorrection(nn.Module):
    def __init__(self, base_opacity, fit_support, logit_bound=0.2, trainable=False):
        super().__init__()
        base = base_opacity.detach().reshape(-1, 1).clone()
        support = torch.as_tensor(fit_support, device=base.device, dtype=torch.bool).reshape(-1)
        if base.shape[0] != support.numel() or base.numel() == 0:
            raise ValueError("IR opacity support must match the Gaussian count")
        if not bool(torch.isfinite(base).all()) or bool(((base < 0) | (base > 1)).any()):
            raise ValueError("RGB opacity must be finite in [0,1]")
        if not math.isfinite(logit_bound) or not 0 < logit_bound <= 2:
            raise ValueError("IR logit bound must be finite in (0,2]")
        if trainable and not bool(support.any()):
            raise ValueError("No fit support for trainable IR opacity")
        self.logit_bound = float(logit_bound)
        self.register_buffer("base_opacity", base)
        self.register_buffer("fit_support", support)
        self.raw = nn.Parameter(torch.zeros_like(base), requires_grad=trainable)

    @property
    def logit_delta(self):
        return self.logit_bound * torch.tanh(self.raw) * self.fit_support[:, None].to(self.raw)

    @property
    def opacity(self):
        base = self.base_opacity.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(base) - torch.log1p(-base)
        shifted = torch.sigmoid(logits + self.logit_delta)
        # Subtract the zero-shift evaluation to preserve exact initialization,
        # even near sigmoid saturation. The clamp affects only roundoff/extrema.
        corrected = (self.base_opacity + (shifted - torch.sigmoid(logits))).clamp(0, 1)
        return torch.where(self.fit_support[:, None], corrected, self.base_opacity)

    def regularizer(self):
        if not bool(self.fit_support.any()):
            return self.raw.sum() * 0
        # Penalize a fraction of the allowed logit budget, not a scene-dependent unit.
        return torch.tanh(self.raw[self.fit_support]).square().mean()

    @torch.no_grad()
    def diagnostics(self):
        absolute = (self.opacity - self.base_opacity).abs().reshape(-1)
        visible = absolute[self.fit_support]
        return {"ir_opacity_fit_support_fraction": float(self.fit_support.float().mean()),
                "ir_opacity_abs_delta_mean": float(absolute.mean()),
                "ir_opacity_abs_delta_max": float(absolute.max()),
                "ir_opacity_fit_abs_delta_p95": float(torch.quantile(visible, 0.95)) if visible.numel() else 0.0,
                "ir_opacity_saturated_fraction": float((torch.tanh(self.raw[self.fit_support]).abs() >= 0.95).float().mean()) if visible.numel() else 0.0}

    def export_state(self):
        return {"format_version": 1, "kind": "bounded_ir_opacity", "logit_bound": self.logit_bound,
                "state_dict": {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint, base_opacity, trainable=False):
        state = checkpoint.get("ir_opacity")
        if state is None:
            return None
        if state.get("format_version") != 1 or state.get("kind") != "bounded_ir_opacity":
            raise ValueError("Unsupported IR opacity checkpoint")
        tensors = state["state_dict"]
        stored_base = tensors["base_opacity"].to(base_opacity)
        if stored_base.shape != base_opacity.shape or not torch.equal(stored_base, base_opacity.detach()):
            raise ValueError("IR opacity reference differs from loaded RGB opacity")
        model = cls(base_opacity, tensors["fit_support"], state["logit_bound"], trainable=trainable).to(base_opacity.device)
        model.load_state_dict({key: value.to(base_opacity.device) for key, value in tensors.items()}, strict=True)
        if not bool(torch.isfinite(model.raw).all()):
            raise ValueError("Non-finite IR opacity checkpoint")
        return model


def render_opacity(opacity_field):
    return None if opacity_field is None else opacity_field.opacity
