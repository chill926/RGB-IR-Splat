"""Differentiable frozen display and fixed-count joint experiment utilities."""
import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F

from utils.thermal_training_utils import masked_mean


GEOMETRY_NAMES = ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest")


def geometry_digest(state):
    digest = hashlib.sha256()
    for name in GEOMETRY_NAMES:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    digest.update(str(int(state["active_sh_degree"])).encode("ascii"))
    return digest.hexdigest()


def install_joint_geometry(gaussians, checkpoint):
    """Restore the trained geometry, never silently evaluate its RGB initializer."""
    state = checkpoint.get("joint_geometry")
    protocol = checkpoint.get("metadata", {}).get("joint_protocol")
    if state is None or not protocol or protocol.get("kind") != "fixed_count_rgb_ir_D_v1":
        raise ValueError("Missing joint geometry/protocol")
    if geometry_digest(state) != protocol.get("trained_geometry_sha256"):
        raise ValueError("Joint geometry checksum mismatch")
    count = len(gaussians.get_xyz)
    if count != protocol.get("gaussian_count"):
        raise ValueError("Joint Gaussian count differs from its Stage1 initializer")
    for name in GEOMETRY_NAMES:
        value, original = state[name], getattr(gaussians, name)
        if value.shape != original.shape or value.dtype != original.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError("Invalid joint geometry tensor: " + name)
    degree = int(state["active_sh_degree"])
    if not 0 <= degree <= gaussians.max_sh_degree:
        raise ValueError("Invalid joint RGB SH degree")
    for name in GEOMETRY_NAMES:
        setattr(gaussians, name, nn.Parameter(state[name].to(gaussians.get_xyz.device).clone(), requires_grad=False))
    gaussians.active_sh_degree = degree


def linear_interp(query, knots, values):
    """Piecewise linear interpolation with linear extrapolation; gradients in query."""
    index = torch.searchsorted(knots, query.contiguous()).clamp(1, len(knots) - 1)
    lower = index - 1
    fraction = (query - knots[lower]) / (knots[index] - knots[lower])
    return values[lower] + fraction * (values[index] - values[lower])


class TorchFrozenDisplay(nn.Module):
    """Exact piecewise-linear frozen mapping; no learnable calibration parameters."""
    def __init__(self, frozen_display, bounds, device):
        super().__init__()
        from utils.flir_radiometry import blackbody_signal
        self.document = frozen_display.document
        self.sha256 = frozen_display.sha256
        calibration = self.document["signal_calibration"]
        self.q_min = float(blackbody_signal(float(bounds[0]), calibration))
        self.q_span = float(blackbody_signal(float(bounds[1]), calibration)) - self.q_min
        if self.q_span <= 0:
            raise ValueError("Invalid display signal normalization")
        self.group_indices = {}
        for index, (key, group) in enumerate(self.document["groups"].items()):
            self.group_indices[key] = index
            response = group["response"]
            for suffix, value in (("x", response["window_knots"]), ("y", response["palette_fraction_knots"]),
                                  ("rgb", group["palette_rgb"]), ("tails", response["tail_slopes"])):
                self.register_buffer("g{}_{}".format(index, suffix), torch.tensor(value, dtype=torch.float64, device=device))

    def forward(self, normalized, name):
        if normalized.ndim != 3 or normalized.shape[0] != 1:
            raise ValueError("Display input must be 1xHxW")
        settings = self.document["frame_display_metadata"][name]
        group = self.group_indices[settings["palette_sha256"]]
        x, y, rgb, tails = (getattr(self, "g{}_{}".format(group, suffix)) for suffix in ("x", "y", "rgb", "tails"))
        # Float64 matches the existing NumPy evaluator, including narrow windows.
        signal = normalized[0].double() * self.q_span + self.q_min
        z = (signal - settings["signal_window_center"]) / settings["raw_value_range"] + .5
        fraction = linear_interp(z, x, y)
        fraction = torch.where(z < x[0], y[0] + (z - x[0]) * tails[0], fraction)
        fraction = torch.where(z > x[-1], y[-1] + (z - x[-1]) * tails[1], fraction).clamp(0, 1)
        palette_x = torch.linspace(0, 1, len(rgb), dtype=rgb.dtype, device=rgb.device)
        return torch.stack([linear_interp(fraction, palette_x, rgb[:, channel]) for channel in range(3)]).to(normalized.dtype)


def masked_ssim(prediction, target, mask=None, window_size=11):
    """Gaussian-window SSIM, using only windows fully inside the valid native FOV."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("SSIM requires matching CxHxW tensors")
    channels = prediction.shape[0]
    coordinate = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype) - window_size // 2
    kernel = torch.exp(-coordinate.square() / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(channels, 1, window_size, window_size).contiguous()
    def blur(value):
        return F.conv2d(value[None], window, padding=window_size // 2, groups=channels)[0]
    mean_a, mean_b = blur(prediction), blur(target)
    var_a = blur(prediction.square()) - mean_a.square()
    var_b = blur(target.square()) - mean_b.square()
    covariance = blur(prediction * target) - mean_a * mean_b
    score = ((2 * mean_a * mean_b + .01 ** 2) * (2 * covariance + .03 ** 2) /
             ((mean_a.square() + mean_b.square() + .01 ** 2) * (var_a + var_b + .03 ** 2)))
    if mask is None:
        return score.mean()
    valid = mask.to(prediction).reshape(1, 1, *prediction.shape[-2:])
    # Invalid/padded neighbours must not leak into structural supervision.
    inside = F.conv2d(valid, torch.ones(1, 1, window_size, window_size, device=valid.device, dtype=valid.dtype),
                      padding=window_size // 2) >= window_size ** 2 - 1e-5
    return masked_mean(score, inside[0])


def image_loss(prediction, target, mask=None):
    return .8 * masked_mean((prediction - target).abs(), mask) + .2 * (1 - masked_ssim(prediction, target, mask))


def joint_signal(field, planck, gaussians, camera, sh_field):
    physical = field.radiance(planck)
    # Include the derivative of SH direction with respect to Gaussian position.
    delta = sh_field(gaussians.get_xyz, camera.camera_center, detach_geometry=False)
    return (physical + delta).clamp(0, 1)


def sequential_joint_step(optimizer, rgb_loss, thermal_loss):
    """Callbacks build separate graphs, with one update after both backward passes."""
    optimizer.zero_grad(set_to_none=True)
    rgb = rgb_loss()
    if rgb.requires_grad:
        rgb.backward()
    rgb_value = float(rgb.detach())
    del rgb
    thermal = thermal_loss()
    thermal.backward()
    thermal_value = float(thermal.detach())
    del thermal
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("Non-finite joint gradient: " + group["name"])
    optimizer.step()
    return rgb_value, thermal_value
