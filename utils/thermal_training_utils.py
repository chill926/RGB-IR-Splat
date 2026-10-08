"""Loss units and stopping state shared by physical and free-signal training.

Torch is imported lazily so protocol/stopping tests can run without CUDA/Torch.
The loss scale changes residual units only, never camera calibration or targets.
"""
import math
import numpy as np


def option(config, name, default):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def scale_from_samples(samples, floor=1e-4):
    if not math.isfinite(float(floor)) or floor <= 0:
        raise ValueError("Loss-scale floor must be finite and positive")
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Fit loss-scale samples must be nonempty and finite")
    lower, upper = np.quantile(values, [0.01, 0.99])
    return max(float(upper - lower), float(floor)), float(lower), float(upper)


def install_loss_scale(args, fit_cameras, transform=None):
    mode = args.loss_scale_mode
    if mode == "auto":
        mode = "fit_quantile" if args.observation_domain == "raw_rjpeg" else "legacy"
    if mode not in ("legacy", "fit_quantile"):
        raise ValueError("Unsupported loss-scale mode")
    recorded = getattr(args, "loss_scale_protocol", None)
    if recorded is not None:
        if recorded.get("mode") != mode or recorded.get("fit_camera_names") != [c.image_name for c in fit_cameras]:
            raise ValueError("Loss-scale mode/fit cameras differ from the reference checkpoint")
        if not math.isfinite(float(recorded["scale"])) or recorded["scale"] <= 0:
            raise ValueError("Invalid recorded loss scale")
        args.loss_scale_protocol = recorded
        return
    samples = []
    if mode == "fit_quantile":
        for camera in fit_cameras:
            image = getattr(camera, "original_physical_image", None)
            image = camera.original_image if image is None else image
            sample = image.detach().mean(dim=0, keepdim=True)[..., ::16, ::16]
            if transform is not None:
                sample = transform(sample)
            samples.append(sample.cpu().numpy().reshape(-1))
        scale, lower, upper = scale_from_samples(np.concatenate(samples), args.loss_scale_floor)
    else:
        scale, lower, upper = 1.0, None, None
    args.loss_scale_protocol = {
        "format_version": 1, "mode": mode, "scale": scale,
        "quantiles": [0.01, 0.99], "quantile_values": [lower, upper],
        "sample_stride": 16, "scale_floor": args.loss_scale_floor,
        "fit_camera_names": [c.image_name for c in fit_cameras],
        "units": "normalized_camera_signal" if args.observation_domain == "raw_rjpeg" else "model_observation",
        "held_out_pixels_used": False,
    }


def thermal_data_loss(prediction, target, config):
    import torch
    import torch.nn.functional as F
    protocol = option(config, "loss_scale_protocol", None) or {}
    scale = float(protocol.get("scale", 1.0))
    beta = float(option(config, "huber_delta", 0.02))
    if not math.isfinite(scale) or not math.isfinite(beta) or min(scale, beta) <= 0:
        raise ValueError("Huber beta and loss scale must be finite and positive")
    residual = (prediction - target) / scale
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=beta)


def held_lr(step, steps, start, end, hold_fraction):
    if steps <= 0 or min(start, end) <= 0 or not 0 <= hold_fraction < 1:
        raise ValueError("Invalid held learning-rate schedule")
    hold = int(steps * hold_fraction)
    progress = min(max((step - hold) / float(max(steps - hold, 1)), 0.0), 1.0)
    return math.exp(math.log(start) * (1 - progress) + math.log(end) * progress)


class PlateauTracker:
    """Track the actual minimum separately from cumulative meaningful progress."""
    def __init__(self, absolute_delta=1e-8, relative_delta=0.002):
        if min(absolute_delta, relative_delta) < 0 or not all(map(math.isfinite, (absolute_delta, relative_delta))):
            raise ValueError("Plateau deltas must be finite and non-negative")
        self.absolute_delta, self.relative_delta = absolute_delta, relative_delta
        self.best = self.reference = float("inf")
        self.best_step = self.stale = 0

    def update(self, value, step):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Validation loss must be finite and non-negative")
        improved = value < self.best
        if improved:
            self.best, self.best_step = value, int(step)
        threshold = max(self.absolute_delta, abs(self.reference) * self.relative_delta)
        meaningful = math.isinf(self.reference) or value < self.reference - threshold
        if meaningful:
            self.reference, self.stale = value, 0
        else:
            self.stale += 1
        return improved, meaningful


def quality_status(metrics, apparent_mae_K=None, signal_rmse_Q=None):
    checks = []
    for key, limit in (("apparent_temperature_MAE_K", apparent_mae_K),
                       ("camera_signal_RMSE", signal_rmse_Q)):
        if limit is not None:
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError("Quality thresholds must be finite and positive")
            if key not in metrics or not math.isfinite(metrics[key]):
                raise ValueError("Requested quality metric unavailable: " + key)
            checks.append(metrics[key] <= limit)
    return "unassessed" if not checks else ("passed" if all(checks) else "poor_fit")
