"""Frozen, local temperature prior with RGB/material boundary attenuation."""
import math

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F

from utils.thermal_stage3 import tensor_digest


class TemperatureSpatialPrior(nn.Module):
    def __init__(self, source, target, weights, temperature_scale_K, diagnostics):
        super().__init__()
        self.temperature_scale_K = float(temperature_scale_K)
        self.register_buffer("source", source)
        self.register_buffer("target", target)
        self.register_buffer("weights", weights)
        self.protocol = dict(diagnostics, temperature_scale_K=self.temperature_scale_K,
                             graph_sha256=tensor_digest({"source": source, "target": target, "weights": weights}),
                             loss="weighted_mean_SmoothL1((Ti-Tj)/scale,beta=1)",
                             temperature_truth_used=False)

    def forward(self, temperature):
        if not self.source.numel():
            return temperature.sum() * 0
        values = temperature.reshape(-1)
        residual = (values[self.source] - values[self.target]) / self.temperature_scale_K
        loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1., reduction="none")
        return (loss * self.weights).sum() / self.weights.sum()

    @classmethod
    @torch.no_grad()
    def build(cls, xyz, opacity, rgb_dc, material_ids, fit_support, neighbors=6,
              temperature_scale_K=10., distance_cutoff=3., rgb_sigma=.15,
              material_boundary_weight=.1, unknown_weight=.1):
        if (neighbors < 1 or not all(math.isfinite(v) and v > 0 for v in
                (temperature_scale_K, distance_cutoff, rgb_sigma))
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in
                           (material_boundary_weight, unknown_weight))):
            raise ValueError("Invalid spatial temperature prior settings")
        count = len(xyz)
        if xyz.shape != (count, 3) or rgb_dc.shape != (count, 3):
            raise ValueError("Spatial xyz/RGB shape mismatch")
        points = xyz.detach().cpu().numpy()
        colors = rgb_dc.detach().clamp(0, 1).cpu().numpy()
        alpha = opacity.detach().reshape(-1).cpu().numpy()
        labels = material_ids.detach().reshape(-1).cpu().numpy()
        support = fit_support.detach().reshape(-1).cpu().numpy().astype(bool)
        if any(len(value) != count for value in (alpha, labels, support)):
            raise ValueError("Spatial prior Gaussian count mismatch")
        if (not all(np.isfinite(value).all() for value in (points, colors, alpha))
                or np.any((alpha < 0) | (alpha > 1)) or np.any(labels < -1)):
            raise ValueError("Invalid spatial prior inputs")
        candidates = np.empty((0, 2), dtype=np.int64)
        if count >= 2:
            k = min(int(neighbors) + 1, count)
            _, indices = cKDTree(points).query(points, k=k)
            src = np.repeat(np.arange(count), k)
            dst = indices.reshape(-1)
            keep = (src != dst) & support[src] & support[dst]
            candidates = np.unique(np.sort(np.stack((src[keep], dst[keep]), axis=1), axis=1), axis=0)
        source, target = candidates[:, 0], candidates[:, 1]
        distance = np.linalg.norm(points[source] - points[target], axis=1)
        positive = distance[distance > 0]
        scale = max(float(np.median(positive)) if len(positive) else 1e-8, 1e-8)
        near = distance <= distance_cutoff * scale
        source, target, distance = source[near], target[near], distance[near]
        unknown = (labels[source] < 0) | (labels[target] < 0)
        boundary = (~unknown) & (labels[source] != labels[target])
        material_weights = np.ones(len(source), dtype=np.float32)
        material_weights[boundary] = material_boundary_weight
        material_weights[unknown] = unknown_weight
        # Unknown is never interpreted as a single homogeneous material.
        rgb_difference = np.square(colors[source] - colors[target]).mean(axis=1)
        weights = (np.exp(-np.square(distance / scale)) * np.exp(-rgb_difference / rgb_sigma ** 2)
                   * np.sqrt(alpha[source] * alpha[target]) * material_weights)
        retained = np.isfinite(weights) & (weights > 1e-12)
        device = xyz.device
        diagnostics = {"kind": "local_RGB_material_opacity_fit_support_temperature_prior_v1",
                       "neighbors": int(neighbors), "distance_cutoff_multiple": float(distance_cutoff),
                       "distance_scale_scene_units": scale, "rgb_sigma": float(rgb_sigma),
                       "material_boundary_weight": float(material_boundary_weight), "unknown_weight": float(unknown_weight),
                       "candidate_unique_edges": len(candidates), "distance_filtered_edges": int((~near).sum()),
                       "weight_filtered_edges": int((~retained).sum()), "retained_edges": int(retained.sum()),
                       "known_material_boundary_edges": int((boundary & retained).sum()),
                       "unknown_touching_edges": int((unknown & retained).sum()),
                       "fit_support_gaussians": int(support.sum()), "RGB_source": "frozen_D_RGB_DC",
                       "support_source": "frozen_common_SH_fit_support", "graph_trainable": False}
        return cls(torch.as_tensor(source[retained].copy(), dtype=torch.long, device=device),
                   torch.as_tensor(target[retained].copy(), dtype=torch.long, device=device),
                   torch.as_tensor(weights[retained].copy(), dtype=xyz.dtype, device=device),
                   temperature_scale_K, diagnostics)
