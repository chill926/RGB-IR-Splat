"""D-preserving temperature controls and a global, normal-dependent SH environment.

Temperature updates use frozen local interpolation, plus a bounded penalized
residual. D's fine detail is a fixed offset, not temperature ground truth.
The existing per-Gaussian thermal appearance SH is an independent, frozen model.
"""
import copy
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import torch
from torch import nn

from utils.thermal_physics import MaterialThermalField
from utils.thermal_sh_residual import sh2_basis
from utils.thermal_stage3 import tensor_digest


@torch.no_grad()
def frozen_surface_normals(xyz, scaling, rotations, fit_camera_centers):
    """Shortest Gaussian axis, oriented toward nearest fit camera (a proxy).

    Confidence is zero for spherical Gaussians. No held-out camera is used.
    """
    from utils.general_utils import build_rotation
    matrices = build_rotation(rotations)
    order = scaling.argsort(dim=1)
    normals = matrices[torch.arange(len(xyz), device=xyz.device), :, order[:, 0]]
    centers = fit_camera_centers.detach().cpu().numpy()
    if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers):
        raise ValueError("Normals require fit-only camera centers")
    _, closest = cKDTree(centers).query(xyz.detach().cpu().numpy())
    toward_camera = fit_camera_centers[torch.as_tensor(closest, device=xyz.device)] - xyz
    normals = torch.where((normals * toward_camera).sum(1, keepdim=True) < 0, -normals, normals)
    sorted_scales = scaling.gather(1, order)
    confidence = (1 - sorted_scales[:, :1] / sorted_scales[:, 1:2].clamp_min(1e-12)).clamp(0, 1)
    return normals, confidence


@torch.no_grad()
def build_temperature_basis(xyz, material_ids, fit_support, spatial_prior, normals,
                            max_controls=4096, interpolation_neighbors=4,
                            min_component_size=4):
    """Voxel controls within fit-supported, material/surface graph components.

    Small disconnected components keep D temperature. Unknown is not given a
    global shared control: connectivity and spatial cells still separate it.
    """
    if max_controls < 1 or interpolation_neighbors < 1 or min_component_size < 2:
        raise ValueError("Invalid temperature basis dimensions")
    points = xyz.detach().cpu().numpy()
    labels = material_ids.detach().cpu().numpy()
    support = fit_support.detach().cpu().numpy().astype(bool)
    normal = normals.detach().cpu().numpy()
    src = spatial_prior.source.cpu().numpy()
    dst = spatial_prior.target.cpu().numpy()
    # Hard boundaries here prevent control sharing across different materials.
    keep = ((labels[src] == labels[dst]) & support[src] & support[dst]
            & ((normal[src] * normal[dst]).sum(1) >= .5))
    src, dst = src[keep], dst[keep]
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(len(points), len(points)))
    _, component = connected_components(graph, directed=False)
    sizes = np.bincount(component)
    active = support & (sizes[component] >= min_component_size)
    active_ids = np.flatnonzero(active)
    if len(active_ids) < 2:
        raise ValueError("Insufficient connected fit support for temperature controls")
    group_ids = component[active_ids]
    group_count = len(np.unique(group_ids))
    if group_count > max_controls:
        raise ValueError("More surface components than max_controls; increase the control budget")
    p = points[active_ids]
    origin = p.min(0)
    cell = max(float(np.max(np.ptp(p, axis=0))) / max_controls ** (1 / 3), 1e-8)
    for _ in range(100):
        voxel = np.floor((p - origin) / cell).astype(np.int64)
        keys = np.column_stack((group_ids, voxel))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        if len(unique) <= max_controls:
            break
        cell *= 1.3
    else:
        raise ValueError("Could not build a bounded temperature basis")
    count = len(unique)
    sums = np.zeros((count, 3), dtype=np.float64)
    np.add.at(sums, inverse, p)
    control_positions = sums / np.bincount(inverse)[:, None]
    width = min(interpolation_neighbors, count)
    indices = np.zeros((len(points), width), dtype=np.int64)
    weights = np.zeros((len(points), width), dtype=np.float32)
    for group in np.unique(group_ids):
        local_rows = np.flatnonzero(group_ids == group)
        local_controls = np.flatnonzero(unique[:, 0] == group)
        k = min(width, len(local_controls))
        distances, nearest = cKDTree(control_positions[local_controls]).query(p[local_rows], k=k)
        distances, nearest = distances.reshape(-1, k), nearest.reshape(-1, k)
        local_weights = 1 / (distances ** 2 + (.1 * cell) ** 2)
        local_weights /= local_weights.sum(1, keepdims=True)
        rows = active_ids[local_rows]
        indices[rows, :k] = local_controls[nearest]
        weights[rows, :k] = local_weights.astype(np.float32)
    protocol = {
        "kind": "frozen_D_detail_plus_surface_control_updates_v1",
        "max_controls": int(max_controls), "control_count": int(count),
        "interpolation_neighbors": int(width), "voxel_cell_scene_units": cell,
        "normal_cosine_min": .5, "min_component_size": int(min_component_size),
        "active_gaussians": int(active.sum()), "frozen_gaussians": int((~active).sum()),
        "surface_components": group_count, "boundary_policy": "same_material_and_graph_component",
        "initial_temperature_source": "frozen_D_estimate_not_temperature_GT",
    }
    device = xyz.device
    buffers = {"control_indices": torch.as_tensor(indices, device=device),
               "control_weights": torch.as_tensor(weights, device=device),
               "temperature_support": torch.as_tensor(active[:, None], device=device),
               "control_positions": torch.as_tensor(control_positions, device=device, dtype=xyz.dtype)}
    return buffers, protocol


class StructuredThermalField(MaterialThermalField):
    KIND = "surface_temperature_controls_global_SH2_environment_v1"
    STRUCTURE_BUFFERS = ("control_indices", "control_weights", "temperature_support",
                         "control_positions", "initial_temperature", "environment_normals",
                         "environment_normal_confidence")

    def __init__(self, base, basis, normals, normal_confidence, protocol,
                 control_bound_K=5., residual_bound_K=.5,
                 residual_sigma_K=.1, environment_sh_bound=.01):
        if not all(math.isfinite(v) and v > 0 for v in
                   (control_bound_K, residual_bound_K, residual_sigma_K, environment_sh_bound)):
            raise ValueError("Structured field bounds must be positive and finite")
        state = base.state_dict()
        config = base.export_state()["config"]
        options = {key: state[key] for key in ("material_confidence", "learn_k_by_material",
            "learn_delta_by_material", "k_prior_by_material", "sigma_k_by_material", "sigma_delta_by_material")}
        super().__init__(state["material_ids"], torch.sigmoid(state["temperature_raw"]),
                         state["epsilon0_by_material"], initial_environment=float(base.environment),
                         branch=base.branch, **config, **options)
        self.load_state_dict(state)
        self.control_bound_K = float(control_bound_K)
        self.residual_bound_K = float(residual_bound_K)
        self.residual_sigma_K = float(residual_sigma_K)
        self.environment_sh_bound = float(environment_sh_bound)
        self.structure_protocol = copy.deepcopy(protocol)
        for name, value in basis.items():
            self.register_buffer(name, value.detach().clone())
        self.register_buffer("initial_temperature", base.temperature.detach().clone())
        self.register_buffer("environment_normals", normals.detach().clone())
        self.register_buffer("environment_normal_confidence", normal_confidence.detach().clone())
        self.temperature_control_raw = nn.Parameter(base.temperature_raw.new_zeros(len(self.control_positions), 1))
        self.temperature_residual_raw = nn.Parameter(torch.zeros_like(base.temperature_raw))
        self.environment_sh_raw = nn.Parameter(base.environment_raw.new_zeros(8))
        self.set_branch_trainability()
        self.frozen_structure_sha256 = self.structure_digest()

    def set_branch_trainability(self):
        super().set_branch_trainability()
        self.temperature_raw.requires_grad_(False)

    @property
    def temperature_control_delta(self):
        return self.control_bound_K * self.temperature_control_raw.tanh()

    @property
    def temperature_local_delta(self):
        return self.residual_bound_K * self.temperature_residual_raw.tanh() * self.temperature_support

    @property
    def temperature(self):
        coarse = (self.temperature_control_delta[self.control_indices, 0] * self.control_weights).sum(1, keepdim=True)
        return (self.initial_temperature + coarse + self.temperature_local_delta).clamp(self.temp_min, self.temp_max)

    @property
    def environment_sh_coefficients(self):
        # SH addition theorem bounds all directional evaluations, including unseen normals.
        energy = 8 / (4 * math.pi)
        allowed = torch.minimum(self.environment.new_tensor(self.environment_sh_bound),
                                .95 * torch.minimum(self.environment, 1 - self.environment))
        return allowed * self.environment_sh_raw / torch.sqrt(energy * (1 + self.environment_sh_raw.square().sum()))

    @property
    def environment_by_gaussian(self):
        directional = (sh2_basis(self.environment_normals) * self.environment_sh_coefficients).sum(1, keepdim=True)
        return self.environment + directional * self.environment_normal_confidence

    def radiance(self, planck_lut):
        epsilon = self.emissivity()
        return epsilon * planck_lut(self.temperature) + (1 - epsilon) * self.environment_by_gaussian

    def structure_digest(self):
        return tensor_digest({name: getattr(self, name) for name in self.STRUCTURE_BUFFERS})

    def structure_regularizers(self):
        local = (self.temperature_local_delta / self.residual_sigma_K).square()
        active = self.temperature_support.reshape(-1)
        residual = local.reshape(-1)[active].mean() if bool(active.any()) else local.sum() * 0
        control = (self.temperature_control_delta / self.control_bound_K).square().mean()
        environment_sh = (self.environment_sh_coefficients / self.environment_sh_bound).square().sum()
        return residual, control, environment_sh

    @torch.no_grad()
    def structured_diagnostics(self):
        return {"temperature_controls": len(self.temperature_control_raw),
                "temperature_control_delta_max_K": float(self.temperature_control_delta.abs().max()),
                "temperature_local_delta_max_K": float(self.temperature_local_delta.abs().max()),
                "temperature_delta_mean_abs_K": float((self.temperature - self.initial_temperature).abs().mean()),
                "environment_sh_coefficients": self.environment_sh_coefficients.cpu().tolist(),
                "environment_range": [float(self.environment_by_gaussian.min()), float(self.environment_by_gaussian.max())]}

    def export_state(self):
        payload = super().export_state()
        payload["field_parameterization"] = {
            "kind": self.KIND, "basis": self.structure_protocol,
            "control_bound_K": self.control_bound_K, "residual_bound_K": self.residual_bound_K,
            "residual_sigma_K": self.residual_sigma_K, "environment_sh_bound": self.environment_sh_bound,
            "structure_sha256": self.frozen_structure_sha256,
            "environment": "E0_plus_global_degree_1_2_SH_on_frozen_surface_normals",
            "environment_parameter_count": 9,
            "normal_source": "shortest_Gaussian_axis_oriented_to_nearest_fit_camera",
            "normal_confidence": "1_minus_min_scale_over_middle_scale",
            "legacy_appearance_SH": "preserved_and_frozen_independent_residual",
        }
        return payload

    @classmethod
    def restore(cls, checkpoint, branch, device, reset_branch_parameters=False):
        description = checkpoint["field_parameterization"]
        if description["kind"] != cls.KIND:
            raise ValueError("Unknown structured thermal field format")
        legacy = dict(checkpoint)
        legacy.pop("field_parameterization")
        base = MaterialThermalField.from_checkpoint(legacy, branch, device, reset_branch_parameters=False)
        state = checkpoint["state_dict"]
        basis = {name: state[name].to(device) for name in
                 ("control_indices", "control_weights", "temperature_support", "control_positions")}
        field = cls(base, basis, state["environment_normals"].to(device),
                    state["environment_normal_confidence"].to(device), description["basis"],
                    **{name: description[name] for name in
                       ("control_bound_K", "residual_bound_K", "residual_sigma_K", "environment_sh_bound")}).to(device)
        field.load_state_dict(state, strict=True)
        if field.structure_digest() != description["structure_sha256"]:
            raise ValueError("Structured field frozen geometry/basis hash mismatch")
        field.frozen_structure_sha256 = description["structure_sha256"]
        if reset_branch_parameters:
            field.k_raw.data.zero_()
            field.delta_epsilon_raw.data.zero_()
        return field
