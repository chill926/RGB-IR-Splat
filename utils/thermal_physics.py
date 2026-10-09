import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from torch import nn
import torch.nn.functional as F


class UniformLWIRPlanckLUT(nn.Module):
    """Differentiable, uniformly weighted 8--14 um passband Planck LUT."""
    def __init__(self, temp_min=250.0, temp_max=450.0, num_temp=4096, num_lambda=256,
                 signal_calibration=None):
        super().__init__()
        temperatures = torch.linspace(float(temp_min), float(temp_max), int(num_temp), dtype=torch.float64)
        if signal_calibration is None:
            wavelengths = torch.linspace(8e-6, 14e-6, int(num_lambda), dtype=torch.float64)
            h, c, kb = 6.62607015e-34, 299792458.0, 1.380649e-23
            lam, temp = wavelengths[:, None], temperatures[None, :]
            exponent = (h * c / (lam * kb * temp)).clamp_max(80.0)
            spectral = (2.0 * h * c * c) / lam.pow(5) / torch.expm1(exponent)
            # Uniform normalized spectral response: integral R(lambda) dlambda = 1.
            response = torch.full_like(wavelengths, 1.0 / float(wavelengths[-1] - wavelengths[0]))
            radiance = torch.trapz(response[:, None] * spectral, wavelengths, dim=0)
        else:
            from utils.flir_radiometry import validate_calibration
            coefficients = validate_calibration(signal_calibration)
            denominator = coefficients["PlanckR2"] * (
                torch.exp(coefficients["PlanckB"] / temperatures) - coefficients["PlanckF"])
            radiance = coefficients["PlanckR1"] / denominator
            if not bool((denominator > 0).all()) or not bool(torch.isfinite(radiance).all()):
                raise ValueError("Temperature bounds exceed the valid FLIR camera-response range")
        if not bool((radiance[1:] > radiance[:-1]).all()):
            raise ValueError("Blackbody response LUT must be strictly increasing")
        normalized = (radiance - radiance[0]) / (radiance[-1] - radiance[0])
        self.temp_min, self.temp_max = float(temp_min), float(temp_max)
        self.register_buffer("temperatures", temperatures.float())
        self.register_buffer("radiance", normalized.float())
        forward_slopes = torch.empty_like(normalized)
        forward_slopes[1:-1] = 0.5 * (normalized[2:] - normalized[:-2])
        forward_slopes[0] = normalized[1] - normalized[0]
        forward_slopes[-1] = normalized[-1] - normalized[-2]
        inverse_slopes = torch.empty_like(temperatures)
        inverse_slopes[1:-1] = ((temperatures[2:] - temperatures[:-2]) /
                                (normalized[2:] - normalized[:-2]))
        inverse_slopes[0] = (temperatures[1] - temperatures[0]) / (normalized[1] - normalized[0])
        inverse_slopes[-1] = (temperatures[-1] - temperatures[-2]) / (normalized[-1] - normalized[-2])
        self.register_buffer("forward_slopes", forward_slopes.float())
        self.register_buffer("inverse_slopes", inverse_slopes.float())
        self.register_buffer("physical_radiance_min", radiance[0].float())
        self.register_buffer("physical_radiance_max", radiance[-1].float())

    def forward(self, temperature):
        temperature = temperature.clamp(self.temp_min, self.temp_max)
        position = (temperature - self.temp_min) / (self.temp_max - self.temp_min) * (self.radiance.numel() - 1)
        lower = position.floor().long().clamp(0, self.radiance.numel() - 2)
        fraction = position - lower.to(position.dtype)
        f2, f3 = fraction.square(), fraction.pow(3)
        h00, h10 = 2 * f3 - 3 * f2 + 1, f3 - 2 * f2 + fraction
        h01, h11 = -2 * f3 + 3 * f2, f3 - f2
        value = (h00 * self.radiance[lower] + h10 * self.forward_slopes[lower] +
                 h01 * self.radiance[lower + 1] + h11 * self.forward_slopes[lower + 1])
        return value.clamp(0.0, 1.0)

    def inverse(self, normalized_radiance):
        value = normalized_radiance.clamp(0.0, 1.0)
        index = torch.searchsorted(self.radiance, value.reshape(-1)).clamp(1, self.radiance.numel() - 1)
        y0, y1 = self.radiance[index - 1], self.radiance[index]
        t0, t1 = self.temperatures[index - 1], self.temperatures[index]
        interval = y1 - y0
        weight = (value.reshape(-1) - y0) / (interval + 1e-12)
        w2, w3 = weight.square(), weight.pow(3)
        h00, h10 = 2 * w3 - 3 * w2 + 1, w3 - 2 * w2 + weight
        h01, h11 = -2 * w3 + 3 * w2, w3 - w2
        temperature = (h00 * t0 + h10 * interval * self.inverse_slopes[index - 1] +
                       h01 * t1 + h11 * interval * self.inverse_slopes[index])
        return temperature.reshape_as(value).clamp(self.temp_min, self.temp_max)

    def normalize_physical_radiance(self, physical_radiance):
        return ((physical_radiance - self.physical_radiance_min) /
                (self.physical_radiance_max - self.physical_radiance_min)).clamp(0.0, 1.0)


class FlirCameraResponseLUT(UniformLWIRPlanckLUT):
    """Normalized FLIR blackbody camera signal, using this capture's coefficients.

    Shares interpolation/inverse with the legacy LUT, but its unnormalized units
    are Q=DN+O, not the uniform 8--14 um SI spectral radiance approximation.
    """
    def __init__(self, calibration, temp_min=250.0, temp_max=450.0, num_temp=4096):
        super().__init__(temp_min, temp_max, num_temp=num_temp,
                         signal_calibration=calibration)


class MaterialThermalField(nn.Module):
    """Per-Gaussian temperature with global environment and material parameters.

    ``material_ids == -1`` denotes an unknown/ambiguous material. Unknown
    Gaussians still receive a fixed fallback emissivity, but never update a
    material-shared K/R parameter.
    """
    BRANCHES = ("stage2", "C", "K", "R")

    def __init__(self, material_ids, initial_radiance, epsilon0_by_material,
                 material_names=None, material_confidence=None,
                 learn_k_by_material=None, learn_delta_by_material=None,
                 k_prior_by_material=None, sigma_k_by_material=None,
                 sigma_delta_by_material=None, unknown_epsilon0=0.95,
                 branch="stage2", temp_min=250.0, temp_max=450.0,
                 temp_ref=300.0, k_max=0.001, delta_epsilon_max=0.2,
                 initial_environment=None):
        super().__init__()
        if branch not in self.BRANCHES:
            raise ValueError("branch must be one of: " + ", ".join(self.BRANCHES))
        material_ids = material_ids.reshape(-1).long()
        epsilon0_by_material = torch.as_tensor(epsilon0_by_material, dtype=torch.float32).reshape(-1)
        if epsilon0_by_material.numel() < 1:
            raise ValueError("At least one material must be defined")
        if material_ids.numel() and int(material_ids.max()) >= epsilon0_by_material.numel():
            raise ValueError("material_ids contains an id absent from epsilon0_by_material")
        if material_ids.numel() and int(material_ids.min()) < -1:
            raise ValueError("Only -1 may be used as the unknown material id")
        if not float(temp_min) < float(temp_ref) < float(temp_max):
            raise ValueError("Temperature bounds must satisfy temp_min < temp_ref < temp_max")
        if not bool(((epsilon0_by_material >= 0.01) & (epsilon0_by_material <= 0.99)).all()):
            raise ValueError("Every fixed epsilon_0 must lie in [0.01, 0.99]")
        if not 0.01 <= float(unknown_epsilon0) <= 0.99:
            raise ValueError("unknown_epsilon0 must lie in [0.01, 0.99]")
        if float(k_max) <= 0.0 or float(delta_epsilon_max) <= 0.0:
            raise ValueError("Material parameter bounds must be positive")
        initial_radiance = initial_radiance.reshape(-1, 1).clamp(1e-4, 1 - 1e-4)
        if initial_radiance.shape[0] != material_ids.shape[0]:
            raise ValueError("Initial temperature proxy and material_ids must have equal length")
        self.branch = branch
        self.temp_min, self.temp_max, self.temp_ref = float(temp_min), float(temp_max), float(temp_ref)
        self.k_max, self.delta_epsilon_max = float(k_max), float(delta_epsilon_max)
        self.material_names = list(material_names or [f"material_{idx}" for idx in range(epsilon0_by_material.numel())])
        if len(self.material_names) != epsilon0_by_material.numel():
            raise ValueError("material_names and epsilon0_by_material must have equal length")
        self.unknown_epsilon0 = float(unknown_epsilon0)
        self.register_buffer("material_ids", material_ids)
        self.register_buffer("material_valid", material_ids >= 0)
        confidence = (torch.ones_like(material_ids, dtype=torch.float32) if material_confidence is None
                      else torch.as_tensor(material_confidence, dtype=torch.float32).reshape(-1))
        if confidence.shape != material_ids.shape:
            raise ValueError("material_confidence and material_ids must have equal length")
        self.register_buffer("material_confidence", confidence.clamp(0.0, 1.0))
        self.register_buffer("epsilon0_by_material", epsilon0_by_material)
        count = epsilon0_by_material.numel()
        def material_vector(value, default, dtype=torch.float32):
            tensor = torch.full((count,), default, dtype=dtype) if value is None else torch.as_tensor(value, dtype=dtype).reshape(-1)
            if tensor.numel() != count:
                raise ValueError("Every material parameter vector must match epsilon0_by_material")
            return tensor
        self.register_buffer("learn_k_by_material", material_vector(learn_k_by_material, True, torch.bool))
        self.register_buffer("learn_delta_by_material", material_vector(learn_delta_by_material, True, torch.bool))
        self.register_buffer("k_prior_by_material", material_vector(k_prior_by_material, 0.0))
        self.register_buffer("sigma_k_by_material", material_vector(sigma_k_by_material, 1e-4).clamp_min(1e-12))
        self.register_buffer("sigma_delta_by_material", material_vector(sigma_delta_by_material, 0.05).clamp_min(1e-12))
        self.temperature_raw = nn.Parameter(torch.logit(initial_radiance))
        ambient = (float(torch.quantile(initial_radiance.detach(), 0.25))
                   if initial_environment is None else float(initial_environment))
        if not 0.0 < ambient < 1.0:
            raise ValueError("Initial environment irradiance must lie strictly inside (0, 1)")
        ambient_raw = math.log(max(ambient, 1e-4) / max(1 - ambient, 1e-4))
        # The first model deliberately uses one environment radiance for the
        # complete scene. It is not indexed by Gaussian or material.
        self.environment_raw = nn.Parameter(initial_radiance.new_tensor([ambient_raw]))
        self.k_raw = nn.Parameter(torch.zeros(count, device=initial_radiance.device))
        self.delta_epsilon_raw = nn.Parameter(torch.zeros(count, device=initial_radiance.device))

    @property
    def temperature(self):
        return self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(self.temperature_raw)

    @property
    def environment(self):
        return torch.sigmoid(self.environment_raw)

    @property
    def k_epsilon_by_material(self):
        return self.k_max * torch.tanh(self.k_raw) * self.learn_k_by_material.to(self.k_raw)

    @property
    def delta_epsilon_by_material(self):
        return (self.delta_epsilon_max * torch.tanh(self.delta_epsilon_raw) *
                self.learn_delta_by_material.to(self.delta_epsilon_raw))

    def base_emissivity(self):
        safe_ids = self.material_ids.clamp_min(0)
        known = self.epsilon0_by_material[safe_ids, None]
        fallback = known.new_full(known.shape, self.unknown_epsilon0)
        return torch.where(self.material_valid[:, None], known, fallback)

    def emissivity(self):
        epsilon0 = self.base_emissivity()
        safe_ids = self.material_ids.clamp_min(0)
        if self.branch == "K":
            k = self.k_epsilon_by_material[safe_ids, None]
            k = torch.where(self.material_valid[:, None], k, torch.zeros_like(k))
            return (epsilon0 + k * (self.temperature - self.temp_ref)).clamp(0.01, 0.99)
        if self.branch == "R":
            delta = self.delta_epsilon_by_material[safe_ids, None]
            delta = torch.where(self.material_valid[:, None], delta, torch.zeros_like(delta))
            return (epsilon0 + delta).clamp(0.01, 0.99)
        return epsilon0

    def radiance(self, planck_lut):
        emission, epsilon = planck_lut(self.temperature), self.emissivity()
        return epsilon * emission + (1 - epsilon) * self.environment

    def branch_regularizer(self, delta=1e-12):
        if self.branch == "K":
            mask = self.learn_k_by_material
            return (((self.k_epsilon_by_material[mask] - self.k_prior_by_material[mask]).square()) /
                    (self.sigma_k_by_material[mask].square() + float(delta))).sum()
        if self.branch == "R":
            mask = self.learn_delta_by_material
            return (self.delta_epsilon_by_material[mask].square() /
                    (self.sigma_delta_by_material[mask].square() + float(delta))).sum()
        return self.temperature_raw.new_zeros(())

    def set_branch_trainability(self):
        self.temperature_raw.requires_grad_(True)
        self.environment_raw.requires_grad_(True)
        self.k_raw.requires_grad_(self.branch == "K")
        self.delta_epsilon_raw.requires_grad_(self.branch == "R")

    @torch.no_grad()
    def apply_identifiability_gate(self, min_gaussians=100, min_temperature_std=1.0,
                                   min_temperature_range=5.0):
        """Disable K and matched R parameters without enough material support."""
        rows = []
        for material_id, name in enumerate(self.material_names):
            selected = self.material_valid & (self.material_ids == material_id)
            count = int(selected.sum())
            values = self.temperature[selected, 0]
            std = float(values.std(unbiased=False)) if count else 0.0
            span = float(values.max() - values.min()) if count else 0.0
            identifiable = (count >= int(min_gaussians) and std >= float(min_temperature_std)
                            and span >= float(min_temperature_range))
            self.learn_k_by_material[material_id] &= identifiable
            # R keeps exactly the same enabled material parameter count as K.
            self.learn_delta_by_material[material_id] &= identifiable
            rows.append({"id": material_id, "name": name, "gaussians": count,
                         "temperature_std_K": std, "temperature_range_K": span,
                         "identifiable": bool(identifiable)})
        return rows

    def export_state(self):
        return {"format_version": 3, "branch": self.branch,
                "config": {"temp_min": self.temp_min, "temp_max": self.temp_max, "temp_ref": self.temp_ref,
                           "k_max": self.k_max, "delta_epsilon_max": self.delta_epsilon_max,
                           "material_names": self.material_names,
                           "unknown_epsilon0": self.unknown_epsilon0},
                "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint, branch, device, reset_branch_parameters=True):
        state = checkpoint["state_dict"]
        config = checkpoint["config"]
        epsilon0 = state.get("epsilon0_by_material")
        if epsilon0 is None:
            epsilon0 = torch.tensor([config["epsilon_nonmetal"], config["epsilon_metal"]])
        model = cls(
            state["material_ids"].to(device), torch.sigmoid(state["temperature_raw"]).to(device),
            epsilon0_by_material=epsilon0.to(device),
            material_names=config.get("material_names"),
            material_confidence=state.get("material_confidence"),
            learn_k_by_material=state.get("learn_k_by_material"),
            learn_delta_by_material=state.get("learn_delta_by_material"),
            k_prior_by_material=state.get("k_prior_by_material"),
            sigma_k_by_material=state.get("sigma_k_by_material"),
            sigma_delta_by_material=state.get("sigma_delta_by_material"),
            unknown_epsilon0=config.get("unknown_epsilon0", float(epsilon0[0])),
            branch=branch, initial_environment=float(torch.sigmoid(state["environment_raw"]).mean()),
            temp_min=config["temp_min"], temp_max=config["temp_max"], temp_ref=config["temp_ref"],
            k_max=config["k_max"], delta_epsilon_max=config["delta_epsilon_max"],
        ).to(device)
        compatible = {key: value.to(device) for key, value in state.items()
                      if key in model.state_dict() and model.state_dict()[key].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
        model.branch = branch
        if reset_branch_parameters:
            model.k_raw.data.zero_()
            model.delta_epsilon_raw.data.zero_()
        model.set_branch_trainability()
        return model


def gaussian_initial_radiance(gaussians):
    from utils.sh_utils import SH2RGB
    rgb = SH2RGB(gaussians.get_thermal_features_dc[:, 0, :]).clamp(0, 1)
    return (0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]).clamp(1e-4, 1 - 1e-4)


def _find_mask(mask_root, camera):
    if not mask_root:
        return None
    root, stem = Path(mask_root), Path(camera.image_name).stem
    candidates = [root / camera.image_name]
    candidates += [root / (stem + ext) for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]
    candidates += [root / split / (stem + ext) for split in ("train", "test") for ext in (".png", ".jpg", ".jpeg")]
    return next((path for path in candidates if path.exists()), None)


def _project_gaussians(gaussians, camera, raster_visibility=None,
                       occlusion_cell_size=1, depth_tolerance=0.02):
    xyz = gaussians.get_xyz.detach()
    xyz_h = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)
    clip = xyz_h @ camera.full_proj_transform
    valid = clip[:, 3] > 1e-6
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-6)
    valid &= (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
    if raster_visibility is not None:
        if raster_visibility.shape != valid.shape:
            raise ValueError("Rasterizer visibility must contain one flag per Gaussian")
        valid &= raster_visibility.to(device=valid.device, dtype=torch.bool)
    # Resolve competing projected centres at full image resolution.  The
    # rasterizer visibility above supplies the exact 3DGS frustum/radius test;
    # this z test prevents a back surface from receiving a foreground mask.
    if valid.any():
        cell = max(int(occlusion_cell_size), 1)
        grid_w = max((int(camera.image_width) + cell - 1) // cell, 1)
        grid_h = max((int(camera.image_height) + cell - 1) // cell, 1)
        pixel_x = (((ndc[:, 0] + 1.0) * 0.5 * camera.image_width).long() // cell).clamp(0, grid_w - 1)
        pixel_y = (((ndc[:, 1] + 1.0) * 0.5 * camera.image_height).long() // cell).clamp(0, grid_h - 1)
        cell_index = pixel_y * grid_w + pixel_x
        min_depth = torch.full((grid_w * grid_h,), float("inf"), device=ndc.device, dtype=ndc.dtype)
        if hasattr(min_depth, "scatter_reduce_"):
            min_depth.scatter_reduce_(0, cell_index[valid], ndc[valid, 2], reduce="amin", include_self=True)
        else:
            order = torch.argsort(cell_index[valid] * 2.0 + ndc[valid, 2])
            ordered_cells = cell_index[valid][order]
            ordered_depth = ndc[valid, 2][order]
            first = torch.ones_like(ordered_cells, dtype=torch.bool)
            first[1:] = ordered_cells[1:] != ordered_cells[:-1]
            min_depth[ordered_cells[first]] = ordered_depth[first]
        visible_depth = min_depth[cell_index]
        valid &= ndc[:, 2] <= visible_depth + float(depth_tolerance)
    return ndc, valid


def _visibility_and_projection_weight(gaussians, visibility_payload):
    """Normalize renderer visibility output and approximate per-view contribution."""
    count = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    if isinstance(visibility_payload, (tuple, list)):
        raster_visibility, radii = visibility_payload
    else:
        raster_visibility, radii = visibility_payload, None
    if raster_visibility is None:
        raster_visibility = torch.ones(count, dtype=torch.bool, device=device)
    raster_visibility = raster_visibility.to(device=device, dtype=torch.bool).reshape(-1)
    if raster_visibility.numel() != count:
        raise ValueError("Rasterizer visibility must contain one flag per Gaussian")
    if radii is None:
        radii = torch.ones(count, dtype=gaussians.get_xyz.dtype, device=device)
    radii = radii.to(device=device, dtype=gaussians.get_xyz.dtype).reshape(-1)
    if radii.numel() != count:
        raise ValueError("Rasterizer radii must contain one value per Gaussian")
    opacity = gaussians.get_opacity.detach().reshape(-1).to(radii)
    # Projected area times opacity is a deterministic proxy for the Gaussian's
    # image-space alpha contribution. A common scale cancels in the weighted mean.
    weight = opacity * radii.clamp_min(0.0).square()
    return raster_visibility, weight


@torch.no_grad()
def map_thermal_observations_to_gaussians(gaussians, cameras, observation_transform=None,
                                          visibility_provider=None, fallback_radiance=None):
    """Initialize Gaussian band radiance by multi-view projected thermal averaging."""
    device = gaussians.get_xyz.device
    total = torch.zeros(gaussians.get_xyz.shape[0], device=device)
    count = torch.zeros_like(total)
    for camera in cameras:
        image = camera.original_physical_image if camera.original_physical_image is not None else camera.original_image
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0).to(device)
        if observation_transform is not None:
            gray = observation_transform(gray)
        visibility_payload = visibility_provider(camera) if visibility_provider is not None else None
        raster_visibility, projection_weight = _visibility_and_projection_weight(
            gaussians, visibility_payload)
        ndc, valid = _project_gaussians(gaussians, camera, raster_visibility=raster_visibility)
        mask = getattr(camera, "thermal_valid_mask", None)
        if mask is not None:
            mask_sample = F.grid_sample(mask.to(device=device, dtype=torch.float32)[None], ndc[:, :2].reshape(1, -1, 1, 2),
                mode="bilinear", padding_mode="zeros", align_corners=False).reshape(-1)
            valid &= mask_sample >= 1 - 1e-6
        sampled = F.grid_sample(gray, ndc[:, :2].reshape(1, -1, 1, 2), mode="bilinear",
                                padding_mode="zeros", align_corners=False).reshape(-1)
        valid_weight = projection_weight[valid]
        total[valid] += sampled[valid] * valid_weight
        count[valid] += valid_weight
    fallback = (gaussian_initial_radiance(gaussians).reshape(-1) if fallback_radiance is None
                else total.new_full(total.shape, float(fallback_radiance)))
    # count is opacity * radius^2, not an integer observation count. Flooring
    # it at one biases low-support Gaussians cold. Only zero support needs a
    # protected denominator; every positive count uses its actual weight.
    supported = count > 0
    safe_count = torch.where(supported, count, torch.ones_like(count))
    observed = torch.where(supported, total / safe_count, fallback)
    print("[thermal-init] " + str({"weighted_mean_version": 2,
        "supported_fraction": float(supported.float().mean()),
        "below_one_weight_fraction": float((supported & (count < 1)).float().mean())}))
    return observed[:, None].clamp(1e-4, 1 - 1e-4)


@torch.no_grad()
def map_material_masks_to_gaussians(gaussians, cameras, mask_root, num_materials,
                                    unknown_label=255, confidence_threshold=0.6,
                                    margin_threshold=0.15, allow_missing=False,
                                    visibility_provider=None):
    """Fuse multi-class material masks into confidence-filtered Gaussian labels.

    Masks contain integer material ids in ``[0, num_materials)`` and
    ``unknown_label`` for ambiguous pixels. Unknown and unseen Gaussians remain
    unknown; they are deliberately not filled by nearest-neighbour propagation.
    """
    device = gaussians.get_xyz.device
    weight_sum = torch.zeros(gaussians.get_xyz.shape[0], device=device)
    class_votes = torch.zeros((int(num_materials), weight_sum.numel()), device=device)
    used = 0
    missing = []
    for camera in cameras:
        mask_path = _find_mask(mask_root, camera)
        if mask_path is None:
            missing.append(camera.image_name)
            continue
        mask_array = np.asarray(Image.open(mask_path))
        if mask_array.ndim != 2:
            raise ValueError(f"Material mask must be a single-channel label image: {mask_path}")
        invalid = ~((mask_array >= 0) & (mask_array < int(num_materials)) | (mask_array == int(unknown_label)))
        if invalid.any():
            values = np.unique(mask_array[invalid])[:8].tolist()
            raise ValueError(f"Material mask {mask_path} contains invalid labels: {values}")
        labels = torch.from_numpy(mask_array.astype(np.int64, copy=False)).to(device)
        planes = torch.stack([(labels == material_id).float() for material_id in range(int(num_materials))])
        confidence_path = mask_path.with_name(mask_path.stem + ".confidence.npy")
        if confidence_path.exists():
            pixel_confidence = np.load(confidence_path, allow_pickle=False)
            if pixel_confidence.shape != mask_array.shape or not np.isfinite(pixel_confidence).all():
                raise ValueError(f"Invalid material confidence map: {confidence_path}")
            confidence_image = torch.from_numpy(pixel_confidence.astype(np.float32, copy=False)).to(device)
            planes = planes * confidence_image.clamp(0.0, 1.0)[None]
        visibility_payload = visibility_provider(camera) if visibility_provider is not None else None
        raster_visibility, projection_weight = _visibility_and_projection_weight(
            gaussians, visibility_payload)
        ndc, valid = _project_gaussians(gaussians, camera, raster_visibility=raster_visibility)
        sampled = F.grid_sample(planes[None], ndc[:, :2].reshape(1, -1, 1, 2), mode="bilinear",
                                padding_mode="zeros", align_corners=False).reshape(int(num_materials), -1)
        class_votes[:, valid] += sampled[:, valid] * projection_weight[valid][None]
        # Keep unknown/low-confidence support in the denominator so SAM score
        # and cross-view agreement both affect the final confidence.
        weight_sum[valid] += projection_weight[valid]
        used += 1
    if missing and not allow_missing:
        preview = ", ".join(str(name) for name in missing[:8])
        raise RuntimeError(
            f"Missing material masks for {len(missing)}/{len(cameras)} training views "
            f"({preview}). Formal multi-view voting requires one label mask per view."
        )
    if used == 0:
        if not allow_missing:
            raise RuntimeError("No material masks were found")
        print("[material][SMOKE-TEST ONLY] No masks found; every Gaussian is unknown.")
        count = gaussians.get_xyz.shape[0]
        return (torch.full((count,), -1, dtype=torch.long, device=device),
                torch.zeros(count, device=device))
    if missing:
        print(f"[material][SMOKE-TEST ONLY] Ignored {len(missing)} views without masks.")
    observed = weight_sum > 0
    if not observed.any():
        raise RuntimeError("Material masks were loaded, but no Gaussian received a projected mask sample")
    if float(class_votes.sum()) <= 0.0 and not allow_missing:
        raise RuntimeError("Material masks contain no known-material support on visible Gaussians")
    probabilities = class_votes / weight_sum.clamp_min(1e-12)[None]
    confidence, winner = probabilities.max(dim=0)
    if int(num_materials) > 1:
        top2 = probabilities.topk(k=2, dim=0).values
        margin = top2[0] - top2[1]
    else:
        margin = confidence
    accepted = (observed & (confidence >= float(confidence_threshold)) &
                (margin >= float(margin_threshold)))
    assignment = torch.full_like(winner, -1)
    assignment[accepted] = winner[accepted]
    counts = {idx: int((assignment == idx).sum()) for idx in range(int(num_materials))}
    print(f"[material] Used {used} masks; accepted={int(accepted.sum())}/{assignment.numel()}, "
          f"unknown={int((~accepted).sum())}, counts={counts}")
    return assignment, confidence


def map_metal_masks_to_gaussians(*args, **kwargs):
    raise RuntimeError(
        "Binary metal/non-metal mapping was removed. Use map_material_masks_to_gaussians "
        "with multi-class label masks and a material configuration file."
    )


def frozen_geometry_state(gaussians):
    """Create the self-contained frozen-geometry part of Theta^(2)."""
    names = ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest")
    state = {}
    for name in names:
        value = getattr(gaussians, name)
        state[name] = value.detach().cpu().clone()
    state["active_sh_degree"] = int(gaussians.active_sh_degree)
    return state


def verify_frozen_geometry_state(gaussians, state):
    if not state:
        raise ValueError("Stage-2 common checkpoint does not contain frozen_geometry")
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest"):
        expected = state.get(name)
        actual = getattr(gaussians, name).detach().cpu()
        if expected is None or expected.shape != actual.shape or not torch.equal(expected.cpu(), actual):
            raise ValueError(f"Embedded frozen geometry differs from the loaded stage-1 tensor: {name}")


def save_thermal_checkpoint(path, model, step, metadata=None, frozen_geometry=None, ir_opacity=None, sh_residual=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = model.export_state()
    payload.update(step=int(step), metadata=metadata or {})
    if frozen_geometry is not None:
        payload["frozen_geometry"] = frozen_geometry
    if ir_opacity is not None:
        payload["ir_opacity"] = ir_opacity.export_state()
    if sh_residual is not None:
        payload["sh_residual"] = sh_residual.export_state()
    torch.save(payload, path)


@torch.no_grad()
def build_spatial_tv_edges(xyz, neighbors=6):
    """Build a full Euclidean k-NN graph for 3D temperature TV."""
    points = xyz.detach().cpu().numpy()
    count = points.shape[0]
    if count < 2:
        empty = torch.empty(0, dtype=torch.long, device=xyz.device)
        return empty, empty, torch.empty(0, dtype=xyz.dtype, device=xyz.device)
    k = min(int(neighbors) + 1, count)
    distances, indices = cKDTree(points).query(points, k=k)
    if k == 2 and distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    source = np.repeat(np.arange(count, dtype=np.int64), k - 1)
    target = indices[:, 1:].reshape(-1).astype(np.int64)
    distance = distances[:, 1:].reshape(-1).astype(np.float32)
    sigma = max(float(np.median(distance)), 1e-8)
    weights = np.exp(-np.square(distance / sigma)).astype(np.float32)
    return (torch.from_numpy(source).to(xyz.device), torch.from_numpy(target).to(xyz.device),
            torch.from_numpy(weights).to(xyz.device))
