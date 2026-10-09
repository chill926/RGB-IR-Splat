"""Train frozen-geometry thermal fields, bounded G-alpha adaptation, and C/K/R."""
import argparse
import hashlib
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.thermal_opacity import IROpacityCorrection, render_opacity
from utils.thermal_sh_residual import ThermalSHResidual, thermal_view_radiance
from scene import GaussianModel, Scene
from utils.thermal_training_utils import (held_lr, install_loss_scale, thermal_data_loss,
    PlateauTracker, quality_status, masked_mean)
from utils.thermal_observations import radiometric_image_errors
from utils.thermal_observations import prepare_thermal_input
from utils.thermal_physics import (MaterialThermalField,
    build_spatial_tv_edges, frozen_geometry_state, map_material_masks_to_gaussians,
    map_thermal_observations_to_gaussians, save_thermal_checkpoint,
    verify_frozen_geometry_state)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, pipeline = ModelParams(parser, device), PipelineParams(parser)
    parser.add_argument("--stage", choices=("stage2", "galpha", "branch"), required=True)
    parser.add_argument("--branch", choices=("C", "K", "R"), default="C")
    parser.add_argument("--geometry_model", required=True)
    parser.add_argument("--geometry_iteration", type=int, default=-1)
    parser.add_argument("--stage2_checkpoint", default="")
    parser.add_argument("--sh_residual_degree", type=int, choices=(0, 2), default=0,
                        help="Fresh Stage2: 0 retains the baseline, 2 adds eight scalar non-DC SH coefficients")
    parser.add_argument("--sh_residual_start_step", type=int, default=3000)
    parser.add_argument("--sh_residual_bound", type=float, default=0.5,
                        help="Maximum signed SH signal correction as a fraction of the fit loss scale")
    parser.add_argument("--sh_residual_lr", type=float, default=0.01)
    parser.add_argument("--sh_residual_lr_final", type=float, default=0.0001)
    parser.add_argument("--lambda_sh_residual", type=float, default=0.001)
    parser.add_argument("--ir_opacity_mode", choices=("bounded", "frozen"), default="bounded",
                        help="G-alpha only: bounded IR correction or matched frozen-opacity control")
    parser.add_argument("--ir_opacity_logit_bound", type=float, default=0.2)
    parser.add_argument("--ir_opacity_lr", type=float, default=1e-3)
    parser.add_argument("--ir_opacity_lr_final", type=float, default=1e-5)
    parser.add_argument("--lambda_ir_opacity", type=float, default=0.01)
    parser.add_argument("--ir_opacity_stability", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--temperature_lr", type=float, default=1e-3)
    parser.add_argument("--temperature_lr_final", type=float, default=1e-5,
                        help="T LR is exponentially decayed from --temperature_lr to this value over --steps")
    parser.add_argument("--environment_lr", type=float, default=1e-4)
    parser.add_argument("--environment_lr_final", type=float, default=1e-6,
                        help="Environment LR is exponentially decayed from --environment_lr to this value over --steps; use equal values for a fixed LR")
    parser.add_argument("--material_lr", type=float, default=1e-5)
    parser.add_argument("--lambda_tv", type=float, default=1e-2)
    parser.add_argument("--loss_scale_mode", choices=("auto", "legacy", "fit_quantile"), default="auto")
    parser.add_argument("--loss_scale_floor", type=float, default=1e-4)
    parser.add_argument("--tv_temperature_scale_K", type=float, default=10.0)
    parser.add_argument("--tv_material_boundary_weight", type=float, default=0.1)
    parser.add_argument("--lr_hold_fraction", type=float, default=0.5)
    parser.add_argument("--gradient_log_every", type=int, default=500)
    parser.add_argument("--tv_neighbors", type=int, default=6)
    parser.add_argument("--lambda_environment", type=float, default=0.01)
    parser.add_argument("--environment_prior_beta", type=float, default=0.02)
    parser.add_argument("--environment_init_quantile", type=float, default=0.1)
    parser.add_argument("--lambda_k", type=float, default=None)
    parser.add_argument("--lambda_delta_epsilon", type=float, default=None)
    parser.add_argument("--regularization_protocol", default="",
                        help="regularization_scan.json selected on synthetic validation; required for formal K/R")
    parser.add_argument("--comparison_protocol", default="",
                        help="shared C/K/R protocol lock file; required for formal branch training")
    parser.add_argument("--regularization_scan_run", action="store_true", default=False,
                        help="Only for tools/scan_material_regularization.py, never for a reported test run")
    parser.add_argument("--huber_delta", type=float, default=0.02)
    parser.add_argument("--temp_min", type=float, default=250.0)
    parser.add_argument("--temp_max", type=float, default=450.0)
    parser.add_argument("--temp_ref", type=float, default=300.0)
    parser.add_argument("--material_config", default="",
                        help="Confirmed material names, fixed epsilon_0 values and priors (JSON)")
    parser.add_argument("--material_mask_dir", "--metal_mask_dir", dest="material_mask_dir", default="")
    parser.add_argument("--material_confidence_threshold", type=float, default=0.6)
    parser.add_argument("--material_margin_threshold", type=float, default=0.15)
    parser.add_argument("--allow_missing_material_masks", "--allow_missing_metal_masks",
                        dest="allow_missing_material_masks", action="store_true", default=False)
    parser.add_argument("--min_material_gaussians", type=int, default=100)
    parser.add_argument("--min_material_temperature_std", type=float, default=1.0)
    parser.add_argument("--min_material_temperature_range", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--stage2_patience", type=int, default=10,
                        help="Plateau mode only: validation checks without meaningful improvement before stopping")
    parser.add_argument("--stage2_min_delta", type=float, default=1e-8)
    parser.add_argument("--stage2_relative_min_delta", type=float, default=0.002)
    parser.add_argument("--stage2_stop_mode", choices=("budget", "plateau"), default="budget")
    parser.add_argument("--stage2_min_steps", type=int, default=10000)
    parser.add_argument("--stage2_temperature_stability_p95_K", type=float, default=0.05)
    parser.add_argument("--quality_apparent_mae_K", type=float, default=None)
    parser.add_argument("--quality_signal_rmse_Q", type=float, default=None)
    parser.add_argument("--stage2_temperature_stability_K", type=float, default=0.01)
    parser.add_argument("--stage2_environment_stability", type=float, default=1e-5)
    parser.add_argument("--stage3_material_stability", type=float, default=1e-6)
    parser.add_argument("--validation_fraction", type=float, default=0.1,
                        help="Fraction of the published training split reserved for validation")
    parser.add_argument("--validation_seed", type=int, default=2027)
    parser.add_argument("--observation_domain",
                        choices=("normalized_dn", "calibrated_radiance", "apparent_temperature", "raw_rjpeg"),
                        default="normalized_dn")
    parser.add_argument("--radiometric_dir", default="",
                        help="Prepared RJPEG signal directory; default SCENE/radiometric. "
                             "Run tools/prepare_rgbt_radiometry.py first; used only in raw_rjpeg mode")
    parser.add_argument("--dn0_radiance", type=float, default=0.0,
                        help="Physical band radiance corresponding to normalized DN=0")
    parser.add_argument("--dn1_radiance", type=float, default=1.0,
                        help="Physical band radiance corresponding to normalized DN=1")
    parser.add_argument("--thermal_raw_max", type=float, default=65535.0,
                        help="Raw integer value represented by normalized thermal value 1")
    parser.add_argument("--temperature_scale", type=float, default=0.01,
                        help="Kelvin per raw thermal unit in apparent_temperature mode")
    parser.add_argument("--temperature_offset", type=float, default=0.0,
                        help="Kelvin offset in apparent_temperature mode")
    parser.add_argument("--temperature_gt", default="",
                        help="Optional per-Gaussian synthetic temperature truth (.npy or .pt, Kelvin)")
    parser.add_argument("--lambda_temperature_supervision", type=float, default=0.0)
    parser.add_argument("--oracle_temperature_supervision", action="store_true", default=False,
                        help="Required acknowledgement for non-main experiments that train with temperature truth")
    args = parser.parse_args()
    args.geometry_model, args.model_path = os.path.abspath(args.geometry_model), os.path.abspath(args.model_path)
    args.source_path = os.path.abspath(args.source_path)
    if args.material_mask_dir:
        args.material_mask_dir = os.path.abspath(args.material_mask_dir)
    if args.material_config:
        args.material_config = os.path.abspath(args.material_config)
    if args.temperature_gt:
        args.temperature_gt = os.path.abspath(args.temperature_gt)
    args.data_branch = "rgbt"
    return args, model.extract(args), pipeline.extract(args), device


def observation_to_model_radiance(image, args, planck):
    if args.observation_domain == "raw_rjpeg":
        # prepare_thermal_input has already converted Q=DN+O with the same
        # fixed normalization used by the camera-response LUT.
        if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
            raise ValueError("Invalid prepared RJPEG target")
        return image
    if args.observation_domain == "normalized_dn":
        return image.clamp(0.0, 1.0)
    if args.observation_domain == "apparent_temperature":
        temperature = image * args.thermal_raw_max * args.temperature_scale + args.temperature_offset
        if float(temperature.min()) < args.temp_min or float(temperature.max()) > args.temp_max:
            raise ValueError(
                "Converted apparent temperature lies outside the Planck LUT bounds; "
                "check --thermal_raw_max/--temperature_scale/--temperature_offset"
            )
        return planck(temperature)
    physical = args.dn0_radiance + image * (args.dn1_radiance - args.dn0_radiance)
    return planck.normalize_physical_radiance(physical)


def load_material_config(path):
    if not path:
        raise ValueError("Stage 2 requires --material_config; binary metal/non-metal defaults were removed")
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    materials = sorted(document.get("materials", []), key=lambda row: int(row["id"]))
    if [int(row["id"]) for row in materials] != list(range(len(materials))):
        raise ValueError("Material ids must be contiguous integers starting at zero")
    if not materials:
        raise ValueError("Material configuration contains no materials")
    unconfirmed = [row.get("name", row["id"]) for row in materials if not row.get("confirmed", False)]
    if unconfirmed:
        raise ValueError(
            "Every epsilon_0 must be manually confirmed before training; unconfirmed materials: "
            + ", ".join(map(str, unconfirmed))
        )
    names = [str(row["name"]) for row in materials]
    if len(set(names)) != len(names):
        raise ValueError("Material names must be unique")
    epsilon0 = [float(row["epsilon0"]) for row in materials]
    if not all(0.01 <= value <= 0.99 for value in epsilon0):
        raise ValueError("Every material epsilon0 must lie in [0.01, 0.99]")
    sigma_k = [float(row.get("sigma_k", 1e-4)) for row in materials]
    sigma_delta = [float(row.get("sigma_delta", 0.05)) for row in materials]
    if min(sigma_k + sigma_delta) <= 0.0:
        raise ValueError("Material prior standard deviations must be positive")
    unknown_epsilon0 = float(document.get("unknown_epsilon0", 0.95))
    unknown_label = int(document.get("unknown_label", 255))
    if not 0.01 <= unknown_epsilon0 <= 0.99 or not 0 <= unknown_label <= 255:
        raise ValueError("Invalid unknown material emissivity or label")
    learn_k = [bool(row.get("learn_k", True)) for row in materials]
    learn_delta = [bool(row.get("learn_delta", row.get("learn_k", True))) for row in materials]
    if learn_delta != learn_k:
        raise ValueError("K and R must enable the same materials to keep equal parameter counts")
    return {
        "document": document,
        "temperature_reference_K": float(document.get("temperature_reference_K", 300.0)),
        "names": names,
        "epsilon0": epsilon0,
        "learn_k": learn_k,
        "learn_delta": learn_delta,
        "k_prior": [float(row.get("k_prior", 0.0)) for row in materials],
        "sigma_k": sigma_k,
        "sigma_delta": sigma_delta,
        "unknown_epsilon0": unknown_epsilon0,
        "unknown_label": unknown_label,
    }


def split_fit_validation_cameras(cameras, fraction, seed):
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("--validation_fraction must lie strictly between 0 and 0.5")
    if len(cameras) < 3:
        raise RuntimeError("At least three published training cameras are needed for an internal validation split")
    count = max(1, min(len(cameras) - 1, int(round(len(cameras) * float(fraction)))))
    order = list(range(len(cameras)))
    random.Random(int(seed)).shuffle(order)
    validation_index = set(order[:count])
    fit = [camera for idx, camera in enumerate(cameras) if idx not in validation_index]
    validation = [camera for idx, camera in enumerate(cameras) if idx in validation_index]
    return fit, validation


def apply_stage2_protocol(args):
    if args.stage not in ("branch", "galpha"):
        return
    if not args.stage2_checkpoint:
        raise ValueError("--stage2_checkpoint is required for branch training")
    checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    args.material_config_document = metadata.get("material_config_document")
    args.material_config_sha256 = metadata.get("material_config_sha256", "")
    shared = metadata.get("shared_training_protocol", {})
    args.radiometric_protocol = shared.get("radiometric_protocol")
    # Preserve old checkpoints' loss/TV units rather than silently changing C/R/K.
    for name, fallback in (("loss_scale_mode", "legacy"), ("loss_scale_floor", 1e-4),
                           ("loss_scale_protocol", None), ("tv_temperature_scale_K", 1.0),
                           ("tv_material_boundary_weight", 1.0), ("lr_hold_fraction", 0.0)):
        setattr(args, name, shared.get(name, fallback))
    if shared.get("observation_domain") == "raw_rjpeg" and not isinstance(args.radiometric_protocol, dict):
        raise ValueError("RJPEG checkpoint is missing its radiometric calibration protocol")
    if not args.radiometric_dir:
        args.radiometric_dir = shared.get("radiometric_dir", "")
    for name in ("temperature_lr", "temperature_lr_final", "environment_lr", "environment_lr_final", "lambda_tv",
                 "lambda_environment",
                 "environment_prior_beta", "environment_init_quantile", "huber_delta",
                 "temp_min", "temp_max", "temp_ref", "observation_domain",
                 "dn0_radiance", "dn1_radiance", "temperature_gt", "lambda_temperature_supervision",
                 "oracle_temperature_supervision",
                 "thermal_raw_max", "temperature_scale", "temperature_offset",
                 "tv_neighbors", "validation_fraction", "validation_seed",
                 "min_material_gaussians", "min_material_temperature_std",
                 "min_material_temperature_range"):
        # G-alpha permits a deliberate TV/T-LR adaptation; C/R/K inherit it afterwards.
        if name in shared and not (args.stage == "galpha" and name in
                                   ("lambda_tv", "temperature_lr", "temperature_lr_final")):
            setattr(args, name, shared[name])
    if "environment_lr_final" not in shared:
        # Old stage-2 checkpoints used a fixed environment LR. Preserve that
        # protocol when initializing C/K/R from an existing common endpoint.
        args.environment_lr_final = args.environment_lr


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_regularization_protocol(args):
    if args.stage != "branch":
        return
    if args.regularization_scan_run:
        print("[protocol][SCAN ONLY] regularization is being selected on synthetic validation.")
        return
    if not args.regularization_protocol:
        if args.branch == "K" and args.lambda_k is None:
            raise ValueError("Branch K needs either --lambda_k or --regularization_protocol")
        if args.branch == "R" and args.lambda_delta_epsilon is None:
            raise ValueError("Branch R needs either --lambda_delta_epsilon or --regularization_protocol")
        print("[protocol][EXPLORATORY RGBT] Regularization was supplied directly, not frozen from synthetic validation.")
        return
    with open(args.regularization_protocol, encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("selection_dataset_role") != "synthetic_validation":
        raise ValueError("Regularization protocol is not marked as selected on synthetic validation")
    if args.branch in ("K", "R"):
        selected = report.get("selected", {}).get(args.branch)
        if not selected or "regularization" not in selected:
            raise ValueError(f"Regularization protocol has no selected value for branch {args.branch}")
        value = float(selected["regularization"])
        if args.branch == "K":
            args.lambda_k = value
        else:
            args.lambda_delta_epsilon = value
    args.regularization_protocol = os.path.abspath(args.regularization_protocol)
    args.regularization_protocol_sha256 = file_sha256(args.regularization_protocol)


def enforce_comparison_protocol(args):
    if args.stage != "branch" or args.regularization_scan_run:
        return
    if not args.comparison_protocol:
        raise ValueError("Formal C/K/R training requires the same --comparison_protocol path")
    path = os.path.abspath(args.comparison_protocol)
    protocol = {
        "format_version": 1,
        "stage2_checkpoint_sha256": file_sha256(args.stage2_checkpoint),
        "regularization_protocol_sha256": getattr(args, "regularization_protocol_sha256", "exploratory-direct"),
        "geometry_sha256": args.geometry_sha256,
        "material_config_sha256": getattr(args, "material_config_sha256", ""),
        "steps": args.steps,
        "seed": args.seed,
        "temperature_lr": args.temperature_lr,
        "temperature_lr_final": args.temperature_lr_final,
        "environment_lr": args.environment_lr,
        "environment_lr_final": args.environment_lr_final,
        "material_lr": args.material_lr,
        "lambda_tv": args.lambda_tv,
        "lambda_environment": args.lambda_environment,
        "environment_prior_beta": args.environment_prior_beta,
        "environment_init_quantile": args.environment_init_quantile,
        "huber_delta": args.huber_delta,
        "loss_scale_protocol": args.loss_scale_protocol,
        "tv_temperature_scale_K": args.tv_temperature_scale_K,
        "tv_material_boundary_weight": args.tv_material_boundary_weight,
        "lr_hold_fraction": args.lr_hold_fraction,
        "eval_every": args.eval_every,
        "stage3_material_stability": args.stage3_material_stability,
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "min_material_gaussians": args.min_material_gaussians,
        "min_material_temperature_std": args.min_material_temperature_std,
        "min_material_temperature_range": args.min_material_temperature_range,
    }
    if args.observation_domain == "raw_rjpeg":
        protocol["radiometric_protocol"] = args.radiometric_protocol
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        differences = {key: (existing.get(key), value) for key, value in protocol.items()
                       if existing.get(key) != value}
        if differences:
            raise ValueError(f"C/K/R comparison protocol mismatch: {differences}")
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, sort_keys=True)
    args.comparison_protocol = path
    args.comparison_protocol_sha256 = file_sha256(path)


def checkpoint_metadata(args, scene, **extra):
    metadata = {
        "geometry_model": args.geometry_model,
        "geometry_iteration": scene.loaded_iter,
        "geometry_sha256": args.geometry_sha256,
        "source_path": os.path.abspath(args.source_path),
        "seed": args.seed,
        "material_mask_dir": args.material_mask_dir,
        "training_stage": args.stage,
        "initialization_weighted_mean_version": 2 if args.stage == "stage2" else None,
        "ir_opacity_protocol": getattr(args, "ir_opacity_protocol", {"enabled": False}),
        "sh_residual_protocol": getattr(args, "sh_residual_protocol", {"enabled": False}),
        "warm_start_checkpoint_sha256": getattr(args, "warm_start_checkpoint_sha256", None),
        "shared_training_protocol": {
            "temperature_lr": args.temperature_lr,
            "temperature_lr_final": args.temperature_lr_final,
            "environment_lr": args.environment_lr,
            "environment_lr_final": args.environment_lr_final,
            "lambda_tv": args.lambda_tv, "lambda_environment": args.lambda_environment,
            "environment_prior_beta": args.environment_prior_beta,
            "environment_init_quantile": args.environment_init_quantile,
            "huber_delta": args.huber_delta, "temp_min": args.temp_min, "temp_max": args.temp_max,
            "loss_scale_mode": args.loss_scale_mode, "loss_scale_floor": args.loss_scale_floor,
            "loss_scale_protocol": args.loss_scale_protocol,
            "tv_temperature_scale_K": args.tv_temperature_scale_K,
            "tv_material_boundary_weight": args.tv_material_boundary_weight,
            "lr_hold_fraction": args.lr_hold_fraction,
            "temp_ref": args.temp_ref, "observation_domain": args.observation_domain,
            "dn0_radiance": args.dn0_radiance, "dn1_radiance": args.dn1_radiance,
            "thermal_raw_max": args.thermal_raw_max,
            "temperature_scale": args.temperature_scale,
            "temperature_offset": args.temperature_offset,
            "temperature_gt": args.temperature_gt,
            "lambda_temperature_supervision": args.lambda_temperature_supervision,
            "oracle_temperature_supervision": args.oracle_temperature_supervision,
            "tv_neighbors": args.tv_neighbors,
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
            "resolution": args.resolution,
            "min_material_gaussians": args.min_material_gaussians,
            "min_material_temperature_std": args.min_material_temperature_std,
            "min_material_temperature_range": args.min_material_temperature_range,
        },
    }
    if args.observation_domain == "raw_rjpeg":
        metadata["shared_training_protocol"].update({
            "radiometric_dir": args.radiometric_dir,
            "radiometric_protocol": args.radiometric_protocol,
        })
    if getattr(args, "material_config", ""):
        metadata["material_config"] = args.material_config
        metadata["material_config_sha256"] = file_sha256(args.material_config)
    if getattr(args, "material_config_sha256", ""):
        metadata["material_config_sha256"] = args.material_config_sha256
    if getattr(args, "material_config_document", None) is not None:
        metadata["material_config_document"] = args.material_config_document
    if hasattr(args, "fit_camera_names"):
        metadata["camera_split"] = {
            "fit": args.fit_camera_names,
            "validation": args.validation_camera_names,
            "test": args.test_camera_names,
        }
    if getattr(args, "comparison_protocol", ""):
        metadata["comparison_protocol"] = args.comparison_protocol
        metadata["comparison_protocol_sha256"] = getattr(args, "comparison_protocol_sha256", "")
    if getattr(args, "regularization_protocol", ""):
        metadata["regularization_protocol"] = args.regularization_protocol
        metadata["regularization_protocol_sha256"] = getattr(args, "regularization_protocol_sha256", "")
    metadata.update(extra)
    return metadata


def geometry_checkpoint_sha256(model_path, iteration):
    path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_temperature_truth(path, count, device):
    if not path:
        return None
    if path.endswith(".npy"):
        value = torch.from_numpy(np.load(path))
    else:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            value = value.get("temperature", value.get("temperature_K"))
    if value is None:
        raise ValueError("Temperature truth file must contain 'temperature' or 'temperature_K'")
    value = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1, 1)
    if value.shape[0] != count:
        raise ValueError(f"Temperature truth has {value.shape[0]} values, expected {count}")
    return value


def make_field(args, scene, device, planck, visibility_provider, fit_cameras, material_config):
    if args.stage in ("branch", "galpha"):
        if not args.stage2_checkpoint:
            raise ValueError("--stage2_checkpoint is required for branch training")
        checkpoint = torch.load(args.stage2_checkpoint, map_location=device)
        if checkpoint.get("branch") != "stage2":
            raise ValueError("G-alpha and C/K/R must start from a stage2 checkpoint")
        metadata = checkpoint.get("metadata", {})
        if not metadata.get("stage2_common_endpoint", False):
            raise ValueError("Warm starts require thermal_stage2_common.pt, not a best/intermediate checkpoint")
        expected_geometry = os.path.abspath(metadata.get("geometry_model", args.geometry_model))
        if expected_geometry != args.geometry_model and args.stage != "galpha":
            raise ValueError(f"Geometry mismatch: stage2 used {expected_geometry}, branch requested {args.geometry_model}")
        expected_iteration = metadata.get("geometry_iteration")
        if expected_iteration is not None and int(expected_iteration) != int(scene.loaded_iter):
            raise ValueError(f"Geometry iteration mismatch: stage2 used {expected_iteration}, loaded {scene.loaded_iter}")
        expected_hash = metadata.get("geometry_sha256")
        if expected_hash and expected_hash != args.geometry_sha256:
            raise ValueError("Geometry checkpoint content hash differs from the stage2 common checkpoint")
        expected_source = os.path.abspath(metadata.get("source_path", args.source_path))
        if expected_source != os.path.abspath(args.source_path) and args.stage != "galpha":
            raise ValueError(f"Dataset mismatch: stage2 used {expected_source}, branch requested {args.source_path}")
        expected_split = {"fit": args.fit_camera_names, "validation": args.validation_camera_names,
                          "test": args.test_camera_names}
        if metadata.get("camera_split") != expected_split:
            raise ValueError("Warm-start camera split differs from the recorded reference")
        verify_frozen_geometry_state(scene.gaussians, checkpoint.get("frozen_geometry"))
        args.reference_ir_opacity_state = checkpoint.get("ir_opacity")
        args.reference_sh_residual_state = checkpoint.get("sh_residual")
        args.material_mask_dir = metadata.get("material_mask_dir", args.material_mask_dir)
        return MaterialThermalField.from_checkpoint(checkpoint,
            "stage2" if args.stage == "galpha" else args.branch, device)
    labels, material_confidence = map_material_masks_to_gaussians(
        scene.gaussians, fit_cameras, args.material_mask_dir,
        num_materials=len(material_config["names"]),
        unknown_label=material_config["unknown_label"],
        confidence_threshold=args.material_confidence_threshold,
        margin_threshold=args.material_margin_threshold,
        allow_missing=args.allow_missing_material_masks,
        visibility_provider=visibility_provider,
    )
    observed = map_thermal_observations_to_gaussians(
        scene.gaussians, fit_cameras,
        observation_transform=lambda image: observation_to_model_radiance(image, args, planck),
        visibility_provider=visibility_provider,
        # RGB SH values have no calibrated thermal meaning. Use a thermal
        # reference-temperature prior only for Gaussians unseen in fit views.
        fallback_radiance=(float(planck(torch.tensor(args.temp_ref, device=device)))
                           if args.observation_domain == "raw_rjpeg" else None),
    )
    epsilon_table = observed.new_tensor(material_config["epsilon0"])
    safe_labels = labels.clamp_min(0)
    epsilon0 = epsilon_table[safe_labels, None]
    epsilon0 = torch.where(labels[:, None] >= 0, epsilon0,
                           observed.new_full(epsilon0.shape, material_config["unknown_epsilon0"]))
    initial_environment = float(torch.quantile(observed, args.environment_init_quantile))
    initial_emission = ((observed - (1.0 - epsilon0) * initial_environment) /
                        epsilon0.clamp_min(1e-4)).clamp(1e-4, 1 - 1e-4)
    initial_temperature = planck.inverse(initial_emission)
    initial_temperature_unit = (initial_temperature - args.temp_min) / (args.temp_max - args.temp_min)
    return MaterialThermalField(labels, initial_temperature_unit,
        epsilon0_by_material=material_config["epsilon0"],
        material_names=material_config["names"], material_confidence=material_confidence,
        learn_k_by_material=material_config["learn_k"],
        learn_delta_by_material=material_config["learn_delta"],
        k_prior_by_material=material_config["k_prior"],
        sigma_k_by_material=material_config["sigma_k"],
        sigma_delta_by_material=material_config["sigma_delta"],
        unknown_epsilon0=material_config["unknown_epsilon0"], branch="stage2",
        temp_min=args.temp_min, temp_max=args.temp_max, temp_ref=args.temp_ref,
        initial_environment=initial_environment).to(device)


@torch.no_grad()
def validation_metrics(field, planck, cameras, gaussians, pipe, background, dataset, args, opacity_field=None, sh_field=None):
    if not cameras:
        raise ValueError("Validation cameras are required")
    physical = field.radiance(planck)
    rows, mse_values = [], []
    for camera in cameras:
        radiance, delta, clipped = thermal_view_radiance(physical, gaussians.get_xyz, camera.camera_center, sh_field)
        prediction = render(camera, gaussians, pipe, background, 0.0, 0.0, 0.0, device=background.device,
            is_6dof=dataset.is_6dof, override_color=radiance.repeat(1, 3), detach_geometry=True,
            override_opacity=render_opacity(opacity_field))["render"][:1]
        target = camera.original_physical_image if camera.original_physical_image is not None else camera.original_image
        target = observation_to_model_radiance(target.mean(dim=0, keepdim=True), args, planck)
        mask = getattr(camera, "thermal_valid_mask", None)
        error = prediction - target
        mse = float(error.square().mean() if mask is None else masked_mean(error.square(), mask))
        huber = F.smooth_l1_loss(prediction, target, beta=args.huber_delta, reduction="none")
        row = {"radiance_loss": float(thermal_data_loss(prediction, target, args, mask)),
               "normalized_signal_huber_loss": float(huber.mean() if mask is None else masked_mean(huber, mask)),
               "signal_MAE": float(error.abs().mean() if mask is None else masked_mean(error.abs(), mask)), "signal_MSE": mse,
               "signal_PSNR": -10 * math.log10(max(mse, 1e-12))}
        if args.observation_domain == "raw_rjpeg":
            row.update(radiometric_image_errors(prediction, target, planck, mask))
        if sh_field is not None:
            row.update(sh_abs_delta_mean=float(delta.abs().mean()), sh_clipped_fraction=float(clipped))
        rows.append(row)
        mse_values.append(mse)
    result = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    result["signal_RMSE"] = math.sqrt(sum(mse_values) / len(mse_values))
    if args.observation_domain == "raw_rjpeg":
        span = float(planck.physical_radiance_max - planck.physical_radiance_min)
        result["camera_signal_RMSE"] = result["signal_RMSE"] * span
    result["views"] = len(rows)
    return result


def main():
    args, dataset, pipe, device = parse_args()
    if args.stage == "stage2" and args.stage2_checkpoint:
        raise ValueError("Fresh Stage2 must not use --stage2_checkpoint; supply only Stage1 geometry")
    if args.sh_residual_degree and args.stage != "stage2":
        raise ValueError("New SH residuals are enabled only in fresh Stage2; C/R/K/G-alpha inherit them frozen")
    if (not math.isfinite(args.sh_residual_bound) or args.sh_residual_bound <= 0 or
            not all(math.isfinite(x) for x in (args.sh_residual_lr, args.sh_residual_lr_final, args.lambda_sh_residual)) or
            not 0 < args.sh_residual_lr_final <= args.sh_residual_lr or args.lambda_sh_residual < 0):
        raise ValueError("Invalid SH residual bound, learning rates, or regularization")
    if args.sh_residual_degree and not 1 <= args.sh_residual_start_step <= args.steps:
        raise ValueError("SH start step must lie inside the training budget")
    if args.sh_residual_degree and args.stage2_stop_mode != "budget":
        raise ValueError("SH experiments use fixed budget; coefficient convergence is not covered by the old plateau rule")
    apply_stage2_protocol(args)
    args.warm_start_checkpoint_sha256 = file_sha256(args.stage2_checkpoint) if args.stage2_checkpoint else None
    apply_regularization_protocol(args)
    if args.steps <= 0 or args.save_every <= 0 or args.eval_every <= 0:
        raise ValueError("--steps, --save_every and --eval_every must all be positive")
    if not args.temp_min < args.temp_ref < args.temp_max:
        raise ValueError("Temperature bounds must satisfy temp_min < temp_ref < temp_max")
    if not 0.0 <= args.environment_init_quantile <= 1.0:
        raise ValueError("--environment_init_quantile must lie in [0, 1]")
    if min(args.temperature_lr, args.temperature_lr_final, args.environment_lr, args.environment_lr_final, args.material_lr,
           args.huber_delta, args.environment_prior_beta) <= 0.0:
        raise ValueError("Learning rates and Huber beta values must be positive")
    if args.temperature_lr_final > args.temperature_lr:
        raise ValueError("--temperature_lr_final must not exceed --temperature_lr")
    if args.environment_lr_final > args.environment_lr:
        raise ValueError("--environment_lr_final must not exceed --environment_lr")
    if min(args.lambda_tv, args.lambda_environment, args.lambda_temperature_supervision) < 0.0:
        raise ValueError("Loss weights must be non-negative")
    if (args.temperature_gt or args.lambda_temperature_supervision > 0.0) and not args.oracle_temperature_supervision:
        raise ValueError("Temperature truth is Oracle-only; pass --oracle_temperature_supervision explicitly")
    if bool(args.temperature_gt) != (args.lambda_temperature_supervision > 0.0):
        raise ValueError("Oracle supervision requires both --temperature_gt and a positive weight")
    if args.tv_neighbors < 1:
        raise ValueError("--tv_neighbors must be at least 1")
    if args.stage == "branch" and args.branch == "K" and (args.lambda_k is None or args.lambda_k < 0.0):
        raise ValueError("Branch K requires a non-negative lambda_k from the scan protocol")
    if args.stage == "branch" and args.branch == "R" and (
            args.lambda_delta_epsilon is None or args.lambda_delta_epsilon < 0.0):
        raise ValueError("Branch R requires a non-negative lambda_delta_epsilon from the scan protocol")
    if args.stage in ("stage2", "galpha") and args.stage2_stop_mode == "plateau" and args.stage2_patience <= 0:
        raise ValueError(
            "Stage 2 requires --stage2_patience > 0; the common endpoint must satisfy "
            "both validation plateau and T/E stability"
        )
    if args.stage == "branch" and args.branch == "K" and args.material_lr >= args.temperature_lr:
        raise ValueError("Branch K requires --material_lr lower than --temperature_lr")
    if args.min_material_gaussians < 1:
        raise ValueError("--min_material_gaussians must be positive")
    if min(args.min_material_temperature_std, args.min_material_temperature_range) < 0.0:
        raise ValueError("Material temperature-spread thresholds must be non-negative")
    if not 0 <= args.lr_hold_fraction < 1 or not 0 <= args.tv_material_boundary_weight <= 1:
        raise ValueError("LR hold fraction/boundary weight outside valid range")
    if (not all(math.isfinite(value) for value in (args.loss_scale_floor, args.tv_temperature_scale_K,
            args.stage2_temperature_stability_K, args.stage2_temperature_stability_p95_K,
            args.stage2_environment_stability, args.stage3_material_stability)) or
            min(args.loss_scale_floor, args.tv_temperature_scale_K) <= 0 or args.gradient_log_every < 0):
        raise ValueError("Loss/TV scales must be positive and gradient logging interval non-negative")
    if min(args.stage2_min_steps, args.stage2_temperature_stability_p95_K) < 0:
        raise ValueError("Minimum steps and stability thresholds must be non-negative")
    if (args.quality_apparent_mae_K is not None or args.quality_signal_rmse_Q is not None) and args.observation_domain != "raw_rjpeg":
        raise ValueError("Physical quality thresholds require raw_rjpeg observations")
    for threshold in (args.quality_apparent_mae_K, args.quality_signal_rmse_Q):
        if threshold is not None and (not math.isfinite(threshold) or threshold <= 0):
            raise ValueError("Quality thresholds must be finite and positive")
    if os.path.isdir(args.model_path) and os.listdir(args.model_path):
        raise ValueError("Output directory is not empty; choose a new -m directory")
    if (not math.isfinite(args.ir_opacity_logit_bound) or not 0 < args.ir_opacity_logit_bound <= 2 or
            not all(math.isfinite(value) for value in
                (args.ir_opacity_lr, args.ir_opacity_lr_final, args.lambda_ir_opacity)) or
            not 0 < args.ir_opacity_lr_final <= args.ir_opacity_lr or args.lambda_ir_opacity < 0 or
            not math.isfinite(args.ir_opacity_stability) or args.ir_opacity_stability < 0):
        raise ValueError("Invalid IR opacity bound, learning rates, or regularization")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.model_path, exist_ok=True)
    dataset.model_path, dataset.load_model_path = args.model_path, args.geometry_model
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians, load_iteration=args.geometry_iteration, shuffle=False)
    if scene.loaded_iter is None:
        raise ValueError("Stage 2/3 requires a saved stage-1 geometry checkpoint; use --geometry_iteration -1 or an explicit iteration")
    args.geometry_sha256 = geometry_checkpoint_sha256(args.geometry_model, scene.loaded_iter)
    if not scene.has_rgbt:
        raise RuntimeError("train_thermal_physics.py currently supports RGBT-Scenes only")
    if not scene.getTrainCameras():
        raise RuntimeError("No training cameras were loaded")
    published_train_cameras = scene.getTrainCameras()
    fit_cameras, validation_cameras = split_fit_validation_cameras(
        published_train_cameras, args.validation_fraction, args.validation_seed)
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        raise RuntimeError("The published RGBT-Scenes test split is required for final-only evaluation")
    print(f"[split] fit={len(fit_cameras)}, validation={len(validation_cameras)}, "
          f"test={len(test_cameras)}; the test split is not used for stopping")
    args.fit_camera_names = [camera.image_name for camera in fit_cameras]
    args.validation_camera_names = [camera.image_name for camera in validation_cameras]
    args.test_camera_names = [camera.image_name for camera in test_cameras]
    for parameter in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                      gaussians._thermal_features_dc, gaussians._thermal_features_rest,
                      gaussians._opacity, gaussians._scaling, gaussians._rotation):
        if torch.is_tensor(parameter): parameter.requires_grad_(False)

    planck = prepare_thermal_input(args, published_train_cameras + test_cameras, device)
    if args.observation_domain == "normalized_dn":
        print("[observation] normalized_dn: temperatures are qualitative model-space estimates, not calibrated absolute truth.")
    elif args.observation_domain == "calibrated_radiance" and args.dn1_radiance <= args.dn0_radiance:
        raise ValueError("--dn1_radiance must be greater than --dn0_radiance")
    elif args.observation_domain == "apparent_temperature" and min(
            args.thermal_raw_max, args.temperature_scale) <= 0.0:
        raise ValueError("Apparent-temperature raw maximum and scale must be positive")
    support_weights = torch.zeros(gaussians.get_xyz.shape[0], device=device)
    visibility_color = torch.ones((gaussians.get_xyz.shape[0], 3), device=device)
    def visibility_provider(camera):
        package = render(camera, gaussians, pipe, torch.zeros(3, device=device), 0.0, 0.0, 0.0,
                         device, dataset.is_6dof, override_color=visibility_color,
                         detach_geometry=True)
        with torch.no_grad():
            support_weights.add_(package["visibility_filter"].to(support_weights) *
                                 gaussians.get_opacity.detach().reshape(-1) * package["radii"].detach().float().square())
        return package["visibility_filter"], package["radii"]
    install_loss_scale(args, fit_cameras,
                       lambda image: observation_to_model_radiance(image, args, planck))
    print("[loss-scale] " + json.dumps(args.loss_scale_protocol))
    enforce_comparison_protocol(args)
    material_config = load_material_config(args.material_config) if args.stage == "stage2" else None
    if material_config is not None and abs(material_config["temperature_reference_K"] - args.temp_ref) > 1e-6:
        raise ValueError("material_config temperature_reference_K must equal --temp_ref")
    if material_config is not None:
        args.material_config_document = material_config["document"]
        args.material_config_sha256 = file_sha256(args.material_config)
    field = make_field(args, scene, device, planck, visibility_provider, fit_cameras, material_config)
    if args.stage == "branch":
        identifiability = field.apply_identifiability_gate(
            args.min_material_gaussians,
            args.min_material_temperature_std,
            args.min_material_temperature_range,
        )
        print("[material-identifiability] " + json.dumps(identifiability))
    temperature_truth = load_temperature_truth(args.temperature_gt, gaussians.get_xyz.shape[0], device)
    pair_i, pair_j, pair_w = build_spatial_tv_edges(gaussians.get_xyz, args.tv_neighbors)
    if pair_i.numel():
        boundary = ((field.material_ids[pair_i] >= 0) & (field.material_ids[pair_j] >= 0) &
                    (field.material_ids[pair_i] != field.material_ids[pair_j]))
        pair_w = pair_w * torch.where(boundary, pair_w.new_tensor(args.tv_material_boundary_weight),
                                     pair_w.new_tensor(1.0))
    if not bool((support_weights > 0).any()):
        for camera in fit_cameras:
            visibility_provider(camera)
    support = support_weights > 0
    if not bool(support.any()):
        raise ValueError("No visible Gaussian support in fit cameras")
    opacity_field = IROpacityCorrection.from_checkpoint(
        {"ir_opacity": getattr(args, "reference_ir_opacity_state", None)},
        gaussians.get_opacity, trainable=(args.stage == "galpha" and args.ir_opacity_mode == "bounded"))
    if args.stage == "galpha" and args.ir_opacity_mode == "bounded":
        if opacity_field is None:
            opacity_field = IROpacityCorrection(gaussians.get_opacity, support,
                args.ir_opacity_logit_bound, trainable=True).to(device)
        elif abs(opacity_field.logit_bound - args.ir_opacity_logit_bound) > 1e-12:
            raise ValueError("G-alpha bound differs from the warm-start opacity checkpoint")
    args.ir_opacity_protocol = {"enabled": opacity_field is not None,
        "trainable": opacity_field is not None and opacity_field.raw.requires_grad,
        "logit_bound": opacity_field.logit_bound if opacity_field is not None else None,
        "lambda_ir_opacity": args.lambda_ir_opacity,
        "ir_opacity_lr": args.ir_opacity_lr, "ir_opacity_lr_final": args.ir_opacity_lr_final,
        "support": "reference_fit_frustum_and_nonzero_RGB_opacity",
        "RGB_opacity_changed": False}
    print("[ir-opacity] " + json.dumps(args.ir_opacity_protocol))
    sh_field = ThermalSHResidual.from_checkpoint(
        {"sh_residual": getattr(args, "reference_sh_residual_state", None)}, gaussians.get_xyz, trainable=False)
    if args.sh_residual_degree == 2:
        signal_bound = args.sh_residual_bound * float(args.loss_scale_protocol["scale"])
        sh_field = ThermalSHResidual(gaussians.get_xyz, support, signal_bound, trainable=True).to(device)
    args.sh_residual_protocol = {"enabled": sh_field is not None,
        "trainable": sh_field is not None and sh_field.raw.requires_grad,
        "degree": 2 if sh_field is not None else 0, "channels": 1, "DC_trainable": False,
        "coefficients_per_gaussian": 8 if sh_field is not None else 0,
        "signal_bound": sh_field.signal_bound if sh_field is not None else None,
        "bound_fraction_of_loss_scale": args.sh_residual_bound if args.sh_residual_degree else None,
        "start_step": args.sh_residual_start_step if args.sh_residual_degree else None,
        "lr": args.sh_residual_lr, "lr_final": args.sh_residual_lr_final,
        "lambda": args.lambda_sh_residual, "fresh_stage2_initialization": args.stage == "stage2",
        "direction": "normalized_Gaussian_xyz_minus_camera_center",
        "support": "fit_frustum_nonzero_RGB_opacity", "signal_clamp_before_blending": [0, 1]}
    print("[sh-residual] " + json.dumps(args.sh_residual_protocol))
    environment_anchor = field.environment.detach().clone()
    optimizer_groups = [
        {"name": "temperature", "params": [field.temperature_raw], "lr": args.temperature_lr},
        {"name": "environment", "params": [field.environment_raw], "lr": args.environment_lr},
    ]
    if field.branch == "K":
        optimizer_groups.append({"name": "material", "params": [field.k_raw], "lr": args.material_lr})
    elif field.branch == "R":
        optimizer_groups.append({"name": "material", "params": [field.delta_epsilon_raw], "lr": args.material_lr})
    if opacity_field is not None and opacity_field.raw.requires_grad:
        optimizer_groups.append({"name": "ir_opacity", "params": [opacity_field.raw], "lr": args.ir_opacity_lr})
    if sh_field is not None and sh_field.raw.requires_grad:
        optimizer_groups.append({"name": "sh_residual", "params": [sh_field.raw], "lr": 0.0})
    optimizer = torch.optim.Adam(optimizer_groups)
    cameras, background = fit_cameras, torch.zeros(3, device=device)
    camera_order = list(range(len(cameras)))
    camera_rng = random.Random(args.seed)
    log_path = os.path.join(args.model_path, "thermal_training.jsonl")
    tracker = PlateauTracker(args.stage2_min_delta, args.stage2_relative_min_delta)
    final_step, stop_reason = 0, "max_steps"
    stable = False
    latest_metrics = validation_metrics(field, planck, validation_cameras, gaussians, pipe, background, dataset, args, opacity_field, sh_field)
    tracker.update(latest_metrics["radiance_loss"], 0)
    initial_status = quality_status(latest_metrics, args.quality_apparent_mae_K, args.quality_signal_rmse_Q)
    save_thermal_checkpoint(os.path.join(args.model_path, f"thermal_{field.branch}_best.pt"), field, 0,
        checkpoint_metadata(args, scene, validation_radiance_loss=tracker.best,
            validation_metrics=latest_metrics, quality_status=initial_status),
        frozen_geometry=frozen_geometry_state(scene.gaussians), ir_opacity=opacity_field, sh_residual=sh_field)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 0, "initial_validation_metrics": latest_metrics,
                                 "quality_status": initial_status}) + "\n")
    previous_eval_temperature = field.temperature.detach().clone()
    previous_eval_environment = field.environment.detach().clone()
    previous_eval_opacity = render_opacity(opacity_field)
    previous_eval_opacity = previous_eval_opacity.detach().clone() if previous_eval_opacity is not None else None
    previous_eval_material = (field.k_epsilon_by_material.detach().clone()
                              if field.branch == "K" else field.delta_epsilon_by_material.detach().clone())

    for step in range(1, args.steps + 1):
        final_step = step
        field.set_branch_trainability()
        camera_position = (step - 1) % len(cameras)
        if camera_position == 0:
            camera_rng.shuffle(camera_order)
        camera = cameras[camera_order[camera_position]]
        radiance, sh_delta, sh_clipped = thermal_view_radiance(
            field.radiance(planck), gaussians.get_xyz, camera.camera_center, sh_field)
        prediction = render(camera, gaussians, pipe, background, 0.0, 0.0, 0.0, device,
            dataset.is_6dof, override_color=radiance.repeat(1, 3), detach_geometry=True,
            override_opacity=render_opacity(opacity_field))["render"]
        target = camera.original_physical_image if camera.original_physical_image is not None else camera.original_image
        target = target.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        target = observation_to_model_radiance(target, args, planck)
        mask = getattr(camera, "thermal_valid_mask", None)
        rad_loss = thermal_data_loss(prediction, target, args, mask)
        if pair_i.numel():
            temperature = field.temperature.reshape(-1)
            tv_loss_K = (pair_w * (temperature[pair_i] - temperature[pair_j]).abs()).sum() / pair_w.sum().clamp_min(1e-12)
            tv_loss = tv_loss_K / args.tv_temperature_scale_K
        else:
            tv_loss = tv_loss_K = rad_loss.new_zeros(())
        environment_loss = F.smooth_l1_loss(
            field.environment, environment_anchor, beta=args.environment_prior_beta)
        material_loss = field.branch_regularizer()
        supervised_temperature_loss = (
            F.smooth_l1_loss(field.temperature, temperature_truth)
            if temperature_truth is not None else rad_loss.new_zeros(())
        )
        opacity_loss = (opacity_field.regularizer() if opacity_field is not None and opacity_field.raw.requires_grad
                        else rad_loss.new_zeros(()))
        loss = rad_loss + args.lambda_tv * tv_loss + args.lambda_environment * environment_loss
        loss = loss + args.lambda_ir_opacity * opacity_loss
        sh_loss = (sh_field.regularizer() if sh_field is not None and sh_field.raw.requires_grad
                   else rad_loss.new_zeros(()))
        loss = loss + args.lambda_sh_residual * sh_loss
        branch_regularization_weight = (
            args.lambda_k if field.branch == "K"
            else (args.lambda_delta_epsilon if field.branch == "R" else 0.0)
        )
        loss = loss + branch_regularization_weight * material_loss
        loss = loss + args.lambda_temperature_supervision * supervised_temperature_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite thermal loss at step {step}: radiance={float(rad_loss.detach())}, "
                f"tv={float(tv_loss.detach())}, environment={float(environment_loss.detach())}, "
                f"material={float(material_loss.detach())}"
            )
        gradient_diagnostics = {}
        if args.gradient_log_every > 0 and (step == 1 or step % args.gradient_log_every == 0):
            data_grad = torch.autograd.grad(rad_loss, field.temperature_raw, retain_graph=True)[0]
            tv_grad = (torch.autograd.grad(args.lambda_tv * tv_loss, field.temperature_raw, retain_graph=True)[0]
                       if pair_i.numel() and args.lambda_tv > 0 else torch.zeros_like(data_grad))
            data_norm, tv_norm = float(data_grad.norm()), float(tv_grad.norm())
            gradient_diagnostics = {"temperature_data_gradient_norm": data_norm,
                                   "temperature_weighted_tv_gradient_norm": tv_norm,
                                   "tv_to_data_gradient_ratio": tv_norm / max(data_norm, 1e-12)}
        if gradient_diagnostics and opacity_field is not None and opacity_field.raw.requires_grad:
            opacity_grad = torch.autograd.grad(rad_loss, opacity_field.raw, retain_graph=True)[0]
            gradient_diagnostics["ir_opacity_data_gradient_norm"] = float(opacity_grad.norm())
        if gradient_diagnostics and sh_field is not None and sh_field.raw.requires_grad:
            sh_grad = torch.autograd.grad(rad_loss, sh_field.raw, retain_graph=True)[0]
            gradient_diagnostics["sh_data_gradient_norm"] = float(sh_grad.norm())
        for group in optimizer.param_groups:
            if group["name"] == "temperature":
                group["lr"] = held_lr(step, args.steps, args.temperature_lr, args.temperature_lr_final, args.lr_hold_fraction)
            elif group["name"] == "environment":
                group["lr"] = held_lr(step, args.steps, args.environment_lr, args.environment_lr_final, args.lr_hold_fraction)
            elif group["name"] == "ir_opacity":
                group["lr"] = held_lr(step, args.steps, args.ir_opacity_lr, args.ir_opacity_lr_final, args.lr_hold_fraction)
            elif group["name"] == "sh_residual":
                group["lr"] = (0.0 if step < args.sh_residual_start_step else held_lr(
                    step - args.sh_residual_start_step + 1, args.steps - args.sh_residual_start_step + 1,
                    args.sh_residual_lr, args.sh_residual_lr_final, args.lr_hold_fraction))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if sh_field is not None and sh_field.raw.requires_grad:
            if sh_field.raw.grad is None or not bool(torch.isfinite(sh_field.raw.grad).all()):
                raise FloatingPointError("Missing or non-finite thermal SH residual gradient")
            if step < args.sh_residual_start_step:
                # Warm up only the physical field; do not accumulate Adam
                # moments from gradients of a branch that is still disabled.
                sh_field.raw.grad = None
        if opacity_field is not None and opacity_field.raw.requires_grad:
            if opacity_field.raw.grad is None or not bool(torch.isfinite(opacity_field.raw.grad).all()):
                raise FloatingPointError("Missing or non-finite IR opacity gradient")
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps or gradient_diagnostics:
            normalized_huber = F.smooth_l1_loss(prediction.detach(), target, beta=args.huber_delta, reduction="none")
            learning_rates = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
            has_temperature_truth = temperature_truth is not None
            current_temperature = field.temperature.detach()
            current_emissivity = field.emissivity().detach()
            temperature_key = ("apparent_temperature_mean_K"
                               if args.observation_domain == "normalized_dn" and not has_temperature_truth
                               else "temperature_mean_K")
            temperature_semantics = (
                "synthetic_supervised_thermodynamic" if has_temperature_truth
                else ("uncalibrated_apparent_proxy" if args.observation_domain == "normalized_dn"
                      else ("calibrated_apparent_temperature" if
                            args.observation_domain == "apparent_temperature"
                            else ("calibrated_camera_signal_inversion" if
                                  args.observation_domain == "raw_rjpeg"
                                  else "calibrated_radiance_inversion")))
            )
            record = {"step": step, "branch": field.branch,
                "sh_residual_lr": learning_rates.get("sh_residual", 0.0),
                "sh_prior_loss": float(sh_loss.detach()),
                "weighted_sh_prior_loss": float((args.lambda_sh_residual * sh_loss).detach()),
                "sh_view_abs_delta_mean": float(sh_delta.detach().abs().mean()),
                "sh_view_abs_delta_max": float(sh_delta.detach().abs().max()),
                "sh_view_clipped_fraction": float(sh_clipped.detach()),
                **(sh_field.diagnostics() if sh_field is not None else {}),
                "temperature_lr": learning_rates["temperature"],
                "environment_lr": learning_rates["environment"],
                "ir_opacity_lr": learning_rates.get("ir_opacity", 0.0),
                "ir_opacity_prior_loss": float(opacity_loss.detach()),
                "weighted_ir_opacity_prior_loss": float((args.lambda_ir_opacity * opacity_loss).detach()),
                **(opacity_field.diagnostics() if opacity_field is not None else {}),
                "loss": float(loss.detach()), "radiance_loss": float(rad_loss.detach()),
                "tv_loss": float(tv_loss.detach()), "tv_loss_K": float(tv_loss_K.detach()),
                "weighted_tv_loss": float((args.lambda_tv * tv_loss).detach()),
                "weighted_environment_loss": float((args.lambda_environment * environment_loss).detach()),
                "weighted_material_loss": float((branch_regularization_weight * material_loss).detach()),
                "normalized_signal_huber_loss": float(normalized_huber.mean() if mask is None else masked_mean(normalized_huber, mask)),
                "tv_to_data_loss_ratio": float((args.lambda_tv * tv_loss).detach()) / max(float(rad_loss.detach()), 1e-12),
                "environment_mean": float(field.environment.detach().mean()),
                **gradient_diagnostics,
                temperature_key: float(field.temperature.detach().mean()),
                "temperature_semantics": temperature_semantics,
                "material_valid_fraction": float(field.material_valid.float().mean()),
                "temperature_bound_fraction": float(
                    ((current_temperature <= args.temp_min + 0.1) |
                     (current_temperature >= args.temp_max - 0.1)).float().mean()),
                "emissivity_bound_fraction": float(
                    ((current_emissivity <= 0.011) | (current_emissivity >= 0.989)).float().mean()),
                "k_by_material": {name: float(value.detach()) for name, value in
                                  zip(field.material_names, field.k_epsilon_by_material)},
                "delta_epsilon_by_material": {name: float(value.detach()) for name, value in
                                               zip(field.material_names, field.delta_epsilon_by_material)}}
            if temperature_truth is not None:
                temperature_error = field.temperature.detach() - temperature_truth
                record["temperature_mae_K"] = float(temperature_error.abs().mean())
                record["temperature_rmse_K"] = float(temperature_error.square().mean().sqrt())
            print(json.dumps(record))
            with open(log_path, "a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            save_thermal_checkpoint(os.path.join(args.model_path, f"thermal_{field.branch}_step_{step}.pt"),
                field, step, checkpoint_metadata(args, scene), ir_opacity=opacity_field, sh_residual=sh_field)
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            latest_metrics = validation_metrics(field, planck, validation_cameras, gaussians,
                                                pipe, background, dataset, args, opacity_field, sh_field)
            validation = latest_metrics["radiance_loss"]
            updates = (field.temperature.detach() - previous_eval_temperature).abs().reshape(-1)
            temperature_update = float((updates * support_weights).sum() / support_weights.sum().clamp_min(1e-12))
            temperature_update_p95 = float(torch.quantile(updates[support], 0.95))
            environment_update = float((field.environment.detach() - previous_eval_environment).abs().mean())
            current_material = (field.k_epsilon_by_material.detach() if field.branch == "K"
                                else field.delta_epsilon_by_material.detach())
            material_update = float((current_material - previous_eval_material).abs().mean())
            previous_eval_temperature = field.temperature.detach().clone()
            previous_eval_environment = field.environment.detach().clone()
            previous_eval_material = current_material.clone()
            stable = (temperature_update <= args.stage2_temperature_stability_K and
                      temperature_update_p95 <= args.stage2_temperature_stability_p95_K and
                      environment_update <= args.stage2_environment_stability and
                      (field.branch in ("stage2", "C") or material_update <= args.stage3_material_stability))
            opacity_update = 0.0
            if opacity_field is not None:
                current_opacity = opacity_field.opacity.detach()
                opacity_update = float((current_opacity[opacity_field.fit_support] -
                    previous_eval_opacity[opacity_field.fit_support]).abs().mean()) if bool(opacity_field.fit_support.any()) else 0.0
                previous_eval_opacity = current_opacity.clone()
                stable = stable and opacity_update <= args.ir_opacity_stability
            improved, meaningful = tracker.update(validation, step)
            status = quality_status(latest_metrics, args.quality_apparent_mae_K, args.quality_signal_rmse_Q)
            evaluation_record = {"step": step, "branch": field.branch,
                "validation_radiance_loss": validation, "validation_metrics": latest_metrics,
                "visible_weighted_temperature_update_K": temperature_update,
                "visible_temperature_update_p95_K": temperature_update_p95,
                "environment_update": environment_update, "material_parameter_update": material_update,
                "parameters_stable": stable, "ir_opacity_update": opacity_update,
                "ir_opacity_diagnostics": opacity_field.diagnostics() if opacity_field is not None else None,
                "actual_best_improved": improved,
                "meaningful_improvement": meaningful, "best_step": tracker.best_step,
                "best_validation_loss": tracker.best, "stale_checks": tracker.stale,
                "quality_status": status}
            print(json.dumps(evaluation_record))
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(evaluation_record) + "\n")
            if improved:
                save_thermal_checkpoint(os.path.join(args.model_path, f"thermal_{field.branch}_best.pt"),
                    field, step, checkpoint_metadata(args, scene, validation_radiance_loss=validation,
                    validation_metrics=latest_metrics, quality_status=status),
                    frozen_geometry=frozen_geometry_state(scene.gaussians), ir_opacity=opacity_field, sh_residual=sh_field)
            if (args.stage in ("stage2", "galpha") and args.stage2_stop_mode == "plateau" and
                    step >= args.stage2_min_steps and tracker.stale >= args.stage2_patience and stable):
                stop_reason = "plateau"
                break

    test_radiance_loss = None
    if args.stage == "branch":
        test_metrics = validation_metrics(field, planck, test_cameras, gaussians, pipe, background, dataset, args, opacity_field, sh_field)
        test_radiance_loss = test_metrics["radiance_loss"]
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": final_step, "branch": field.branch,
                "final_test_metrics": test_metrics, "test_used_for_training_or_stopping": False}) + "\n")
    status = quality_status(latest_metrics, args.quality_apparent_mae_K, args.quality_signal_rmse_Q)
    training_summary = {"stop_reason": stop_reason, "stopped_step": final_step,
        "best_step": tracker.best_step, "best_validation_loss": tracker.best,
        "final_validation_metrics": latest_metrics, "final_quality_status": status,
        "parameters_stable_at_stop": stable, "quality_thresholds": {
            "apparent_temperature_MAE_K": args.quality_apparent_mae_K,
            "camera_signal_RMSE": args.quality_signal_rmse_Q},
        "stage2_stop_mode": args.stage2_stop_mode,
        "stage2_min_steps": args.stage2_min_steps,
        "stage2_patience": args.stage2_patience,
        "stage2_min_delta": args.stage2_min_delta,
        "stage2_relative_min_delta": args.stage2_relative_min_delta,
        "selection": "minimum_internal_validation_scaled_Huber", "test_used_for_selection": False,
        "training_stage": args.stage, "ir_opacity_protocol": args.ir_opacity_protocol,
        "sh_residual_protocol": args.sh_residual_protocol,
        "final_sh_diagnostics": sh_field.diagnostics() if sh_field is not None else None,
        "final_ir_opacity_diagnostics": opacity_field.diagnostics() if opacity_field is not None else None}
    final_metadata = checkpoint_metadata(args, scene, best_validation_radiance_loss=tracker.best,
        final_test_radiance_loss=test_radiance_loss, stage2_common_endpoint=False,
        training_summary=training_summary, quality_status=status)
    save_thermal_checkpoint(os.path.join(args.model_path, f"thermal_{field.branch}_final.pt"),
        field, final_step, final_metadata, frozen_geometry=frozen_geometry_state(scene.gaussians), ir_opacity=opacity_field, sh_residual=sh_field)
    if args.stage in ("stage2", "galpha"):
        best = torch.load(os.path.join(args.model_path, "thermal_stage2_best.pt"), map_location="cpu")
        best["metadata"].update(stage2_common_endpoint=True,
            stage2_endpoint_policy="best_validation_within_training_budget",
            optimization_converged=(stop_reason == "plateau"), training_summary=training_summary)
        training_summary["selected_quality_status"] = best["metadata"]["quality_status"]
        training_summary["selected_validation_metrics"] = best["metadata"]["validation_metrics"]
        if best.get("sh_residual") is not None:
            selected_sh = ThermalSHResidual.from_checkpoint(best, gaussians.get_xyz, trainable=False)
            training_summary["selected_sh_diagnostics"] = selected_sh.diagnostics()
        if best.get("ir_opacity") is not None:
            best_opacity = IROpacityCorrection.from_checkpoint(best,
                best["ir_opacity"]["state_dict"]["base_opacity"], trainable=False)
            training_summary["selected_ir_opacity_diagnostics"] = best_opacity.diagnostics()
        torch.save(best, os.path.join(args.model_path, "thermal_stage2_common.pt"))
    with open(os.path.join(args.model_path, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2, allow_nan=False)
    print("[training-summary] " + json.dumps(training_summary, allow_nan=False))



if __name__ == "__main__":
    main()
