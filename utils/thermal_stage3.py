"""Frozen D initialization and explicit, matched C/R/K pilot protocols."""
import copy
import hashlib
import json
import os
from pathlib import Path

import torch

from utils.thermal_joint import GEOMETRY_NAMES, geometry_digest, install_joint_geometry
from utils.thermal_physics import MaterialThermalField
from utils.thermal_sh_residual import ThermalSHResidual

COMMON_KIND = "frozen_D_stage3_common_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def camera_input_hashes(source):
    source = Path(source)
    paths = []
    for directory in ("rgb/train", "rgb/test", "sparse/0", "colmap/sparse/0"):
        root = source / directory
        if root.is_dir():
            paths.extend(p for p in root.iterdir() if p.is_file())
    return {str(p.relative_to(source)): file_sha256(p) for p in sorted(paths)}


def validate_common(checkpoint):
    metadata = checkpoint.get("metadata", {})
    common = metadata.get("stage3_common", {})
    if (checkpoint.get("branch") != "stage2" or common.get("kind") != COMMON_KIND
            or not metadata.get("stage3_common_endpoint")):
        raise ValueError("Stage3 requires the exported frozen D common checkpoint")
    geometry = checkpoint.get("joint_geometry")
    joint = metadata.get("joint_protocol", {})
    if (geometry is None or geometry_digest(geometry) != common.get("geometry_sha256")
            or geometry_digest(geometry) != joint.get("trained_geometry_sha256")):
        raise ValueError("Stage3 frozen geometry checksum mismatch")
    state = checkpoint["state_dict"]
    if tensor_digest(state) != common.get("field_sha256"):
        raise ValueError("Stage3 common field checksum mismatch")
    sh = checkpoint.get("sh_residual", {})
    if (sh.get("kind") != "single_band_sh2_no_dc" or sh.get("degree") != 2
            or tensor_digest(sh["state_dict"]) != common.get("sh_sha256")):
        raise ValueError("Stage3 requires the unchanged scalar SH8 residual")
    count = len(geometry["_xyz"])
    if count != len(state["material_ids"]) or sh["state_dict"]["raw"].shape != (count, 8):
        raise ValueError("Stage3 field, SH and geometry counts differ")
    for section in (state, sh["state_dict"], {k: geometry[k] for k in GEOMETRY_NAMES}):
        if any(not bool(torch.isfinite(value).all()) for value in section.values()):
            raise ValueError("Non-finite common checkpoint tensor")
    if not torch.equal(state["learn_k_by_material"], state["learn_delta_by_material"]):
        raise ValueError("K and R must enable the same materials")
    if bool(state["k_raw"].count_nonzero()) or bool(state["delta_epsilon_raw"].count_nonzero()):
        raise ValueError("Common material corrections must start at zero")
    shared = metadata["shared_training_protocol"]
    if (shared.get("observation_domain") != "raw_rjpeg" or shared.get("lambda_tv") != 0
            or shared.get("temperature_gt") or shared.get("lambda_temperature_supervision", 0)
            or shared.get("oracle_temperature_supervision", False)):
        raise ValueError("Stage3 pilot requires raw_rjpeg, no TV, and no GT supervision")
    return common


def export_common(checkpoint, source_path, enabled_materials):
    """Preserve trained D tensors; only replace explicit K/R candidate switches."""
    metadata = checkpoint.get("metadata", {})
    joint = metadata.get("joint_protocol", {})
    if checkpoint.get("branch") != "stage2" or joint.get("kind") != "fixed_count_rgb_ir_D_v1":
        raise ValueError("Input must be a completed D scalar-SH checkpoint")
    geometry = checkpoint.get("joint_geometry")
    if geometry is None or geometry_digest(geometry) != joint.get("trained_geometry_sha256"):
        raise ValueError("D trained geometry checksum mismatch")
    names = checkpoint["config"]["material_names"]
    if len(set(enabled_materials)) != len(enabled_materials) or set(enabled_materials) - set(names):
        raise ValueError("Unknown/duplicate candidate material names")
    payload = copy.deepcopy(checkpoint)
    state = payload["state_dict"]
    if bool(state["k_raw"].count_nonzero()) or bool(state["delta_epsilon_raw"].count_nonzero()):
        raise ValueError("D checkpoint unexpectedly contains material corrections")
    enabled = torch.tensor([name in enabled_materials for name in names], dtype=torch.bool)
    counts = [(state["material_ids"] == i).sum().item() for i in range(len(names))]
    if any(active and not count for active, count in zip(enabled.tolist(), counts)):
        raise ValueError("An enabled material has no assigned Gaussians")
    state["learn_k_by_material"] = enabled.clone()
    state["learn_delta_by_material"] = enabled.clone()
    info = payload["metadata"]
    # This is a separate contract. The old Stage1-frozen Stage2 loader must reject it.
    info["stage2_common_endpoint"] = False
    info["stage3_common_endpoint"] = True
    info["training_stage"] = "stage3_common_from_D"
    info["stage3_common"] = {
        "kind": COMMON_KIND, "source_checkpoint": str(Path(source_path).resolve()),
        "source_checkpoint_sha256": file_sha256(source_path), "source_step": int(payload["step"]),
        "geometry_sha256": geometry_digest(geometry), "field_sha256": tensor_digest(state),
        "sh_sha256": tensor_digest(payload["sh_residual"]["state_dict"]),
        "enabled_materials": list(enabled_materials),
        "material_policy": [{"id": i, "name": name, "gaussians": counts[i],
                             "epsilon0": float(state["epsilon0_by_material"][i]),
                             "learn_k_and_delta": bool(enabled[i])} for i, name in enumerate(names)],
        "unknown_gaussians": int((state["material_ids"] == -1).sum()),
        "epsilon0_fixed": True, "geometry_opacity_RGB_SH_thermal_SH_frozen": True,
        "K_UQ_enabled": False, "surface_parameter_GT_available": False,
        "camera_input_sha256": camera_input_hashes(info["source_path"]),
    }
    validate_common(payload)
    return payload


def verify_inputs(checkpoint):
    common = validate_common(checkpoint)
    metadata = checkpoint["metadata"]
    geometry = Path(metadata["geometry_model"]) / "point_cloud" / (
        "iteration_" + str(metadata["geometry_iteration"])) / "point_cloud.ply"
    if file_sha256(geometry) != metadata["geometry_sha256"]:
        raise ValueError("Stage1 initializer file changed")
    if camera_input_hashes(metadata["source_path"]) != common["camera_input_sha256"]:
        raise ValueError("RGB observations or camera/scene inputs changed")
    joint = metadata["joint_protocol"]
    if file_sha256(joint["display_calibration"]) != joint["display_calibration_sha256"]:
        raise ValueError("Frozen display calibration changed")


def restore_models(checkpoint, gaussians, branch, device, reset=True):
    install_joint_geometry(gaussians, checkpoint)
    for name in GEOMETRY_NAMES + ("_thermal_features_dc", "_thermal_features_rest"):
        getattr(gaussians, name).requires_grad_(False)
    field_checkpoint = dict(checkpoint, state_dict={key: value.clone() for key, value in checkpoint["state_dict"].items()})
    field = MaterialThermalField.from_checkpoint(field_checkpoint, branch, device,
                                                reset_branch_parameters=reset)
    sh = ThermalSHResidual.from_checkpoint(checkpoint, gaussians.get_xyz, trainable=False)
    if sh is None:
        raise ValueError("Missing frozen thermal SH")
    return field, sh


def assert_frozen(checkpoint, field, gaussians, sh):
    common = checkpoint["metadata"]["stage3_common"]
    from utils.thermal_physics import frozen_geometry_state
    if geometry_digest(frozen_geometry_state(gaussians)) != common["geometry_sha256"]:
        raise RuntimeError("Stage3 changed geometry, opacity or RGB SH")
    if tensor_digest(sh.state_dict()) != common["sh_sha256"] or sh.raw.requires_grad:
        raise RuntimeError("Stage3 changed thermal SH")
    for name, value in field.named_buffers():
        if not torch.equal(value.cpu(), checkpoint["state_dict"][name].cpu()):
            raise RuntimeError("Stage3 changed frozen material buffer: " + name)


def lock_protocol(path, protocol):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(document)
    except FileExistsError:
        if json.loads(path.read_text()) != protocol:
            raise ValueError("C/R/K protocol mismatch; use a separate lock for a new seed/settings")
    return file_sha256(path)


def atomic_save(payload, path):
    path = Path(path)
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)
