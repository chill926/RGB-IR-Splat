"""B0: non-physical thermal-image fitting on frozen RGB geometry."""
import argparse
import json
import os
import random

import numpy as np
import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, pipeline = ModelParams(parser, device), PipelineParams(parser)
    parser.add_argument("--geometry_model", required=True)
    parser.add_argument("--geometry_iteration", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    args = parser.parse_args()
    args.data_branch = "rgbt"
    args.geometry_model, args.model_path = os.path.abspath(args.geometry_model), os.path.abspath(args.model_path)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.model_path, exist_ok=True)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    dataset.model_path, dataset.load_model_path = args.model_path, args.geometry_model
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians, load_iteration=args.geometry_iteration, shuffle=False)
    if scene.loaded_iter is None:
        raise RuntimeError("B0 requires a saved stage-1 geometry checkpoint")
    if not scene.has_rgbt or not scene.getTrainCameras() or not scene.getTestCameras():
        raise RuntimeError("B0 requires an evaluated RGBT scene with non-empty train/test splits")
    for parameter in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                      gaussians._opacity, gaussians._scaling, gaussians._rotation):
        parameter.requires_grad_(False)
    gaussians._thermal_features_dc.requires_grad_(True)
    gaussians._thermal_features_rest.requires_grad_(True)
    optimizer = torch.optim.Adam([
        {"params": [gaussians._thermal_features_dc], "lr": args.feature_lr},
        {"params": [gaussians._thermal_features_rest], "lr": args.feature_lr / 20.0},
    ])
    cameras, background = scene.getTrainCameras(), torch.zeros(3, device=device)
    camera_order, camera_rng = list(range(len(cameras))), random.Random(args.seed)
    log_path = os.path.join(args.model_path, "thermal_image_baseline.jsonl")
    for step in range(1, args.steps + 1):
        camera_position = (step - 1) % len(cameras)
        if camera_position == 0:
            camera_rng.shuffle(camera_order)
        camera = cameras[camera_order[camera_position]]
        prediction = render(camera, gaussians, pipe, background, 0.0, 0.0, 0.0, device,
            dataset.is_6dof, feature_set="thermal", detach_geometry=True)["render"]
        target = camera.original_image
        l1 = l1_loss(prediction, target)
        loss = (1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * (1.0 - ssim(prediction, target))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            record = {"step": step, "loss": float(loss.detach()), "l1": float(l1.detach())}
            print(json.dumps(record))
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
    scene.save(args.steps)
    validation = []
    with torch.no_grad():
        for camera in scene.getTestCameras():
            prediction = render(camera, gaussians, pipe, background, 0.0, 0.0, 0.0, device,
                dataset.is_6dof, feature_set="thermal", detach_geometry=True)["render"].clamp(0.0, 1.0)
            validation.append({"l1": float(l1_loss(prediction, camera.original_image)),
                               "psnr": float(psnr(prediction, camera.original_image).mean())})
    validation_summary = None
    if validation:
        validation_summary = {"l1": sum(row["l1"] for row in validation) / len(validation),
                              "psnr": sum(row["psnr"] for row in validation) / len(validation)}
    with open(os.path.join(args.model_path, "B0_METADATA.json"), "w", encoding="utf-8") as handle:
        json.dump({"physical_temperature_meaning": False, "geometry_model": args.geometry_model,
                   "geometry_iteration": scene.loaded_iter, "steps": args.steps, "seed": args.seed,
                   "validation": validation_summary}, handle, indent=2)


if __name__ == "__main__":
    main()
