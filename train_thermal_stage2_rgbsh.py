"""Fresh Stage2: physical core + 24 non-DC RGB SH coefficients; joint geometry."""
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import train_thermal_physics as base
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.thermal_rgb_sh import KIND, ThermalRGBSH, appearance
from utils.thermal_joint import geometry_digest, image_loss
from utils.thermal_physics import frozen_geometry_state, build_spatial_tv_edges
from utils.thermal_training_utils import held_lr, install_loss_scale, thermal_data_loss, masked_mean

def add_arguments(parser):
    for action in parser._actions:
        if action.dest == 'stage':
            action.required = False
            action.default = 'stage2'
        if action.dest == 'observation_domain':
            action.choices = tuple(action.choices) + ('pseudocolor_rgb',)
    parser.add_argument('--rgb_sh_bound', type=float, default=0.5)
    parser.add_argument('--rgb_sh_lr', type=float, default=0.01)
    parser.add_argument('--rgb_sh_lr_final', type=float, default=0.0001)
    parser.add_argument('--lambda_rgb_sh', type=float, default=0.001)
    parser.add_argument('--joint_position_lr', type=float, default=1.6e-5)
    parser.add_argument('--joint_position_lr_final', type=float, default=1.6e-7)
    parser.add_argument('--joint_scaling_lr', type=float, default=5e-4)
    parser.add_argument('--joint_rotation_lr', type=float, default=1e-4)
    parser.add_argument('--joint_opacity_lr', type=float, default=5e-3)
    parser.add_argument('--lambda_rgb', type=float, default=0.0,
                        help='Optional paired RGB anchor; Stage1 RGB SH stays frozen')
    parser.add_argument('--photometric_loss', choices=('huber', 'l1_ssim', 'mse'), default='huber')

def validate_args(args):
    if args.stage != 'stage2' or args.stage2_checkpoint or args.sh_residual_degree != 0:
        raise ValueError('This entry starts fresh Stage2 from Stage1, with its own RGB SH24, not old Stage2/SH8 weights')
    if args.stage2_stop_mode != 'budget':
        raise ValueError('Joint Stage2 currently uses a fixed training budget')
    if args.temperature_gt or args.lambda_temperature_supervision or args.oracle_temperature_supervision:
        raise ValueError('This entry does not use temperature ground-truth/oracle supervision')
    if args.steps < 1 or args.eval_every < 1 or args.save_every < 1:
        raise ValueError('Steps and evaluation/save intervals must be positive')
    positive = ('rgb_sh_bound', 'rgb_sh_lr', 'rgb_sh_lr_final', 'joint_position_lr',
                'joint_position_lr_final', 'joint_scaling_lr', 'joint_rotation_lr',
                'joint_opacity_lr', 'temperature_lr', 'temperature_lr_final',
                'environment_lr', 'environment_lr_final', 'huber_delta',
                'loss_scale_floor', 'tv_temperature_scale_K', 'environment_prior_beta')
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid positive parameter: ' + name)
    for name in ('lambda_rgb_sh', 'lambda_rgb', 'lambda_tv', 'lambda_environment'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid regularization/anchor weight: ' + name)
    if any(getattr(args, end) > getattr(args, start) for start, end in (
        ('rgb_sh_lr', 'rgb_sh_lr_final'), ('joint_position_lr', 'joint_position_lr_final'),
        ('temperature_lr', 'temperature_lr_final'), ('environment_lr', 'environment_lr_final'))):
        raise ValueError('Final fine-tuning LR must not exceed initial LR')
    if not 0 <= args.lr_hold_fraction < 1 or not args.temp_min < args.temp_ref < args.temp_max:
        raise ValueError('Invalid LR hold/temperature bounds')
    if not 0 < args.validation_fraction < .5 or not 0 < args.environment_init_quantile < 1:
        raise ValueError('Invalid validation fraction/environment quantile')
    if not Path(args.material_config).is_file():
        raise ValueError('Prepare --material_config with confirmed material emissivities first')
    if args.tv_neighbors < 1 or not 0 <= args.tv_material_boundary_weight <= 1:
        raise ValueError('Invalid TV graph/boundary weight')
    if not 0 <= args.material_confidence_threshold <= 1 or not 0 <= args.material_margin_threshold <= 1:
        raise ValueError('Invalid material confidence/margin')
    if not args.material_mask_dir or not Path(args.material_mask_dir).is_dir():
        raise ValueError('Prepare --material_mask_dir with material-label PNG masks first')
    if args.observation_domain == 'raw_rjpeg':
        directory = Path(args.radiometric_dir or 'radiometric')
        if not directory.is_absolute(): directory = Path(args.source_path) / directory
        if not (directory / 'manifest.json').is_file():
            raise ValueError('Prepare the raw RJPEG signal manifest in --radiometric_dir first')
    if not args.eval:
        raise ValueError('Pass --eval to reserve the published test cameras')
    output = Path(args.model_path).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new -m')
    for protected in (Path(args.source_path).resolve(), Path(args.geometry_model).resolve()):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Output must be separate from source data and Stage1 models')
    base.load_material_config(args.material_config)

def load_targets(args, cameras, device):
    if args.observation_domain == 'pseudocolor_rgb':
        for camera in cameras:
            candidates = [Path(args.source_path) / 'thermal' / split / (camera.image_name + ext)
                          for split in ('train', 'test') for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')]
            found = [p for p in candidates if p.is_file()]
            if len(found) != 1:
                raise ValueError('Missing/ambiguous thermal RGB target: ' + camera.image_name)
            with Image.open(found[0]) as image:
                image = image.convert('RGB')
                if image.size != (camera.image_width, camera.image_height):
                    old_ratio = image.width / image.height
                    if abs(old_ratio - camera.image_width / camera.image_height) > 1e-6:
                        raise ValueError('RGB/thermal image aspect ratios differ: ' + camera.image_name)
                    image = image.resize((camera.image_width, camera.image_height), Image.Resampling.BILINEAR)
                array = np.asarray(image).copy().astype(np.float32) / 255.
            camera.original_image = torch.from_numpy(array.transpose(2, 0, 1)).to(device)
            camera.original_physical_image = None
        return base.prepare_thermal_input(args, cameras, device)
    return base.prepare_thermal_input(args, cameras, device)

def target_image(camera, args, planck, device):
    if args.observation_domain == 'pseudocolor_rgb':
        return camera.original_image.to(device)
    if args.observation_domain == 'raw_rjpeg' and camera.original_physical_image is None:
        raise ValueError('Missing prepared physical signal for ' + camera.image_name)
    image = camera.original_physical_image if camera.original_physical_image is not None else camera.original_image
    scalar = base.observation_to_model_radiance(image.to(device).mean(0, keepdim=True), args, planck)
    return scalar.expand(3, -1, -1)

def predict(field, planck, gaussians, camera, sh, pipe, dataset, device):
    colors, residual, clipped = appearance(field, planck, gaussians, camera, sh)
    image = render(camera, gaussians, pipe, torch.zeros(3, device=device),
                   0., 0., 0., device, dataset.is_6dof, override_color=colors,
                   detach_geometry=False)['render']
    return image, residual, clipped

def data_loss(prediction, target, args, mask=None):
    if args.photometric_loss == 'mse': return masked_mean((prediction - target).square(), mask)
    if args.photometric_loss == 'l1_ssim': return image_loss(prediction, target, mask)
    return thermal_data_loss(prediction, target, args, mask)

@torch.no_grad()
def evaluate(field, planck, gaussians, sh, cameras, pipe, dataset, args, device):
    if not cameras: raise ValueError('Evaluation cameras are empty')
    rows = []
    for camera in cameras:
        prediction, _, clipped = predict(field, planck, gaussians, camera, sh, pipe, dataset, device)
        target = target_image(camera, args, planck, device)
        mask = getattr(camera, 'thermal_valid_mask', None)
        mse = float(masked_mean((prediction - target).square(), mask))
        row = {'loss': float(data_loss(prediction, target, args, mask)), 'MSE': mse,
               'PSNR': -10 * math.log10(max(mse, 1e-12)), 'SH_clipped_fraction': float(clipped)}
        if not all(math.isfinite(x) for x in row.values()): raise FloatingPointError('Non-finite validation')
        rows.append(row)
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}

def save_checkpoint(path, field, gaussians, sh, metadata, step):
    state = frozen_geometry_state(gaussians)
    protocol = dict(metadata['joint_protocol'])
    protocol['trained_geometry_sha256'] = geometry_digest(state)
    meta = dict(metadata, joint_protocol=protocol, stage2_common_endpoint=False)
    payload = field.export_state()
    payload.update(kind=KIND, step=int(step), metadata=meta,
                   joint_geometry=state, rgb_sh_residual=sh.export_state())
    temporary = str(path) + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)

def main():
    args, dataset, pipe, device = base.parse_args(add_arguments)
    validate_args(args)
    if device.type != 'cuda': raise RuntimeError('Joint Stage2 requires CUDA')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset.model_path, dataset.load_model_path = args.model_path, args.geometry_model
    # Only the raw/pseudocolor input adapters skip the legacy grayscale loader.
    # Raw targets MUST subsequently be loaded from a validated manifest.
    dataset.rgb_geometry_stage = args.observation_domain in ('raw_rjpeg', 'pseudocolor_rgb')
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians, load_iteration=args.geometry_iteration, shuffle=False)
    if scene.loaded_iter is None or not scene.has_rgbt: raise ValueError('A saved RGBT Stage1 model is required')
    args.geometry_sha256 = base.geometry_checkpoint_sha256(args.geometry_model, scene.loaded_iter)
    fit, validation = base.split_fit_validation_cameras(scene.getTrainCameras(), args.validation_fraction, args.validation_seed)
    test = scene.getTestCameras()
    if not fit or not validation or not test: raise ValueError('Nonempty fit/validation/test splits are required')
    args.fit_camera_names = [c.image_name for c in fit]
    args.validation_camera_names = [c.image_name for c in validation]
    args.test_camera_names = [c.image_name for c in test]
    for name in ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest', '_thermal_features_dc', '_thermal_features_rest'):
        getattr(gaussians, name).requires_grad_(False)
    planck = load_targets(args, fit + validation, device)
    materials = base.load_material_config(args.material_config)
    if abs(materials['temperature_reference_K'] - args.temp_ref) > 1e-6:
        raise ValueError('Material temperature reference differs from --temp_ref')
    args.material_config_document = materials['document']
    args.material_config_sha256 = base.file_sha256(args.material_config)
    support_weights = torch.zeros(len(gaussians.get_xyz), device=device)
    @torch.no_grad()
    def visibility(camera):
        package = render(camera, gaussians, pipe, torch.zeros(3, device=device), 0., 0., 0., device,
                         dataset.is_6dof, override_color=torch.ones_like(gaussians.get_xyz), detach_geometry=True)
        support_weights.add_(package['visibility_filter'].float() * gaussians.get_opacity.reshape(-1) * package['radii'].float().square())
        return package['visibility_filter'], package['radii']
    install_loss_scale(args, fit, lambda image: base.observation_to_model_radiance(image, args, planck))
    field = base.make_field(args, scene, device, planck, visibility, fit, materials)
    field.set_branch_trainability()
    sh = ThermalRGBSH(gaussians.get_xyz, support_weights > 0, args.rgb_sh_bound)
    count = len(gaussians.get_xyz)
    extent = float(scene.cameras_extent)
    if not math.isfinite(extent) or extent <= 0: raise ValueError('Invalid scene extent')
    groups = [
        {'name': 'temperature', 'params': [field.temperature_raw], 'start': args.temperature_lr, 'end': args.temperature_lr_final},
        {'name': 'environment', 'params': [field.environment_raw], 'start': args.environment_lr, 'end': args.environment_lr_final},
        {'name': 'rgb_sh', 'params': [sh.raw], 'start': args.rgb_sh_lr, 'end': args.rgb_sh_lr_final}]
    geometry = [('xyz', '_xyz', args.joint_position_lr * extent, args.joint_position_lr_final * extent),
                ('scaling', '_scaling', args.joint_scaling_lr, args.joint_scaling_lr * .01),
                ('rotation', '_rotation', args.joint_rotation_lr, args.joint_rotation_lr * .01),
                ('opacity', '_opacity', args.joint_opacity_lr, args.joint_opacity_lr * .01)]
    for name, attribute, start, end in geometry:
        parameter = getattr(gaussians, attribute)
        parameter.requires_grad_(True)
        groups.append({'name': name, 'params': [parameter], 'start': start, 'end': end})
    optimizer = torch.optim.Adam([dict(group, lr=group['start']) for group in groups])
    pair_i, pair_j, pair_w = build_spatial_tv_edges(gaussians.get_xyz, args.tv_neighbors) if args.lambda_tv else (
        torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, device=device))
    if pair_i.numel():
        boundary = field.material_ids[pair_i] != field.material_ids[pair_j]
        pair_w = pair_w * torch.where(boundary, args.tv_material_boundary_weight, 1.)
    anchor = field.environment.detach().clone()
    args.ir_opacity_protocol = {'enabled': False}
    args.sh_residual_protocol = {'enabled': False}
    metadata = base.checkpoint_metadata(args, scene, training_stage='stage2_joint_rgbsh',
        temperature_semantics='uncalibrated_pseudocolor_proxy' if args.observation_domain == 'pseudocolor_rgb' else
                              ('uncalibrated_model_proxy' if args.observation_domain == 'normalized_dn' else 'calibrated_model_estimate'),
        joint_protocol={'kind': KIND, 'gaussian_count': count, 'joint_start_step': 1,
            'densification': False, 'RGB_SH_frozen': True, 'material_emissivity_frozen': True,
            'material_assignment': 'fixed_Gaussian_identity_from_initial_fit_masks',
            'formula': 'RGB=repeat(epsilon*B(T)+(1-epsilon)*E,3)+SH_l1_l2_RGB',
            'channels': 3, 'coefficients_per_gaussian': 24, 'DC_trainable': False,
            'direction_geometry_gradient': True, 'color_clamp_before_blending': [0., 1.],
            'photometric_loss': args.photometric_loss, 'lambda_rgb': args.lambda_rgb,
            'rgb_sh_bound': args.rgb_sh_bound, 'lambda_rgb_sh': args.lambda_rgb_sh,
            'camera_model': {'sh_degree': dataset.sh_degree, 'is_6dof': dataset.is_6dof, 'images': dataset.images},
            'pipeline': {'compute_cov3D_python': pipe.compute_cov3D_python, 'convert_SHs_python': pipe.convert_SHs_python},
            'steps': args.steps, 'optimizer_groups': [{k: v for k, v in group.items() if k != 'params'} for group in groups]})
    output = Path(args.model_path); output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(field, planck, gaussians, sh, validation, pipe, dataset, args, device)
    best_loss, best_step = metrics['loss'], 0
    save_checkpoint(output / 'thermal_stage2_joint_best.pt', field, gaussians, sh, dict(metadata, validation_metrics=metrics), 0)
    initial_geometry = {attribute: getattr(gaussians, attribute).detach().clone() for _, attribute, _, _ in geometry}
    order = list(range(len(fit))); generator = random.Random(args.seed)
    print('[joint-stage2] ' + json.dumps(metadata['joint_protocol']), flush=True)
    with (output / 'training.jsonl').open('w') as log:
        log.write(json.dumps({'step': 0, 'validation': metrics}) + '\n')
        for step in range(1, args.steps + 1):
            if (step - 1) % len(order) == 0: generator.shuffle(order)
            camera = fit[order[(step - 1) % len(order)]]
            for group in optimizer.param_groups:
                group['lr'] = held_lr(step, args.steps, group['start'], group['end'], args.lr_hold_fraction)
            optimizer.zero_grad(set_to_none=True)
            rgb_value = 0.
            if args.lambda_rgb:
                rgb = render(camera, gaussians, pipe, torch.zeros(3, device=device), 0., 0., 0., device,
                             dataset.is_6dof, feature_set='rgb', detach_geometry=False)['render']
                rgb_loss = args.lambda_rgb * image_loss(rgb, camera.original_rgb_image)
                rgb_loss.backward(); rgb_value = float(rgb_loss.detach()); del rgb, rgb_loss
            prediction, residual, clipped = predict(field, planck, gaussians, camera, sh, pipe, dataset, device)
            target = target_image(camera, args, planck, device)
            photo = data_loss(prediction, target, args, getattr(camera, 'thermal_valid_mask', None))
            tv = prediction.new_zeros(())
            if pair_i.numel():
                temperature = field.temperature.reshape(-1)
                tv = (pair_w * (temperature[pair_i] - temperature[pair_j]).abs()).sum() / pair_w.sum().clamp_min(1e-12) / args.tv_temperature_scale_K
            environment = F.smooth_l1_loss(field.environment, anchor, beta=args.environment_prior_beta)
            loss = photo + args.lambda_tv * tv + args.lambda_environment * environment + args.lambda_rgb_sh * sh.regularizer()
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
            loss.backward()
            gradient_norms = {}
            for group in optimizer.param_groups:
                parameter = group['params'][0]
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError('Missing/non-finite gradient: ' + group['name'])
                gradient_norms[group['name']] = float(parameter.grad.norm())
            optimizer.step()
            for group in optimizer.param_groups:
                if not bool(torch.isfinite(group['params'][0]).all()):
                    raise FloatingPointError('Non-finite parameter after update: ' + group['name'])
            if len(gaussians.get_xyz) != count: raise RuntimeError('Gaussian identity/count changed')
            if step == 1 or step % 100 == 0 or step == args.steps:
                row = {'step': step, 'loss': float(loss.detach()), 'data_loss': float(photo.detach()),
                       'RGB_anchor_loss': rgb_value, 'SH_clipped_fraction': float(clipped.detach()),
                       'gradient_norms': gradient_norms}
                print(json.dumps(row), flush=True); log.write(json.dumps(row) + '\n'); log.flush()
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(field, planck, gaussians, sh, validation, pipe, dataset, args, device)
                row = {'step': step, 'validation': metrics, 'test_used_for_selection': False}
                print(json.dumps(row), flush=True); log.write(json.dumps(row) + '\n'); log.flush()
                if metrics['loss'] < best_loss:
                    best_loss, best_step = metrics['loss'], step
                    save_checkpoint(output / 'thermal_stage2_joint_best.pt', field, gaussians, sh,
                                    dict(metadata, validation_metrics=metrics), step)
            if step % args.save_every == 0:
                save_checkpoint(output / ('thermal_stage2_joint_step_%d.pt' % step), field, gaussians, sh, metadata, step)
    changes = {name: float((getattr(gaussians, name).detach() - original).abs().max()) for name, original in initial_geometry.items()}
    save_checkpoint(output / 'thermal_stage2_joint_final.pt', field, gaussians, sh, metadata, args.steps)
    summary = {'steps': args.steps, 'best_step': best_step, 'best_validation_loss': best_loss,
               'final_validation': metrics, 'geometry_max_abs_updates': changes,
               'coefficients_per_gaussian': 24, 'temperature_semantics': metadata['temperature_semantics'],
               'test_used_for_training_or_selection': False}
    (output / 'training_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('STAGE2_JOINT_RGBSH_COMPLETE ' + json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
