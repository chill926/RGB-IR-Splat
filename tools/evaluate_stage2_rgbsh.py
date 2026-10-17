"""Evaluate joint Stage2 RGB SH24 using its trained geometry and appearance."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchvision.utils import save_image
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from train_thermal_stage2_rgbsh import load_targets, evaluate, predict, target_image
from train_thermal_physics import geometry_checkpoint_sha256
from utils.thermal_rgb_sh import KIND, ThermalRGBSH, install_geometry
from utils.thermal_physics import MaterialThermalField

@torch.no_grad()
def main():
    device = torch.device('cuda:0')
    parser = argparse.ArgumentParser(description=__doc__)
    model_params = ModelParams(parser, device)
    pipeline_params = PipelineParams(parser)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--geometry_model', default='')
    parser.add_argument('--radiometric_dir', default='')
    parser.add_argument('--split', choices=('fit', 'validation', 'test', 'all'), default='validation')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--save_images', action='store_true')
    cli = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    checkpoint_path = Path(cli.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('kind') != KIND: raise ValueError('Expected joint Stage2 RGB SH24 checkpoint')
    metadata = checkpoint['metadata']
    args = SimpleNamespace(**metadata['shared_training_protocol'])
    args.photometric_loss = metadata['joint_protocol']['photometric_loss']
    args.source_path = str(Path(cli.source_path or metadata['source_path']).resolve())
    args.fit_camera_names = metadata['camera_split']['fit']
    if cli.radiometric_dir: args.radiometric_dir = cli.radiometric_dir
    geometry_model = str(Path(cli.geometry_model or metadata['geometry_model']).resolve())
    iteration = int(metadata['geometry_iteration'])
    if geometry_checkpoint_sha256(geometry_model, iteration) != metadata['geometry_sha256']:
        raise ValueError('Stage1 initializer checksum differs from the recorded model')
    dataset = model_params.extract(cli)
    dataset.source_path = args.source_path
    dataset.resolution = args.resolution
    for key, value in metadata['joint_protocol']['camera_model'].items(): setattr(dataset, key, value)
    dataset.data_branch, dataset.eval = 'rgbt', True
    dataset.model_path = dataset.load_model_path = geometry_model
    dataset.rgb_geometry_stage = args.observation_domain in ('raw_rjpeg', 'pseudocolor_rgb')
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    install_geometry(gaussians, checkpoint)
    field = MaterialThermalField.from_checkpoint(checkpoint, 'stage2', device, reset_branch_parameters=False)
    field.requires_grad_(False)
    sh = ThermalRGBSH.from_checkpoint(checkpoint, gaussians.get_xyz)
    available = {camera.image_name: camera for camera in scene.getTrainCameras() + scene.getTestCameras()}
    if len(available) != len(scene.getTrainCameras()) + len(scene.getTestCameras()):
        raise ValueError('Duplicate camera names')
    splits = ('fit', 'validation', 'test') if cli.split == 'all' else (cli.split,)
    names = [name for split in splits for name in metadata['camera_split'][split]]
    if not names or len(names) != len(set(names)) or any(name not in available for name in names):
        raise ValueError('Missing or duplicate recorded evaluation cameras')
    cameras = [available[name] for name in names]
    planck = load_targets(args, cameras, device)
    pipe = pipeline_params.extract(cli)
    for key, value in metadata['joint_protocol'].get('pipeline', {}).items(): setattr(pipe, key, value)
    metrics = evaluate(field, planck, gaussians, sh, cameras, pipe, dataset, args, device)
    output = Path(cli.output_dir).resolve() if cli.output_dir else checkpoint_path.parent / ('evaluation_' + checkpoint_path.stem + '_' + cli.split)
    for protected in (Path(args.source_path), Path(geometry_model)):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Evaluation output must be separate from data and Stage1 models')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(metrics=metrics, checkpoint=str(checkpoint_path), step=checkpoint['step'], split=cli.split,
                  camera_names=names, observation_domain=args.observation_domain,
                  temperature_semantics=metadata['temperature_semantics'], coefficients_per_gaussian=24,
                  geometry_evaluated='checkpoint_joint_geometry',
                  geometry_sha256=metadata['joint_protocol']['trained_geometry_sha256'])
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    if cli.save_images:
        image_dir = output / 'images'; image_dir.mkdir(exist_ok=True)
        for camera in cameras:
            prediction, _, _ = predict(field, planck, gaussians, camera, sh, pipe, dataset, device)
            target = target_image(camera, args, planck, device)
            save_image(torch.cat((target, prediction.clamp(0, 1), (prediction - target).abs()), dim=2),
                       image_dir / (camera.image_name + '.png'))
    print('STAGE2_RGBSH_EVALUATION ' + json.dumps(report), flush=True)

if __name__ == '__main__': main()
