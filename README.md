# PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting

This repository is organized around the CVPR 2026 paper **"PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting"**.

PhysIR-Splat models thermal infrared image formation on Gaussian primitives. Each primitive can carry radiative-transfer attributes such as temperature, emissivity, and environmental irradiance.

## Method Overview

The paper describes two coupled components:

- **PhysIR-Splat**: a thermal Gaussian rendering framework based on passband Planck radiance, environmental reflection, atmospheric transmittance, and monotonic radiometric response.
- **VGGT-IR**: a Transformer-based thermal infrared initializer that directly regresses camera poses, depth, and point maps from multi-view TIR input with optional RGB.

## Dataset

The training examples use TI-NSD scenes organized as:

```shell
data/
  TI-NSD/
    apples/
    basketball_court/
    ...
```

The split follows the paper setup: with `--eval`, every 8th image is used as the test set and the remaining images are used for training.

RGB+thermal scenes are also supported when organized like RGBT-Scenes:

```shell
scene_name/
  sparse/0/                 # or colmap/sparse/0/
  rgb/
    train/
    test/
  thermal/
    train/
    test/
```

## Environment

```shell
conda create -n physir python=3.7
conda activate physir

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

If you already use an existing environment, keep it active and only install missing dependencies.

## Training

```shell
python train.py -s path/to/your/TI-NSD/scene_name --eval
```

RGB+thermal:

```shell
python train.py -s path/to/your/RGBT-Scenes/scene_name --eval
```

For RGBT training, the code keeps a mirrored copy of the best evaluated
thermal checkpoint as `iteration_30001`. This is used only by the RGBT branch;
TI-NSD IR-only training keeps its existing behavior.

During training, validation PSNR/L1 is computed on save-image-compatible
8-bit tensors. This matches `metrics.py` on rendered PNGs and is used only for
logging/checkpoint selection; it does not enter the optimization loss.

Select a data branch explicitly:

```shell
python train.py -s path/to/your/TI-NSD/scene_name --eval --data_branch ir
python train.py -s path/to/your/RGBT-Scenes/scene_name --eval --data_branch rgbt
```

`--data_branch auto` is the default and keeps automatic scene detection.

The default schedule trains for 30k iterations.

### Temperature-dependent emissivity training

Stage 1 uses only RGB observations and jointly optimizes the standard RGB 3DGS
geometry variables (position, rotation, scale, and opacity) and RGB SH appearance
(`f_dc`/`f_rest`). Thermal SH appearance, thermal correction, temporal deformation,
and physical parameters are frozen:

```shell
python train.py -s data/RGBT-Scenes/scene -m output/scene_stage1 --data_branch rgbt \
  --eval --rgb_geometry_stage --iterations 30000
```

Stage 1 follows the original RGB 3DGS optimization and density-control schedule.
Validation is diagnostic only: it does not select a checkpoint or stop training.
With the command above, the shared geometry checkpoint is saved at iteration
30000.

Then freeze this checkpoint and train the constant-emissivity stage 2:

```shell
python train_thermal_physics.py -s data/RGBT-Scenes/scene -m output/scene_stage2 \
  --eval \
  --stage stage2 --geometry_model output/scene_stage1 --geometry_iteration -1 \
  --material_mask_dir data/RGBT-Scenes/scene/material_masks \
  --material_config material_config.json --steps 10000
```

The optional B0 control fits thermal images with unconstrained thermal SH on the
same frozen geometry and explicitly marks its output as having no physical
temperature meaning:

```shell
python train_thermal_image_baseline.py -s data/RGBT-Scenes/scene -m output/scene_B0 \
  --eval --geometry_model output/scene_stage1 --geometry_iteration -1 --steps 5000
```

A grayscale material label image stores integer ids from `material_config.json`;
255 is unknown by default. Put masks under `material_masks/`,
`material_masks/train/`, or `material_masks/test/`. Multi-view projection keeps
only labels that pass both confidence and winner-margin thresholds. Unseen or
ambiguous Gaussians remain unknown rather than being forced into a catch-all
non-metal class. `--allow_missing_material_masks` is for smoke tests only.

Stage 2 optimizes per-Gaussian temperature and one global environmental irradiance while
keeping geometry and epsilon_0 fixed. It writes `thermal_stage2_best.pt` using
an internal split of the published training cameras and stops only when validation reaches a plateau and the
updates of both T and E are stable.
Its environment parameterization and prior follow the repository's simplified
physical baseline: one sigmoid-bounded scene scalar, initialized from the
0.1 observation quantile and Huber-anchored with beta 0.02 and weight 0.01.
If these two conditions are not met by `--steps`, the diagnostic final checkpoint
is retained but `thermal_stage2_common.pt` is not created, so C/K/R cannot start
from a falsely declared common endpoint.

The stage-1 endpoint is always mirrored to an iteration newer than the optional
best-PSNR snapshot, so `--geometry_iteration -1` resolves to the actual stable
stage endpoint. Stage 2 writes `thermal_stage2_common.pt` from its actual stopping
state; C, K, and R must use this file. `thermal_stage2_best.pt` is diagnostic only.

Run `tools/audit_rgbt_thermal_data.py` before choosing the observation domain.
RGBT PNG/JPEG observations without a reversible temperature mapping must remain
`normalized_dn`; this mode is qualitative and must not be reported as an
absolute temperature. If calibration supplies the physical 8--14 um
band radiance at DN=0 and DN=1, enable calibrated training:

```shell
python train_thermal_physics.py ... --observation_domain calibrated_radiance \
  --dn0_radiance RADIANCE_AT_DN_0 --dn1_radiance RADIANCE_AT_DN_1
```

For a raw integer image with a documented apparent-temperature mapping use:

```shell
python train_thermal_physics.py ... --observation_domain apparent_temperature \
  --thermal_raw_max 65535 --temperature_scale SCALE_K_PER_UNIT \
  --temperature_offset OFFSET_K
```

For synthetic data, optional per-Gaussian temperature truth can be shared by
all branches with `--temperature_gt temperatures.npy` and
`--lambda_temperature_supervision WEIGHT`; MAE and RMSE in Kelvin are then logged.

Start C, K and R from the exact same stage-2 checkpoint, with identical `--steps`
and `--seed`. Formal runs must share one comparison lock file. K/R additionally
must consume the regularization values selected on synthetic validation:

```shell
python train_thermal_physics.py -s data/RGBT-Scenes/scene -m output/scene_C --eval --stage branch --branch C --geometry_model output/scene_stage1 --stage2_checkpoint output/scene_stage2/thermal_stage2_common.pt --comparison_protocol output/comparison_protocol.json --regularization_protocol output/regularization_scan/regularization_scan.json --steps 5000 --seed 0
python train_thermal_physics.py -s data/RGBT-Scenes/scene -m output/scene_K --eval --stage branch --branch K --geometry_model output/scene_stage1 --stage2_checkpoint output/scene_stage2/thermal_stage2_common.pt --comparison_protocol output/comparison_protocol.json --regularization_protocol output/regularization_scan/regularization_scan.json --steps 5000 --seed 0
python train_thermal_physics.py -s data/RGBT-Scenes/scene -m output/scene_R --eval --stage branch --branch R --geometry_model output/scene_stage1 --stage2_checkpoint output/scene_stage2/thermal_stage2_common.pt --comparison_protocol output/comparison_protocol.json --regularization_protocol output/regularization_scan/regularization_scan.json --steps 5000 --seed 0
```

Before the synthetic validation set exists, exploratory RGBT code tuning may
pass `--lambda_k` and `--lambda_delta_epsilon` directly. Such runs are logged as
exploratory and must use only the internal validation split; do not select these
values from the published RGBT test split.

The physical renderer uses a uniform 8--14 um Planck passband and blends
radiance. C keeps fixed emissivity, K learns a temperature coefficient per
material, and R learns an equal-count constant emissivity residual. Geometry and
base emissivity are frozen in all three branches.

In the thesis naming, B0 is `train_thermal_image_baseline.py`, B1 is branch C,
B2 is branch R, and B3 is branch K.

To run the complete training sequence with the documented budgets:

```shell
bash scripts/run_thesis_thermal_training.sh data/RGBT-Scenes/scene output/scene_experiment \
  data/RGBT-Scenes/scene/material_masks material_config.json
```

### SAM2 material masks

The official Meta SAM2 source is vendored under `third_party/sam2`. A human assigns
each prompted object a material name; SAM2 segments the region but does not infer
its material. Unprompted pixels remain unknown. SAM2 requires Python
3.10+ and PyTorch 2.5.1+, while this PhysIR baseline pins older PyTorch, so generate
masks in a separate environment and consume the PNG masks in the training environment:

```shell
conda create -n sam2 python=3.10
conda activate sam2
# Install a CUDA-compatible PyTorch >=2.5.1 first.
pip install -e third_party/sam2
bash scripts/download_sam2_checkpoint.sh
```

Create `material_config.json` from `material_config.example.json`, inspect the
surface condition and source for every fixed epsilon_0, and set `confirmed` to
true. `tools/material_emissivity_lookup.py` can create an unconfirmed candidate
template. Then create `material_prompts.json` and generate label masks:

```shell
python tools/generate_sam2_metal_masks.py \
  --image_dir data/RGBT-Scenes/scene/rgb/train \
  --prompts material_prompts.json \
  --material_config material_config.json \
  --output_dir data/RGBT-Scenes/scene/material_masks/train \
  --checkpoint third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt
```

Material masks and thermal observations are mapped to Gaussians using multi-view
probability averaging, the 3DGS rasterizer visibility test, and a full-resolution
projected-centre depth test that rejects back surfaces.

For an ordered capture sequence, add a stable `object_id` to each prompted
instance and use `tools/generate_sam2_material_video_masks.py`. Prompts are only
needed on keyframes and may be repeated later to correct tracking drift.

### Physical-field rendering and conditional UQ

Render radiance, apparent temperature, emissivity, material id and material
confidence from a trained branch without changing the checkpoint:

```shell
python render.py -s data/RGBT-Scenes/scene -m output/scene_stage1 --data_branch rgbt \
  --mode physical_fields --thermal_checkpoint output/scene_K/thermal_K_final.pt \
  --physical_output output/scene_K/physical_fields
```

For repeated seeds, estimate conditional uncertainty with:

```shell
python tools/estimate_conditional_uq.py output/seed_*/thermal_K_final.pt \
  --output output/K_conditional_uq.pt
```

This spread is conditional on frozen geometry/cameras, material masks, fixed
epsilon_0, observation conversion, regularization and the implemented forward
model. It is not total physical uncertainty.

### Material-prior regularization scan

Do not select lambda values on a real test scene. On a synthetic validation
scene, scan K and R regularization with repeated seeds:

```shell
python tools/scan_material_regularization.py \
  --source data/synthetic/validation_scene \
  --geometry_model output/synthetic_stage1 \
  --stage2_checkpoint output/synthetic_stage2/thermal_stage2_common.pt \
  --output_root output/regularization_scan \
  --confirm_synthetic_validation
```

The generated `regularization_scan.json` contains validation means, standard
deviations, material-parameter stability, and the selected K/R values. Freeze
those values for all subsequent scenes. Formal C/K/R commands reject missing or
non-synthetic scan protocols. The complete script also writes
`thermal_branch_comparison.json`, which validates equal budgets/seeds and reports
training loss, validation radiance error, available temperature error, and
parameter stability for C/K/R.

## Rendering And Evaluation

```shell
python render.py -m output/exp-name
python metrics.py -m output/exp-name
```

## Acknowledgements

We thank the authors of the following thermal 3D Gaussian Splatting works for their inspiring contributions:

```bibtex
@inproceedings{luthermalgaussian,
  title={ThermalGaussian: Thermal 3D Gaussian Splatting},
  author={Lu, Rongfeng and Chen, Hangyu and Zhu, Zunjie and Qin, Yuhang and Lu, Ming and Yan, Chenggang and others},
  booktitle={The Thirteenth International Conference on Learning Representations}
}

@inproceedings{chen2024thermal3d,
  title={Thermal3d-gs: Physics-induced 3d gaussians for thermal infrared novel-view synthesis},
  author={Chen, Qian and Shu, Shihao and Bai, Xiangzhi},
  booktitle={European Conference on Computer Vision},
  pages={253--269},
  year={2024},
  organization={Springer}
}

@inproceedings{nam2025veta,
  title={Veta-GS: View-dependent deformable 3D Gaussian Splatting for thermal infrared Novel-view Synthesis},
  author={Nam, Myeongseok and Park, Wongi and Kim, Minsol and Hur, Hyejin and Lee, Soomok},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)},
  pages={965--970},
  year={2025},
  organization={IEEE}
}
```

## Paper Reference

```bibtex
@inproceedings{gao2026physir,
  title={PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting},
  author={Gao, Jingyuan and Hu, Yumeng and Gao, Fei and Zhang, Mingjin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11818--11828},
  year={2026}
}
```
