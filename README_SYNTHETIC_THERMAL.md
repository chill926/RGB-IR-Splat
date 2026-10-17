# Physical multi-view thermal benchmark

The sparse interpolation speed fix was verified and pushed first, in commit
`261106ce21f6e0eee4766e9c10eafc25e88cc0cf` on `stage3-sparse-speedup-20261017`.
This new benchmark does not modify the real RoadBlock model or its observations.

## Physical generation

The default scene has two differently oriented finite panels and an occluding
sphere. Each surface has known geometry, material, smooth temperature field,
epsilon0, temperature coefficient k, and normals. Coefficients describe controlled
synthetic materials, not literature values for specific real substances.

The generator independently integrates Planck's law over a normalized uniform
8–14 micrometer response. Its output is band-averaged spectral radiance, in
W m^-3 sr^-1 (not unnormalized band-integrated W m^-2 sr^-1). It calculates

```
epsilon(T) = epsilon0 + k*(T-300K)
E(n) = (1/pi) integral L_environment(w) max(n.w,0) dw
L_surface = epsilon(T)*B(T) + (1-epsilon(T))*E(n)
DN = gain*L_surface + offset + noise
```

The far-field illumination has a constant and degree-1/2 SH terms. Lambertian
hemisphere integration attenuates SH bands 1/2 by 2/3 and 1/4. There is no material
transmission, specular reflection or atmosphere in this controlled first version.
Temperature is a prescribed physical snapshot, not a heat-conduction simulation;
illumination is external, with inter-object multiple thermal reflections omitted.
Generation rejects unphysical emissivity rather than silently clipping it.

The default `analytic` backend traces perspective rays and intersects opaque
analytic geometry, selecting the nearest surface. It does not use the inverse
Gaussian renderer. RGB images show the scene/materials; thermal PNGs use a fixed
palette for preview. Training reads calibrated float DN arrays, never inversely
maps PNG colors to temperature. Noise and optional 16-bit quantization are explicit.

Forty cameras cover a front orbit with varying elevation. The deterministic
interleaved split has 24 fit, 8 validation and 8 test views. Gaussian surface
samples provide known frozen geometry/material labels to isolate thermal
inversion; this is not an end-to-end geometry recovery benchmark. Invalid
background rays and one pixel at each object boundary are masked. Full-frame
signal metrics are also reported to make the image coverage policy visible.

The analytic targets and Gaussian representation need not coincide perfectly.
With `--truth`, evaluation additionally renders the known physical parameters
through the Gaussian renderer, reporting `truth_parameter_render_signal_PSNR`.
This quantifies representation/response approximation error, not a mathematical
upper bound on arbitrary image fitting.

An optional `gaussian_matched` generator uses the same known physical radiances
but the Gaussian renderer to create observations. It is an explicitly labeled
matched-model implementation control, not independent evidence of real accuracy.
Matched observations retain the renderer's Gaussian coverage outside the analytic
silhouette; the analytic foreground mask only selects surface-evaluation pixels.
Use it first to test the known-T/E, k-only Oracle diagnostic.

## Data separation

Public training inputs:

- dataset.json: cameras, split, calibrated sensor, epsilon0 priors, hashes;
- geometry.npz: positions, scales, rotations, material IDs, normals, areas and
  fit-camera geometric visibility;
- observations/*.npz: measured float DN, valid mask and foreground mask.

Hidden truth for generation/final evaluation:

- truth/scene.json: all thermal generation quantities and environment lighting;
- truth/gaussians.npz: surface T/epsilon/E/k, all split visibility;
- truth/frames/*.npz: surface truth maps, depth, normals and clean observations.

Main C/R/K training calls only the public loader for fit/validation. It initializes
temperature to the public constant 300 K and E from fit-observation quantiles.
It does not use hidden truth for initialization, losses or checkpoint selection.
Test images are read by the evaluator only. Default epsilon0 priors are correct
as a declared known-input condition; use epsilon_bias for misspecification tests.

## Generate and train

```bash
export GS_WORK=/home/liupengyu/graduate
export THERMAL_PYTHON=/home/liupengyu/miniconda3/envs/3dgs/bin/python

# Independent analytic physical scene and float observations.
$THERMAL_PYTHON -B "$GS_WORK/tools/generate_thermal_synthetic.py" \
  --output "$GS_WORK/data/SyntheticThermal/physical_scene_v1"

# Same seed/hyperparameters/initialization for C, R, K; 2000 steps each.
CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_synthetic_thermal.sh"
```

The wrapper refuses existing branch outputs. Set SYNTH_DATA, SYNTH_RUN,
SYNTH_STEPS or SYNTH_SEED to change a run. Best is selected only by validation
image objective, with step 0 eligible. Both best/final are evaluated afterward.
The comparison lock rejects different shared settings among C/R/K.

Fit/validation scalar loss is fit-contrast-scaled SmoothL1, with optional fixed
palette L1 (weight 0 by default). All branches use the same shared temperature
controls plus bounded residual and global SH2 E; priors are recorded separately.
epsilon0 remains fixed, R learns material delta-epsilon, K learns material k.
The synthetic physics control has no free appearance SH to mask the physical
terms; the existing RoadBlock appearance SH remains unchanged.

## Explicit Oracle implementation control

```bash
CUDA_VISIBLE_DEVICES=0 $THERMAL_PYTHON -B "$GS_WORK/tools/generate_thermal_synthetic.py" \
  --output "$GS_WORK/data/SyntheticThermal/matched_scene_v1" --renderer gaussian_matched
CUDA_VISIBLE_DEVICES=0 $THERMAL_PYTHON -B -u "$GS_WORK/tools/train_thermal_synthetic.py" \
  --dataset "$GS_WORK/data/SyntheticThermal/matched_scene_v1" \
  --output "$GS_WORK/output/SyntheticThermal_oracle1/K" --branch K --oracle --steps 2000
CUDA_VISIBLE_DEVICES=0 $THERMAL_PYTHON -B "$GS_WORK/tools/evaluate_thermal_synthetic.py" \
  --checkpoint "$GS_WORK/output/SyntheticThermal_oracle1/K/best.pt" --truth --save_images
```

Oracle explicitly supplies true T/E/epsilon0, freezes them and fits k only, with
no k shrinkage penalty or material warmup. It is not the main unsupervised
physical inversion result and does not establish joint identifiability.

## Controlled cases and interpretation

Generator options support `--k_scale 0`, `--temperature_span 2`,
`--epsilon_bias 0.03`, `--noise_std_dn 2 --quantize`, with fresh output names.
Change one factor at a time; repeat seeds for the optimizer before drawing a
physical conclusion. Do not expect a unique T/k/E decomposition merely because
the data are synthetic. Temperature MAE/RMSE, emissivity error, k error and E
error are the physical metrics. IR PSNR/SSIM and calibrated signal errors describe
image reconstruction. All/fit-visible/test-visible surface errors are separate.

Use tools/train_thermal_synthetic.py for one branch and
tools/evaluate_thermal_synthetic.py for evaluation. Synthetic checkpoints use a
separate metadata kind; do not feed them to the RoadBlock-specific launcher.

## Verified implementation

51 unit/regression checks passed across the existing physics/Stage3, sparse
interpolation and synthetic modules. C/R/K each ran 500 steps on the independent
40-view analytic dataset, with identical comparison protocols and separate
best/final image and physical-truth evaluation. They are pipeline pilots, not
converged scientific comparisons. A training copy with no truth directory also
ran successfully, and the public loader test rejects truth/test reads.

A 1000-step k-only Oracle on a small matched-renderer dataset recovered
[-1e-4, 2e-4, 1e-4]/K as approximately
[-1.00789e-4, 1.99643e-4, 0.99414e-4]/K, with absolute errors below 8e-7/K.
This verifies the physical coefficient path under known T/E/epsilon0; it does
not prove that main C/R/K can jointly identify the unknown parameters.

Existing generated example data are at data/SyntheticThermal/physical_scene_v1;
implementation pilots are under output/SyntheticThermal_code_validation_20261017.
Use a fresh output root for the default 2000-step formal run. Synthetic data,
truth and checkpoints are excluded from GitHub by the existing data/output rules.
