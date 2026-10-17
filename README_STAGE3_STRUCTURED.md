# Stage3: shared temperature controls and SH environment

The pre-change server source is backed up on GitHub at commit
`136a70772b47708c6d3a05155cff72bbf088ca19`, branch
`backup/stage3-before-lowdim-shenv-20261017`.

## Temperature

`T_i = clamp(T_D,i + sum_j w_ij * 5K*tanh(a_j) + 0.5K*tanh(r_i), 250K, 450K)`.

At most 4096 local controls share temperature updates across fit-supported
surface graph components. Four-neighbor interpolation stays within components
and material labels; small isolated components retain D temperature. Graph
connections also require oriented normal cosine >= 0.5. Unknown remains unknown,
and connectivity/spatial cells prevent treating all unknown points as one object.

D's initial temperature detail is a fixed offset. All control and local updates
start at zero, preserving D exactly at step 0. This is a reduced-dimensional
**temperature update** model, not a claim that D's temperature is ground truth.
The per-Gaussian local residual is still trainable, but bounded to +/-0.5 K and
penalized by mean `(residual / 0.1K)^2` with weight 1e-3. Controls have a weak
zero-update prior (1e-4); none of these scales were selected using test views.

## Environment

`E_i = E0 + confidence_i * sum_(l=1,2;m) c_lm Y_lm(n_i)`.
`L_i = epsilon_i B(T_i) + (1-epsilon_i) E_i`.

E0 and eight degree-1/2 scalar SH coefficients are shared across the scene:
9 environment parameters total. This models low-order outgoing diffuse reflected
environment signal as a function of normal; it is not a calibrated illumination
map or a model of specular reflection. No view-direction SH is substituted for
the normal-dependent term. Normals are frozen shortest Gaussian scale axes,
oriented toward the nearest **fit** camera. Confidence `1-s_min/s_middle`
suppresses directional response for nearly spherical Gaussians; these normals
are geometry proxies, not ground truth.

An SH coefficient ball bounds the directional contribution over all directions
to at most 0.01 normalized Q units, reduced automatically near the E0 endpoints
to keep E in (0,1). The coefficient prior has weight 1e-4. SH starts at zero;
E0 starts at D's environment value. Its fit-only initialization and frozen
geometry do not use held-out views.

The original per-Gaussian appearance SH (8 scalar degree-1/2 coefficients) remains
unchanged and frozen, outside the physical term. Geometry, opacity, RGB SH,
epsilon0, material IDs and split are also preserved. Only matte_plastic has
material coefficients enabled, consistently with the common checkpoint.

## Run all three branches

```bash
export GS_WORK=/home/liupengyu/graduate
CUDA_VISIBLE_DEVICES=0 STAGE3_RUN=RoadBlock_stage3_structured_run1 \
  bash "$GS_WORK/scripts/run_roadblock_stage3_structured.sh"
```

Use a new STAGE3_RUN if its output already exists. C/R/K use the same
temperature/environment model, graph, initialization, seed, budget and shared
hyperparameters, recorded in a comparison lock. Defaults: 10000 steps,
temperature control LR 3e-4->3e-6, local residual LR 1e-4->1e-6,
E0 LR 1e-4->1e-6, environment SH LR 1e-3->1e-5,
material LR 1e-4->1e-6, hold 0.1, material prior 1e-5,
spatial temperature prior 0.01. They are initial settings, not proven best values.
Use STRUCTURED_STEPS/STRUCTURED_T_LR/STRUCTURED_E_LR/etc. to override this
profile. Old search exports such as STAGE3_T_LR do not silently override the new
profile's learning rates. STAGE3_RUN still chooses the output prefix.

Each branch evaluates both best and final thermal checkpoints, and best RGB.
Best remains selected by validation signal+display loss, with step 0 eligible.
Checkpoints save all float parameters and frozen interpolation/normal buffers;
the evaluator dispatches by field_parameterization and never silently replaces
this model with the old per-Gaussian/global-environment field. Existing old
checkpoints and old launcher defaults retain their original model.

The changes do not resolve single-band temperature/emissivity ambiguity by
themselves and do not establish physical parameter accuracy without independent
truth. Evaluate improvement using matched C/R/K and ablations.

## Implementation verification (2026-10-17)

36 unit/regression tests passed. Real GPU C/R/K verification ran 56 steps each
in output/RoadBlock_stage3_structured_code_validation_20261017, using an identical
frozen structure hash across branches. The actual basis has 3210 controls,
126862 active Gaussians and 16912 retained at D temperature (unsupported or small
disconnected components). Step-zero validation matches D; control, local residual
and environment SH gradients are nonzero. Frozen state checks and exact CPU
checkpoint roundtrips passed. These short runs validate the implementation;
they are not full-budget evidence of improved reconstruction or temperature.
The standard thermal evaluator also loaded the nonzero K final checkpoint,
evaluated all 56 fit / 6 validation / 9 test views, and reproduced its training
validation PSNR within 1e-5 dB. Launcher checks verified identical C/R/K arguments,
best/final evaluation calls, and isolation from old search learning-rate exports.
