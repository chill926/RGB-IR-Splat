# Gα：受限的 IR opacity 适配

本轮只解锁独立的 IR opacity 修正。RGB 的位置、尺度、旋转、opacity、SH、相机和 Gaussian 数量继续冻结。
从已有 `thermal_stage2_common.pt` 的温度/环境/材料状态开始，不重新标注、不重新生成 Stage1 或 radiometric 数据。
Gα 仍是 `k=0` 的公共初始化阶段，不是 K 分支。

## 上传文件

将更新包解压到服务器 graduate 根目录，保留相对路径。修改：

1. `gaussian_renderer/__init__.py`
2. `utils/thermal_physics.py`
3. `train_thermal_physics.py`
4. `train_thermal_gray_control.py`
5. `tools/evaluate_thermal_physics.py`
6. `tools/recolor_thermal_renders.py`

新增：`utils/thermal_opacity.py`、`tools/run_galpha.py`、`tools/test_thermal_opacity.py`、本说明。
这是 Python 接口修改，不需要重新编译已有 CUDA 光栅化扩展。服务器仍需原 physir CUDA/3DGS 环境及上一轮辐射/伪彩更新。

## 运行

沿用服务器 GS_WORK：

```bash
CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/tools/run_galpha.py" \
  --reference "$GS_WORK/output/RoadBlock_stage2_retry_tv0/thermal_stage2_common.pt" \
  --output "$GS_WORK/output/RoadBlock_Galpha_v1"
```

入口会强制执行 CPU Torch 梯度、保存/读取、冻结和光栅化 Python 接口测试，然后开始 GPU 训练。
默认跑满 10000 步，选内部验证数据损失最优的 checkpoint（含第 0 步候选），
并自动标定显示映射、评价 fit/validation/test 的原始伪彩 RGB。输出目录必须是新的或空的。

如果先检查真实 CUDA 反向传播，把输出目录改成 `RoadBlock_Galpha_smoke` 并加 `--steps 5`。
这个入口会继续完成评估；5 步只能用于运行检查，不能用来判断性能。
本地后续不执行训练或测试；真实 CUDA 验证由你在服务器完成。

## 默认约束和超参数

IR opacity 修正为：

```
delta_logit = 0.2 * tanh(raw) * fit_support
alpha_IR ≈ sigmoid(logit(alpha_RGB) + delta_logit)
```

实现对零修正做了数值抵消，确保初始化保持原 opacity；只有进入 fit 视锥且 RGB opacity 非零的点开放修正。
fit 支持来自 RGB opacity × 投影半径平方，是潜在可见性筛选，不是精确的像素贡献或表面可见性证明。
修正与视角无关。0.2 的 logit 上限对应最多约 0.05 的绝对 opacity 变化；高/低 opacity 区域的实际变化更小。

| 参数 | 默认值 |
|---|---:|
| opacity logit 上限 | 0.2 |
| opacity 学习率 | 1e-3 → 1e-5 |
| opacity 先验权重 | 0.01 |
| 温度学习率 | 1e-4 → 1e-6 |
| 温度 TV 权重 | 0.001 |
| TV 的温度尺度 | 继承参考 checkpoint，上一轮为 10 K |
| 环境学习率、损失尺度、Huber 阈值 | 继承参考 checkpoint |
| 训练预算 | 10000 步 |

opacity 先验惩罚 fit 支持点上 `tanh(raw)^2` 的均值，相当于惩罚使用修正预算的比例。
以上为有界适配的试验起点，不是已经证明的最优超参数。
支持覆盖率、修正均值/最大值/P95、达到 tanh 预算 95% 的比例和数据梯度范数都会写入日志。

## 同设置的 G0 对照

Gα 同时恢复了弱 TV，并降低了温度学习率，不能把相对上一轮的所有提升都归因于 opacity。
需要隔离 opacity 的作用时，从同一个原参考运行以下对照，其余设置完全一致：

```bash
CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/tools/run_galpha.py" \
  --reference "$GS_WORK/output/RoadBlock_stage2_retry_tv0/thermal_stage2_common.pt" \
  --output "$GS_WORK/output/RoadBlock_G0_matched_v1" --mode frozen
```

也可以给 Gα 和这个 G0 同时加 `--lambda_tv 0`，只观察 opacity 适配。
原参考应使用上一轮未适配 opacity 的 Stage2；若以已有 Gα 作参考，frozen 模式会保留那份适配后的 opacity。

## 查看结果

脚本结束会直接打印三组 PSNR/SSIM。查看已有结果：

```bash
GALPHA="$GS_WORK/output/RoadBlock_Galpha_v1"
python - "$GALPHA/evaluation_pseudocolor/metrics.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    report = json.load(f)
for split, m in report['splits'].items():
    print(f"{split:10s} PSNR={m['PSNR']:.4f} dB SSIM={m['SSIM']:.6f}")
print('opacity:', report.get('ir_opacity_diagnostics'))
PY
```

- `training_summary.json`：停止原因、best_step、最优与最终 opacity 修正统计。
- `thermal_training.jsonl`：损失、opacity 梯度、修正量和饱和比例。
- `evaluation_pseudocolor/validation/comparisons/`：原图、预测、误差。
- `evaluation_pseudocolor/metrics.json`：伪彩和物理观测误差。

若 best_step=0，说明适配后没有更好的验证候选；保留初始模型，不会把最后一帧强行报告为最优。
如果修正大量达到上限，结合对照和误差图判断是否需要扩大预算，不仅凭 PSNR 决定。

## checkpoint 与后续 C/R/K

每份 checkpoint 的 `ir_opacity` 字段保存基准 RGB opacity、fit 支持、修正 raw 和上限，
温度与 opacity 在同一训练步保存。评估自动使用这一份 IR opacity，RGB 评估仍用原 RGB opacity。
没有 `ir_opacity` 的旧 checkpoint 自动沿用原冻结 opacity，渲染行为不变。

后续 C/R/K 从 Gα 目录的 `thermal_stage2_common.pt` 开始，自动加载并冻结 IR opacity 修正；
三个物理分支复用同一 RGB 几何和同一 IR opacity。参考 checkpoint 的内容 hash 继续锁定共同起点。
自由信号对照也会加载并冻结参考 IR opacity，而不会错误退回 RGB opacity。
通用 `render.py` 不是此物理 checkpoint 的入口，请使用 `tools/evaluate_thermal_physics.py`。

recolor 的相机信号 RMSE 汇总已改为先合并平方误差再开方，与直接评估保持一致。
这项修复不改变 PSNR/SSIM，也不改变任何渲染数组。

## 验证边界

Gα 的真实 CUDA 光栅化/反向传播尚未验证，服务器入口会先执行全部 9 项 Gα CPU 检查，缺少 Torch 时拒绝训练。
其中光栅化接口测试使用可微分的 CPU 替身，真实 CUDA 集成仍需上述短步运行检查。
独立 IR opacity 是渲染适配参数；图像指标提升不能单独证明真实几何或真实温度更准确。
