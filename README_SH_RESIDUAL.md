# 从头训练 Stage2：二阶 SH 方向残差

本实验只复用 Stage1 RGB 几何、材料标注和原始辐射输入。温度、环境项和 SH 全部重新初始化，
不接着旧 Stage2 的权重训练，也不复用 Gα 的 IR opacity 修正。旧 checkpoint 在运行脚本中仅提供数据、
温度范围、材料、学习率、正则及划分设置。默认旧训练入口的 `--sh_residual_degree 0` 保留原基线。

## 模型

```
q_i(view) = clamp(q_physical_i + sum_{l=1..2,m} a_i,lm * Y_lm(direction), 0, 1)
direction = normalize(Gaussian_xyz - camera_center)
```

单波段信号使用每个 Gaussian **8 个非 DC 系数**，不含 `+0.5` 颜色偏置。
结果复制到光栅化接口的三个通道；不是独立训练三组 RGB 系数，也不是直接拟合伪彩。
先合成信号，之后再应用已有的固定伪彩映射。

SH 从零初始化；默认第 3000 步开始优化，此前只优化物理主干，且不累积 SH Adam 动量。
开放后温度、环境和 SH 联合训练，k 仍为 0。位置、尺度、旋转、RGB opacity/SH、相机及 Gaussian 数量冻结。
只有 fit 视锥内且 RGB opacity 非零的点能学习残差；这只是潜在可见性筛选。

系数通过平滑 L2 球限制：`a = b * raw / sqrt(C * (1 + ||raw||²))`，其中 `C = 8/(4π)`。
由 SH 加法定理，任意单位方向上的绝对残差不超过 b。限制系数而不是对 SH 求值做 tanh，
保证加到主干之前仍是纯二阶非 DC SH。最终信号 clamp 会改变完整函数的球面均值，因此记录截断比例。
无 DC 只能减少与基础亮度的竞争，不能保证温度/发射率唯一可辨识；本分支是数据驱动的辐射失配近似。

| 参数 | 默认值 |
|---|---:|
| 训练步数 | 30000 |
| SH 开放步数 | 3000 |
| b / fit 损失尺度 | 0.5 |
| SH raw 学习率 | 0.01 → 0.0001 |
| SH 正则权重 | 0.001 |

SH 正则惩罚平均系数预算使用比例的平方。损失尺度是同一 fit 子集的 P99−P1（新 raw_rjpeg 协议），
上限 0.5 表示半个该信号跨度，不是 0.5 K、0.5 DN 或固定温度修正。旧协议沿用其损失尺度。
这些是实验起点，尚无实际 RoadBlock 收益验证。温度 LR、TV、环境正则等由参考配置继承。

## 上传与训练

将本更新包按相对路径覆盖服务器 graduate 目录；原始修改文件另有备份包。
不需要重编译 CUDA 扩展。沿用服务器环境的 GS_WORK，把 REFERENCE 改成你的实际路径。

```bash
REFERENCE="$GS_WORK/output/RoadBlock_stage2_retry_tv0/thermal_stage2_common.pt"

CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/tools/run_sh_residual.py" \
  --reference "$REFERENCE" \
  --output "$GS_WORK/output/RoadBlock_stage2_SH2_fresh_v1"
```

它重新从观测初始化 Stage2，而不是加载 REFERENCE 的物理权重。输出目录必须是新的或空的。
若参考路径迁移，加 `--scene`、`--geometry_model`、`--radiometric_dir`、`--material_config`、
`--material_mask_dir` 指定实际目录；数据、几何和材料配置内容仍须匹配。
脚本运行 CPU preflight，训练后自动评价 fit/validation/test 的伪彩与信号误差，以及关闭 SH 的物理主干结果。

建议同时运行 **匹配的从头训练基线**，与 SH 实验使用同一个参考和相同预算：

```bash
CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/tools/run_sh_residual.py" \
  --reference "$REFERENCE" --degree 0 \
  --output "$GS_WORK/output/RoadBlock_stage2_noSH_fresh_v1"
```

两组默认继承相同物理超参数；若改 `--lambda_tv`，两组都要修改。
默认复用参考目录的 `pseudocolor_calibration.json`；没有该文件时各自仅从 fit 数据生成映射。
要强制复用同一映射，两组都传 `--display_calibration "实际路径/pseudocolor_calibration.json"`。
脚本不使用 validation/test 像素拟合显示映射；checkpoint 由内部验证信号 Huber 选择，测试集只用于最终评价。

真实 CUDA 短步检查：另选输出目录并添加 `--steps 5 --start_step 2`。
它也会执行评估，5 步不能用于性能判断。本地 CPU 测试不等价于真实 CUDA 反向传播验证。

直接调用训练入口时，在原 Stage2 命令末尾添加：

```bash
--sh_residual_degree 2 --sh_residual_start_step 3000 \
--sh_residual_bound 0.5 --sh_residual_lr 0.01 \
--sh_residual_lr_final 0.0001 --lambda_sh_residual 0.001
```

这里需要 `--stage stage2` 和一个新 `-m` 目录，不传 `--stage2_checkpoint`；SH 实验使用 budget 停止模式。

## 查看结果

- `evaluation_pseudocolor/metrics.json`：完整模型的三组伪彩 PSNR/SSIM 和原始信号误差。
- `evaluation_physical_only/metrics.json`：同一已训练模型关闭 SH 后的结果；它不是独立训练的无 SH 基线。
- `evaluation_pseudocolor/validation/comparisons/`：GT、预测、绝对误差。
- `training_summary.json`：最优步数和 SH 预算使用统计；best_step=0 意味着没有更好的验证候选。
- `thermal_training.jsonl`：SH 梯度、残差幅度、预算饱和与信号截断比例。
- `sh_experiment_plan.json`：参考仅用于配置的记录与实际训练命令。

若 SH 预算大量接近上限，可另开实验增大 `--bound`；先看验证误差，不只看训练集提升。
检查信号指标与伪彩指标是否同时改善。SH 收益不单独证明温度恢复更准确。

## 保存、恢复与后续实验

所有 initial/best/step/final/common checkpoint 保存对应训练步的 `sh_residual`。
评估按每个相机重新计算方向；不能预计算一份固定颜色用于所有视角。
旧 checkpoint 没有此字段时完全沿用原渲染。评估的 `--physical_only` 可关闭残差，且默认输出目录独立。
RGB 评估继续使用 Stage1 RGB 外观。

后续 C/R/K/Gα 读取新 common 时自动继承并冻结 SH；不能在这些阶段新建 SH。
原自由标量灰度对照不能表达方向，因此拒绝从含 SH 的 checkpoint 静默丢弃残差，
请用本轮 degree=0 common 做该对照。通用 `render.py` 不是此物理 checkpoint 的评估入口。

```bash
python tools/test_thermal_sh_residual.py --require_torch
```

该检查覆盖 SH 与已有基函数的一致性、零初始化、方向拟合、幅度上限、无支持点冻结、
物理/SH 联合梯度、保存恢复、旧 checkpoint，以及实际 Python 渲染和验证路径。
光栅化检查用 CPU 替身，服务器仍须上述短步运行。
