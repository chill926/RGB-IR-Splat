# 坐标对应与热场初始化修复

## 原因与修改

1. `map_thermal_observations_to_gaussians` 的 count 是 opacity × 投影半径² 的累计权重。
   旧代码 `count.clamp_min(1)` 在权重小于 1 时压低辐射平均值，使一部分点初始温度偏冷。
   新代码对每个正权重使用真实分母，仅零支持点采用 fallback。日志打印低于 1 的权重点比例。
2. native raw 信号与数据集 thermal 导出存在稳定的约两像素偏移。它不是 RGB–IR 外参的恢复。
   新增按场景、仅 fit 像素估计的固定平移；不会从 validation/test 的像素估计参数。
3. 位移与原生→目标尺寸转换合成一次浮点信号采样。不是“先 resize，再 shift”。
   其他训练分辨率也从原生信号直接采样，避免再下采样已经插值的信号。
4. 超出原生视场的边缘以有限值填充，但通过有效 mask 排除出训练、信号误差与显示标定样本。
   原始伪彩 RGB PSNR/SSIM 仍使用完整画幅，防止通过裁边提高 benchmark 指标。
   信号 MAE/MSE、相机信号误差、表观温度 MAE 使用有效区；signal SSIM 保留全画幅并明确标记。
5. 新 radiometric manifest 版本 2 保存固定变换、fit 划分、输入 hash 与每帧 mask hash。
   新旧数据、显示映射和 checkpoint 不能静默混用；默认仍兼容旧版本 1。

相机内外参、COLMAP、RGB/thermal 原图、Stage1 几何不修改。旧 checkpoint 的已有权重不被重写。
从头训练的初始化行为修复会生效；加载旧 checkpoint 不会倒推修复过去的初始化。

## 本地验证

- 70 项 CPU 检查通过：包括低权重/零支持初始化、坐标符号、半像素约定、仅 fit 估计、
  更换分辨率仍一次采样、mask 梯度、有效区验证损失、输入 hash 和新旧协议冲突。
- 全部 RoadBlock 71 帧核查，使用原实验 fit 56 / validation 6 / test 9 划分。
- 新纯 SciPy 估计器得到参考画幅 640×480 下 `dx≈-1.8551, dy≈+0.9784`；56 张 fit 均通过，
  MAD 分别约 0.131、0.309 像素。采样含义为目标 (x,y) 取旧全帧信号 (x+dx,y+dy)。
- 新准备与显示标定流程的真值信号往返 PSNR：fit 36.59、validation 36.50、test 36.29 dB。
  原本地流程 test 约 31.49 dB。这里不经过 Gaussian 模型，不能把提升直接加到模型 PSNR。
- 实际 GPU 重训、真实 CUDA 光栅化、模型指标改善仍需服务器验证。

估计器只依赖 NumPy/Pillow/SciPy，不新增 OpenCV 要求。读取参考 checkpoint 和训练仍需要 Torch。
不同场景必须各自估计；若相关性不足、估计落在搜索边界或跨帧分散过大，会报错，不自动套用 RoadBlock 的值。

## 需要上传的文件

将更新包按相对路径覆盖服务器 graduate 目录。本包包括之前的 SH 支持文件，防止遗漏运行依赖。

```
utils/thermal_coordinates.py             新增：估计、单次采样、mask 与协议验证
tools/prepare_aligned_radiometry.py       新增：读取参考划分并准备新数据
tools/test_thermal_coordinates.py        新增：自包含 CPU 回归检查
scripts/run_roadblock_coordinate_repair.sh 新增：完整重跑入口
utils/thermal_physics.py
utils/thermal_observations.py
utils/thermal_training_utils.py
utils/thermal_pseudocolor.py
tools/prepare_rgbt_radiometry.py
train_thermal_physics.py
train_thermal_gray_control.py
tools/evaluate_thermal_physics.py
tools/run_sh_residual.py
tools/calibrate_thermal_pseudocolor.py
tools/recolor_thermal_renders.py
utils/thermal_sh_residual.py              现有 SH 支持，随包包含
tools/test_thermal_sh_residual.py         现有 CPU 预检查，随包包含
README_COORDINATE_REPAIR.md
README_SH_RESIDUAL.md
```

不用重新编译 CUDA 扩展。

## 一条命令执行完整流程

激活服务器 physir 环境。把更新包上传到 graduate 根目录；若实际根路径或参考不同，修改对应变量。

```bash
export GS_WORK=/mnt/sdg/liupengyu/graduate
cd "$GS_WORK"
unzip -o graduate_coordinate_repair_update.zip -d "$GS_WORK"

CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_roadblock_coordinate_repair.sh"
```

脚本默认参考：

```
output/RoadBlock_stage2_retry_tv0/thermal_stage2_common.pt
```

只读取旧 Stage2 的配置和划分，不加载其物理、SH 或 IR opacity 权重。RGB Stage1 复用。
默认新位置：

```
data/RGBT-Scenes/RGBT-Scenes/RoadBlock/radiometric_coordfix_v1/
output/RoadBlock_coordfix_v1_alignment.json
output/RoadBlock_coordfix_v1_display.json
output/RoadBlock_stage2_SH2_coordfix_v1/
```

步骤为 CPU 检查→仅 fit 估计→新信号数据→新显示映射→从头训练 Stage2→全模型和关闭 SH 的评估。
预算为 30000，SH 第 3000 步开放，其余 SH 参数和上一轮相同。物理超参数继承参考。

路径或版本需改变时：

```bash
REPAIR_REFERENCE="/实际路径/thermal_stage2_common.pt" \
REPAIR_SCENE="/实际路径/RoadBlock" \
REPAIR_TAG=coordfix_v2 \
CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_roadblock_coordinate_repair.sh"
```

输出必须是新的或空的，旧目录不会自动删除。中途失败时先看错误，重新完整跑可另设 REPAIR_TAG。

## 分步骤完整命令

```bash
export GS_WORK=/mnt/sdg/liupengyu/graduate
SCENE="$GS_WORK/data/RGBT-Scenes/RGBT-Scenes/RoadBlock"
REFERENCE="$GS_WORK/output/RoadBlock_stage2_retry_tv0/thermal_stage2_common.pt"
ALIGNED="$SCENE/radiometric_coordfix_v1"
ALIGNMENT="$GS_WORK/output/RoadBlock_coordfix_v1_alignment.json"
DISPLAY="$GS_WORK/output/RoadBlock_coordfix_v1_display.json"
OUTPUT="$GS_WORK/output/RoadBlock_stage2_SH2_coordfix_v1"

python "$GS_WORK/tools/test_thermal_coordinates.py" --require_torch

python "$GS_WORK/tools/prepare_aligned_radiometry.py" \
  --reference "$REFERENCE" --scene "$SCENE" \
  --output_dir "$ALIGNED" --alignment_output "$ALIGNMENT"

python "$GS_WORK/tools/calibrate_thermal_pseudocolor.py" \
  --scene "$SCENE" --checkpoint "$REFERENCE" \
  --radiometric_dir "$ALIGNED" --allow_new_radiometric \
  --output "$DISPLAY"

CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/tools/run_sh_residual.py" \
  --reference "$REFERENCE" --scene "$SCENE" \
  --radiometric_dir "$ALIGNED" --allow_new_radiometric \
  --display_calibration "$DISPLAY" --output "$OUTPUT" \
  --degree 2 --steps 30000 --start_step 3000 \
  --bound 0.5 --lr 0.01 --lr_final 0.0001 --regularization 0.001
```

`--allow_new_radiometric` 只用于从头训练/重新标定，显式允许参考配置与新 manifest 不同。
仍要求相机响应相同、坐标变换使用参考的同一划分。C/R/K 继续绑定新 common 的 manifest，不允许切换数据。

对新数据独立训练匹配无 SH 基线：

```bash
CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/tools/run_sh_residual.py" \
  --reference "$REFERENCE" --scene "$SCENE" \
  --radiometric_dir "$ALIGNED" --allow_new_radiometric \
  --display_calibration "$DISPLAY" \
  --output "$GS_WORK/output/RoadBlock_stage2_noSH_coordfix_v1" \
  --degree 0 --steps 30000
```

要隔离初始化修复效果，可用新代码在旧 radiometric 上再从头跑一组（不加 allow_new_radiometric），
再与“初始化修复+新坐标”比较。不要把同一个联合模型关闭 SH 的结果当成独立无 SH 基线。

## 结果

```
output/RoadBlock_stage2_SH2_coordfix_v1/training_summary.json
output/RoadBlock_stage2_SH2_coordfix_v1/evaluation_pseudocolor/metrics.json
output/RoadBlock_stage2_SH2_coordfix_v1/evaluation_physical_only/metrics.json
```

终端 `[thermal-init] below_one_weight_fraction` 会报告旧分母规则原本会影响的点比例。
它是 Gaussian 数量比例，不能直接当作像素贡献比例。新结果的信号误差排除了无效边缘，
伪彩指标保持完整原图；比较新旧信号指标时要同时看 signal_metric_region。
模型 PSNR 改善不单独证明真实表面温度更准确。
