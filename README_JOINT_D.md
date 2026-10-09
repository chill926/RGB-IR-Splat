# RoadBlock D 实验：RGB–IR 联合优化＋信号与显示监督

本实验使用现有 RGB Stage1 作为初始化，重新初始化温度和热 SH；坐标修正后的 Stage2 checkpoint 只提供设置、相机划分和输入来源，不加载其温度、环境、SH 或 IR 透明度权重。保留原训练入口和原实验结果。

## 默认实验设置

- 共 30000 步；第 1～2999 步冻结几何和 RGB 外观，训练温度、全局环境，使用信号与显示监督；从第 3000 步开始联合优化，并开放二阶热 SH。
- 更新位置、尺度、旋转、共享透明度、RGB SH、温度、全局环境和单通道二阶非 DC 热 SH（8 个系数）。材料发射率固定，K/R 参数冻结。
- 不增密、不剪枝，高斯数量固定。材料标签和 SH 支持绑定于初始高斯身份，来自 fit 相机；不使用 validation/test 图像更新材料归属。
- 信号损失沿用训练集信号跨度缩放的 SmoothL1；显示和 RGB 损失均为 `0.8 L1 + 0.2 (1-SSIM)`。三项权重都为 1；沿用环境、SH 先验，温度 TV 为 0。此次不额外加入边缘损失。
- 显示映射使用已有的 fit-only 校准，固定参数，可微插值；先合成信号，再着色，不直接学习三个伪彩色通道。
- 显示监督只使用 native FOV 有效像素，SSIM 只使用完全有效的 11×11 窗口；正式伪彩色评价仍使用完整原始图像。
- 每 500 步验证；按联合阶段 validation 的“信号＋显示”目标选择 checkpoint，不按 test 指标选择。继承的 RGB Stage1 可能训练过内部 validation 的 RGB 图像，这与旧基线相同；联合阶段只使用 fit 的 RGB 和热图。
- 每步一个视图，RGB 和热损失分别前向、反向，梯度累积后统一更新。图像和有效掩码驻留 CPU，相机矩阵及高斯驻留 GPU。
- 位置 LR 为 `1.6e-5 → 1.6e-7`，乘场景尺度；尺度 `5e-4`、旋转 `1e-4`、透明度 `5e-3`、RGB DC `2.5e-4`，联合阶段末降至各自初值的 1%；RGB 高阶为 DC 的 1/20。热场和 SH LR 沿用旧设置。

## 上传和运行

把 `graduate_joint_D_update.zip` 上传到服务器 `/mnt/sdg/liupengyu/graduate/`。需要此前已经安装坐标修正更新包、生成修正后的 radiometric 数据和 display 校准，并完成 SH2_coordfix_v1。

```bash
conda activate physir
export GS_WORK=/mnt/sdg/liupengyu/graduate
cd "$GS_WORK"
unzip -o graduate_joint_D_update.zip -d "$GS_WORK"
CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_roadblock_joint_D.sh"
```

脚本默认读取：

```text
output/RoadBlock_stage2_SH2_coordfix_v1/thermal_stage2_common.pt
data/RGBT-Scenes/RGBT-Scenes/RoadBlock/radiometric_coordfix_v1
output/RoadBlock_coordfix_v1_display.json
```

默认新输出为 `output/RoadBlock_joint_D_v1`。输出非空会拒绝覆盖；重复实验请换一个名字：

```bash
D_OUTPUT="$GS_WORK/output/RoadBlock_joint_D_v2" \
CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_roadblock_joint_D.sh"
```

若你的原文件位置不同，可以设置 `D_REFERENCE`、`D_SCENE`、`D_SIGNAL`、`D_DISPLAY`。默认读取 Stage1、材料配置和掩码位置自 reference；需要更改路径时使用下面的显式 Python 入口。

## 完整 Python 命令

```bash
conda activate physir
export GS_WORK=/mnt/sdg/liupengyu/graduate
CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/tools/run_joint_thermal.py" \
  --reference "$GS_WORK/output/RoadBlock_stage2_SH2_coordfix_v1/thermal_stage2_common.pt" \
  --scene "$GS_WORK/data/RGBT-Scenes/RGBT-Scenes/RoadBlock" \
  --radiometric_dir "$GS_WORK/data/RGBT-Scenes/RGBT-Scenes/RoadBlock/radiometric_coordfix_v1" \
  --display_calibration "$GS_WORK/output/RoadBlock_coordfix_v1_display.json" \
  --output "$GS_WORK/output/RoadBlock_joint_D_v1" \
  --steps 30000 --joint_start_step 3000 --start_step 3000 \
  --bound 0.5 --lr 0.01 --lr_final 0.0001 --regularization 0.001 \
  --lambda_rgb 1.0 --lambda_signal 1.0 --lambda_display 1.0 \
  --joint_position_lr 0.000016 --joint_position_lr_final 0.00000016 \
  --joint_scaling_lr 0.0005 --joint_rotation_lr 0.0001 \
  --joint_opacity_lr 0.005 --joint_rgb_lr 0.00025
```

入口自动运行 CPU 和可用的 CUDA 混合设备预检，再训练、评估选中模型的完整热图和 RGB，并评估原 Stage1 的 RGB 作为对照。输入 manifest、材料配置、Stage1、相机划分和 display 校准都有来源检查；不需要重新生成 radiometric 或 display。

## 看结果

训练后会输出 `[D comparison]`，并保存：

```text
training_summary.json                  best_step、最终验证指标、峰值显存
joint_training.jsonl                   分项损失、验证结果、显存
joint_comparison.json                  D 与旧热图、Stage1 RGB 的对比
thermal_joint_D_selected.pt            联合阶段选中的完整模型
thermal_joint_D_final.pt               最后一步完整模型
thermal_joint_D_warmup.pt              开放几何前的模型
thermal_joint_D_latest.pt              最近一次定期保存
evaluation_pseudocolor/metrics.json    正式 fit / validation / test 热图指标
evaluation_rgb/metrics.json            D 模型的 RGB 指标
evaluation_rgb_initializer/metrics.json 原 Stage1 RGB 对照
```

```bash
cat "$GS_WORK/output/RoadBlock_joint_D_v1/joint_comparison.json"
cat "$GS_WORK/output/RoadBlock_joint_D_v1/training_summary.json"
```

热图旧基线自动读取 reference 所在目录的 `evaluation_pseudocolor/metrics.json`，存在时要求与 D 的输入和 display 协议一致；不存在时仍完成 D 与 RGB 对照，但不填旧热图差值。你这次旧基线 test PSNR 为 17.4082 dB。

D checkpoint 内包含更新后的几何、透明度、RGB SH、物理场和热 SH。评估会恢复这些几何并校验其 checksum，报告 `geometry_evaluated=checkpoint_joint_geometry`。它不是旧冻结 Stage2 的 common endpoint，不能直接接到旧 C/K/R/G-alpha 分支。

## 本地验证范围

本地通过可微显示、梯度、顺序反传、热场重新初始化、warmup、保存与实际评估入口恢复等 CPU 检查。真实 RoadBlock 71 个视图的抽样像素与现有 NumPy 显示映射完全一致，扰动信号的梯度均有限且非零。

本机没有可用的服务器 CUDA 光栅化训练环境，尚未运行完整 D 训练，也未测量其实际显存峰值。服务器预检会额外测试 CPU 图像/掩码与 CUDA 几何的初始化；真实训练和收益以运行结果为准。
