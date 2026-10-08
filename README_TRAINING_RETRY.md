# 原始信号 Stage2 重跑

第一轮建议先跑 **固定损失尺度 + TV=0 + 30000 步固定预算**，看当前几何能否拟合热细节。
这是排查温度正则是否过强的实验，不是最终选定的论文超参数。Stage1 和 radiometric 数据可复用。

## 更新文件

三个修改文件：`train_thermal_physics.py`、`train_thermal_gray_control.py`、`tools/evaluate_thermal_physics.py`。
新增：`utils/thermal_training_utils.py`、`tools/test_thermal_training.py`、`tools/run_radiometric_retry.py`、本说明。
把更新包解压到服务器的 graduate 根目录，保留相同相对路径。

## 一条命令训练并评价

沿用之前服务器的 GS_WORK、SCENE、OUTPUT 变量（OUTPUT 指旧 Stage2 目录）：

```bash
CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/tools/run_radiometric_retry.py" \
  --reference "$OUTPUT/thermal_stage2_common.pt" \
  --output "$GS_WORK/output/RoadBlock_stage2_retry_tv0"
```

脚本复用参考 checkpoint 记录的数据、几何、温度范围、随机种子和划分，重新初始化 Stage2。
从记录的材料配置文件所在目录寻找原材料掩码；若掩码实际在别处，加上
`--material_mask_dir "原来用的掩码目录"`。数据或几何路径迁移时可分别传 `--scene`、
`--geometry_model`、`--radiometric_dir`、`--material_config`；文件内容仍必须匹配原记录。

训练前执行新 CPU 测试，随后跑满预算、选内部验证损失最低的共享 checkpoint，
重新生成 fit-only 显示标定，并输出 fit/validation/test 的原始伪彩 RGB 评价。
输出目录必须是新的或空的，已有实验不覆盖。

测试 CUDA/光栅化是否正常时，可以先用 `--steps 5` 和另一个输出目录运行；这只是运行检查。

## 改了什么

- **损失尺度**：raw_rjpeg 默认使用 fit 像素 P99−P1（16 像素固定步长采样，最小尺度 1e-4）。
  对残差除以这一全场景尺度，再计算 Huber；不改变相机信号、Planck 响应、温度范围或图像内容。
  validation/test 不参与尺度估计。checkpoint 保存尺度，C/R/K 和自由信号对照继承同一尺度。
- **温度 TV**：用固定的 10 K 尺度无量纲化；两端均有可信材料标签且类别不同时降低边权。
  本轮运行脚本默认 TV=0，主训练入口默认 lambda_tv 仍为 0.01。
- **学习率**：默认前半程保持初始值，后半程指数衰减至原有末值。
- **最优保存**：任何更好的验证损失都更新 best；patience 单独累计有意义的改善。
  平台判断默认绝对门槛 1e-8、相对门槛 0.2%，不再用 1e-5 统一卡住小损失实验。
- **停止**：默认 `--stage2_stop_mode budget`，达到预算不报“已收敛”。
  可选 `plateau` 模式还要求最少训练步数、可见贡献加权温度更新、可见更新 P95 及环境稳定性。
- **共享起点**：common 改为训练预算内验证最优的 Stage2（包含初始化候选），
  并保存 `stage2_endpoint_policy` 和实际停止原因。所有 C/R/K 仍复用这同一个起点。
- **诊断**：记录各项加权损失、温度数据/TV 梯度范数及比例、可见温度更新量和物理观测误差。
- **评价**：直接伪彩评价读取原 thermal RGB，不再使用 Camera 内的灰度图作 RGB 真值。
  相机信号 RMSE 汇总改为从各视角 MSE 合并计算，旧的逐视角 RMSE 平均可能略有不同。
- **旧模型**：没有新尺度协议的 checkpoint 仍按旧损失/TV 单位加载，不会自动升级成新训练。
  新训练流程的 `radiance_loss` 是缩放后的 Huber；`normalized_signal_huber_loss` 保留未缩放 Huber。
  比较新旧性能用伪彩 PSNR/SSIM、相机信号误差与表观温度误差，不比较不同单位下的训练 loss。

## 看哪些结果

- `training_summary.json`：实际停止步数、best_step、停止原因、最优与最终验证误差。
- `thermal_training.jsonl`：验证曲线和 `tv_to_data_gradient_ratio` 等诊断。
- `evaluation_pseudocolor/metrics.json`：fit/validation/test 的伪彩指标与信号指标。
- `evaluation_pseudocolor/validation/comparisons/`：原图、预测、误差。

默认质量状态是 `unassessed`，不会把跑满预算或参数稳定自动说成质量合格。
如需明确的观测拟合目标，可加 `--quality_apparent_mae_K 0.1`，用于标记 `passed/poor_fit`。
0.1 K 是人为设定的诊断目标，不是传感器精度或真实表面测温标准，也不影响固定预算停止。

第二轮仅恢复温度 TV，其余条件不变，可用另一个输出目录加 `--lambda_tv 0.001`。
不同 TV 强度属于消融，选定统一设置后再训练 C/R/K；旧的 K/R 正则扫描不能直接当作新损失单位下的最优值。

## 自由信号对照

```bash
RETRY="$GS_WORK/output/RoadBlock_stage2_retry_tv0"
CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/train_thermal_gray_control.py" \
  -s "$SCENE" -m "$GS_WORK/output/RoadBlock_free_signal_retry" \
  --stage2_checkpoint "$RETRY/thermal_stage2_common.pt" \
  --steps 30000 --seed 0 --save_images
```

对照继承同一尺度；新尺度协议下默认直接信号学习率为 1e-4→1e-6，
避免原来的 0.01 步长远大于 RoadBlock 的信号细节。
它也保存浮点数组和兼容 recolor 的根报告，可复用 RETRY 的显示标定：

```bash
python "$GS_WORK/tools/recolor_thermal_renders.py" \
  --scene "$SCENE" \
  --evaluation_dir "$GS_WORK/output/RoadBlock_free_signal_retry/evaluation_best" \
  --display_calibration "$RETRY/pseudocolor_calibration.json" \
  --split validation --output_dir "$GS_WORK/output/RoadBlock_free_signal_retry/pseudocolor"
```

## 验证范围

本地语法检查通过；新训练测试、原始辐射测试和伪彩测试共 29 项通过，8 项因无 Torch 跳过。
新测试包含停止与质量分离、累计小幅改善、旧/新 checkpoint 的单位继承和损失尺度不变性。
本地没有 Torch/CUDA，尚未验证完整 GPU 训练、光栅化和自定义反向传播；
服务器运行脚本会先执行全部 10 项新训练测试，包括 Torch 梯度与尺度继承检查。
