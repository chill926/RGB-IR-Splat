# 从热场渲染信号恢复数据集伪彩热图

本功能复用 `raw_rjpeg` Stage2/C/R/K checkpoint：

**热场 → 指定视角信号合成 → 反归一化 Q → 固定显示映射 → 伪彩 RGB → 原始 thermal RGB 指标。**

映射应用于二维信号合成之后，未把 Gaussian 温度或伪彩颜色提前进行 alpha blending。
无需重新训练 Stage1/Stage2，无需重新转换原有 `radiometric` 数据。

## 实现与适用范围

读取原始 JPEG 的 YCrCb 调色板和 RawValueMedian/RawValueRange，按已知拍摄设置
把 `Q=DN+O` 转成显示窗口坐标。从 checkpoint 的 **fit 子集**采样信号与原始伪彩图，
选择调色板的 full/limited 编码，再标定单调的“窗口坐标 → 调色板位置”查找表。
使用加权单调回归约束映射，不利用模型的预测误差调颜色，不从 validation/test 像素拟合参数，
也不按每张预测图的 min/max 自动拉伸。

各场景独立标定，不绑定场景名、温度范围、材料名或固定调色板常数。
held-out 图像的相机显示元数据仅用于选择调色板、应用已知窗口参数；不会读取其像素进行标定。
若某个调色板只出现在 held-out 拍摄中，则采用该拍摄的已知调色板，
共享训练数据中支持最多的调色板所标定的响应，并在 JSON 中明确记录这一回退。
C/R/K 应复用同一个显示标定文件。

**这是从训练数据估计的固定显示映射，不保证精确复刻相机及数据集导出程序的 AGC。**
所以每次评价都会报告“真实信号经此映射能否恢复原始伪彩图”的往返误差。
该误差参考不能简单当作模型指标的数学上限，也不能直接从模型 MSE 中减掉。
如果往返图明显有颜色、边缘或局部对比度差异，说明后续仍需改进显示模型/配准。
目前数据集的原始信号输入画幅仍沿用此前的全帧缩放方案。

## 服务器需要更新的五个运行文件

1. `utils/flir_radiometry.py`
2. `utils/thermal_pseudocolor.py`
3. `tools/calibrate_thermal_pseudocolor.py`
4. `tools/recolor_thermal_renders.py`
5. `tools/evaluate_thermal_physics.py`

保留上一轮已上传的原始辐射输入相关文件。新测试文件和本说明无需用于运行。
标定需要 NumPy、Pillow、SciPy、PyTorch；无 CUDA 要求。
已有数组恢复只需要 NumPy、Pillow、SciPy，不需要 PyTorch、CUDA 或 3DGS 扩展。

## 已有渲染数组：推荐使用这条流程

你已经运行了 `evaluate_thermal_physics.py --save_images`，所以可以直接恢复已有数组。
以目前已有的 RoadBlock 目录为例，场景路径可替换为其他场景：

```bash
GS_WORK=/mnt/sdg/liupengyu/graduate
SCENE="$GS_WORK/data/RGBT-Scenes/RGBT-Scenes/RoadBlock"
OUTPUT="$GS_WORK/output/RoadBlock_stage2_radiometric_v1"
DISPLAY="$OUTPUT/pseudocolor_calibration.json"
```

先标定一次：

```bash
python "$GS_WORK/tools/calibrate_thermal_pseudocolor.py" \
  --scene "$SCENE" \
  --checkpoint "$OUTPUT/thermal_stage2_common.pt" \
  --radiometric_dir "$SCENE/radiometric" \
  --output "$DISPLAY"
```

该步骤读取 checkpoint 记录的 56 个 fit 视角（当前 RoadBlock 情况），不使用六个验证视角
或官方测试图像的像素拟合显示参数。其他场景按各自 checkpoint 的划分处理。
标定文件已存在时不会覆盖；直接复用它。需要重新标定时选择新的 `--output` 路径。

然后恢复已有的六张验证渲染图：

```bash
python "$GS_WORK/tools/recolor_thermal_renders.py" \
  --scene "$SCENE" \
  --evaluation_dir "$OUTPUT/render_validation" \
  --display_calibration "$DISPLAY" \
  --radiometric_dir "$SCENE/radiometric" \
  --split validation \
  --output_dir "$OUTPUT/render_validation_pseudocolor"
```

新输出目录必须不存在或为空。`--radiometric_dir` 的相对路径相对于场景目录解析。
不更改原渲染数组、模型或数据集图像，不需要 GPU，也不需要再次执行 3DGS 渲染。
源目录必须有 `metrics.json`、`validation/per_view.json` 和 `float_arrays`。
恢复脚本验证输入域、标定来源、视角划分与原始伪彩图哈希。

## 从 checkpoint 直接输出伪彩图

已有显示标定文件后，可以一步完成真正的三维渲染与恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python "$GS_WORK/tools/evaluate_thermal_physics.py" \
  -s "$SCENE" \
  --checkpoint "$OUTPUT/thermal_stage2_common.pt" \
  --modality thermal \
  --thermal_output pseudocolor \
  --display_calibration "$DISPLAY" \
  --split validation \
  --output_dir "$OUTPUT/render_validation_pseudocolor_direct" \
  --save_images
```

此命令需要原 physir CUDA/3DGS 环境。省略 `--thermal_output pseudocolor` 时仍按旧方式输出信号图。
正式测试可将 `--split validation` 改为 `--split test`，并使用新的输出目录；
`--split all` 则分别报告 fit/validation/test。显示映射保持固定。

## 看哪些结果

- `validation/renders/`：模型恢复后的伪彩 RGB 图。
- `validation/gt/`：数据集原始伪彩图。
- `validation/comparisons/`：左为原始真值、中间为模型输出、右为绝对 RGB 误差。
- `validation/roundtrip/`：真实信号经过显示映射的恢复图，用于隔离显示误差。
- `validation/roundtrip_comparisons/`：左为原始真值、中间为真实信号恢复图、右为绝对误差。
- `validation/per_view.json`：每张图的指标。
- 根目录 `metrics.json`：各 split 的汇总指标及标定来源。

新模式中的 `PSNR/SSIM/MSE/MAE/RMSE` 都在原始伪彩 RGB `[0,1]` 图像域计算。
`signal_PSNR/signal_SSIM` 等保留旧归一化信号域的指标；`camera_signal_MAE`、
`apparent_temperature_MAE_K` 等继续评价物理观测拟合。
`mapping_roundtrip_PSNR/SSIM` 评价真实信号的显示恢复误差，不是三维模型的成绩。
图像指标在保存 PNG 前用浮点数组计算，避免额外量化或 JPEG 再压缩。
SSIM 使用与现有实现一致的 11 像素高斯窗口、sigma=1.5、零填充、data_range=1。
没有把新伪彩图再转灰度后计算这些 RGB 指标。

## 实际验证

12 项新 CPU 测试覆盖：调色板读取、单调拟合、held-out 像素隔离、固定映射、
已知窗口变化、未见调色板的元数据回退、输入来源检查、反归一化、外推范围、
已有数组恢复、RGB 指标单位以及与原 PyTorch SSIM 公式的一致性。
此前 15 项辐射输入测试也用于回归检查。

全部十个场景完成了训练标定与 held-out 原始信号往返检查。
以下结果是显示映射的验证往返 PSNR，不是任何 Stage2 模型的结果：

| 场景 | validation 往返 PSNR dB（约） |
|---|---:|
| Building | 30.63 |
| DailyStuff | 21.35 |
| Dimsum | 30.61 |
| Ebike | 29.21 |
| IronIngot | 29.30 |
| LandScape | 31.37 |
| Parterre | 30.56 |
| RoadBlock | 31.63 |
| RotaryKiln | 31.18 |
| Truck | 23.48 |

DailyStuff、Truck 等场景说明固定显示映射仍有明显误差，目前不应称为精确原图恢复。
所有场景都可执行恢复和原图评价，报告会保留这一误差。
本地没有 CUDA；没有运行服务器上的 Stage2 checkpoint，也未在本地验证 GPU 光栅化。
已有数组恢复流程已用真实数据及合成已知预测完成 CPU 验证。

任意新视角的接口是 `FrozenDisplay.colorize_with_settings(normalized, bounds, settings)`：
传入已知/指定的相机显示设置，使用冻结曲线生成 RGB，不需要该视角真值图。
