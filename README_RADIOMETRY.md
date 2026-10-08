# RGBT-Scenes 原始辐射输入

新增 `--observation_domain raw_rjpeg`，支持整个 RGBT-Scenes 的原始 FLIR JPEG。
转换时读取各场景 JPEG 自带的标定，不使用任何特定场景的固定标定常数。
原有 `normalized_dn`、`calibrated_radiance`、`apparent_temperature` 输入模式保留。

## 输入与前向模型

从 `raw_images` 的 FLIR APP1/FFF 数据中提取 16 位原始 DN 与 `PlanckR1/R2/B/F/O`。
支持 16 位 PNG（自动检测字节序）、JPEG-LS、未压缩二进制数据。
RGBT-Scenes 当前十个场景都已实际验证；JPEG-LS 出现在 Building、Parterre、Truck 各一张。

使用相机等效信号：

\[
Q_{obs}=DN+O,\qquad Q_{BB}(T)=\frac{R_1}{R_2[\exp(B/T)-F]}.
\]

所有视角使用相同的固定温度上下界，对观测和前向响应应用同一仿射归一化：

\[
s(Q)=\frac{Q-Q_{BB}(T_{min})}{Q_{BB}(T_{max})-Q_{BB}(T_{min})}.
\]

因此 `ε*s(Q_BB(T))+(1-ε)*s(Q_env)` 与先混合信号再归一化一致。
不做逐图 min/max 拉伸、不从伪彩灰度反推辐射、不把 DN 当作线性温度。
前向模型在此模式下使用相机响应 LUT，替代均匀 8–14 μm LUT；两者的坐标不能混用。

`Q` 是相机等效辐射信号，单位不标成 W/(m²·sr)。输出温度是依赖材料发射率、
环境项及前向模型的反演估计，不是测得的表面温度真值。
JPEG 的发射率、反射温度、距离、空气与窗口设置仅记录，不先进行物体温度补偿，
避免与模型中的发射率/环境项重复处理。此版本不增加大气传输、深度/法向或 IR opacity 优化。

## 转换任意场景

转换仅需 NumPy、Pillow。JPEG-LS 另外需要 `imagecodecs` 或 PATH 中的 FFmpeg；
解码器优先使用 imagecodecs，其次使用 FFmpeg。转换完成后，训练不再需要这两个解码依赖。
可在单独的转换环境安装依赖，保持现有 3DGS 环境的 NumPy/PyTorch 版本。

```bash
PROJECT=/mnt/sdg/liupengyu/graduate
DATA_ROOT="$PROJECT/data/RGBT-Scenes/RGBT-Scenes"
SCENE_NAME=Ebike
SCENE="$DATA_ROOT/$SCENE_NAME"

python "$PROJECT/tools/prepare_rgbt_radiometry.py" --scene "$SCENE"
```

或一次转换整个数据集：

```bash
python "$PROJECT/tools/prepare_rgbt_radiometry.py" --dataset_root "$DATA_ROOT"
```

默认输出各场景的 `radiometric/manifest.json` 和 `radiometric/train|test/图像名.npy`。
输出目录必须不存在或为空；需要重新转换时，指定一个新的 `--output_dir`。
单场景的该选项指定输出目录；批量模式的该选项指定输出父目录，每个场景有独立子目录。
不覆盖原始 RGB、thermal、raw_images、材料掩码或 Stage1 模型。

数组为 float32 的 Q 信号，与提供的 thermal 图相同画幅、相同大小。
缩放在线性信号域完成，随后才转为温度或归一化信号。
沿用数据集的 thermal 画幅与现有 RGB/IR 配准，未新增外参估计。
信号与伪彩图灰度的相关性只用于画幅诊断，不能证明精确的 RGB/IR 配准。
若换用裁剪、畸变校正或另行配准的数据，需要对浮点信号施加同样的变换。

## Stage2

沿用已有的 RGB Stage1 几何和材料掩码，在新的模型目录重新训练 Stage2。
把原先 `--observation_domain normalized_dn` 替换为：

```bash
--observation_domain raw_rjpeg --radiometric_dir radiometric
```

`--radiometric_dir` 可使用场景目录下的相对路径或绝对路径，默认就是 `radiometric`。
例如：

```bash
GEOMETRY="$PROJECT/output/${SCENE_NAME}_stage1_3dgs_radii"
MATERIALS="$SCENE/material_masks_v1/train"
OUTPUT="$PROJECT/output/${SCENE_NAME}_stage2_radiometric_v1"

CUDA_VISIBLE_DEVICES=0 python "$PROJECT/train_thermal_physics.py" \
  -s "$SCENE" -m "$OUTPUT" --data_branch rgbt --eval \
  --stage stage2 --geometry_model "$GEOMETRY" --geometry_iteration 30000 \
  --material_mask_dir "$MATERIALS" --material_config "$MATERIALS/material_config.json" \
  --observation_domain raw_rjpeg --radiometric_dir radiometric \
  --steps 20000 --seed 0
```

上面的几何、掩码路径和迭代数应填各场景已有结果。新模式需要重新训练 Stage2，
旧灰度 checkpoint 不会自动变成辐射标定结果。Stage1 可复用。

### 温度范围

训练仍使用显式 `--temp_min` / `--temp_max`，默认 250–450 K；不按每张图自动拉伸。
转换程序会输出场景观测的黑体等效温度范围，用来发现配置范围不足。
范围不足时训练会报错，不会把高温像素静默截到上限。

当前本地数据的黑体等效观测范围如下。这些数值不是物体表面温度真值。

| 场景 | 图像数 | 观测范围 K（约） |
|---|---:|---:|
| Building | 273 | 282.9–298.1 |
| DailyStuff | 78 | 289.9–359.7 |
| Dimsum | 154 | 293.6–350.2 |
| Ebike | 48 | 288.5–293.5 |
| IronIngot | 61 | 293.2–649.6 |
| LandScape | 103 | 285.3–299.1 |
| Parterre | 66 | 288.1–300.9 |
| RoadBlock | 71 | 289.1–300.5 |
| RotaryKiln | 106 | 275.9–480.5 |
| Truck | 73 | 287.2–779.4 |

因此 IronIngot、RotaryKiln、Truck 不能直接沿用 450 K 的上限。
全场景输入一致性检查使用固定的 `--temp_min 250 --temp_max 1000`，验证未截断观测；
正式实验的模型范围应结合场景的物理预期设定。低发射率目标的表面温度可能高于上述观测温度，
不应把观测最大值直接当成真实表面温度上限。C/R/K 必须使用相同范围。

改变归一化范围会改变辐射损失的数值尺度，原有 TV/环境正则的相对强度可能需要重新检查。
本次保持这些正则设置与停止规则不变。若没有生成 common，仍按原协议处理，不重命名 final。

## 分支、评估与对照

C/R/K 自动继承新 Stage2 checkpoint 的输入域、标定与归一化范围。
checkpoint 记录 manifest SHA256，manifest 记录逐图信号文件 SHA256；
后续输入内容、标定或范围不一致时拒绝继续。场景路径移动后，可以传新的 `--radiometric_dir`，
但目录中的实际内容必须与 checkpoint 一致。

评估命令沿用已有用法，无需再手填温度范围或标定系数：

```bash
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/tools/evaluate_thermal_physics.py" \
  -s "$SCENE" --checkpoint "$OUTPUT/thermal_stage2_common.pt" \
  --split all --output_dir "$OUTPUT/evaluation" --save_images
```

如只有 `thermal_stage2_final.pt`，可以对它进行诊断评估；它仍不满足 C/R/K 的 common 起点要求。
`train_thermal_gray_control.py` 同时支持旧灰度输入与新的标定信号输入，保持与 Stage2 的对照一致。
通用 `train_thermal_image_baseline.py` 仍按原始图像训练，不应用此功能，不应直接作为新辐射域的公平对照。

评估仍报告 PSNR/SSIM，并增加：

- `camera_signal_MAE`、`camera_signal_RMSE`：相机等效信号误差。
- `apparent_temperature_MAE_K`：预测与观测的黑体等效温度图误差，不是表面温度 GT 误差。

新模式的 PSNR/SSIM 在归一化相机信号域计算，与旧伪彩灰度指标不能直接比较。
即使同为新模式，采用不同温度范围的 PSNR 也不能直接比较。

## 验证与文件

CPU 测试：`python tools/test_thermal_radiometry.py`。
15 项测试覆盖 PNG 字节序、APP1 分段、大小端二进制、截断输入、标定往返、
仿射混合、浮点缩放、范围检查、文件完整性、相机响应差异、LUT 梯度、
checkpoint 输入继承、旧灰度行为以及误差单位。
Torch/SciPy 未安装时模型测试会明确跳过。

全数据集实际完成：10 个场景、1133 张图的转换、RGB/IR 文件名匹配、逐文件哈希、
CPU 相机浮点输入验证，覆盖 PNG16 和 JPEG-LS16、两套场景标定。
本地没有 CUDA，未运行完整 3DGS 训练或 GPU 光栅化；服务器需在原 physir 环境运行。

运行所需更新/新增的七个 Python 文件：

1. `train_thermal_physics.py`
2. `train_thermal_gray_control.py`
3. `tools/evaluate_thermal_physics.py`
4. `tools/prepare_rgbt_radiometry.py`
5. `utils/thermal_physics.py`
6. `utils/flir_radiometry.py`
7. `utils/thermal_observations.py`

另有 `tools/test_thermal_radiometry.py` 与本说明文件。
无需修改 RGB Stage1、材料标注转换或 SAM2 传播代码。

实现依据：[ExifTool FLIR 格式说明](https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/FLIR.pm)、
[Thermimage 相机标定公式](https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R)、
[imagecodecs JPEG-LS 实现](https://github.com/cgohlke/imagecodecs/blob/master/imagecodecs/_jpegls.pyx)。
