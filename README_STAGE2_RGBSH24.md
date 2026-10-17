已部署至 2080 的 `/home/liupengyu/graduate`。Stage1 无须重跑；沿用 `output/RoadBlock_3dgs_run1/point_cloud/iteration_30000/point_cloud.ply`。本地桌面原项目未改，附带补丁和源码用于同步或审阅。

本版每个高斯新增三通道、一阶与二阶 SH，共 `3 × (3+5) = 24` 个可训练参数，没有新增零阶项。按相机到高斯的方向计算，结果直接加在原核心公式之后：

```
RGB_i(v) = repeat(ε_i B(T_i) + (1−ε_i)E_i, 3)
           + Σ_{l=1,2} Σ_m c_{i,lm}^{RGB} Y_lm(v)
```

温度、环境项、24 个 SH 参数、位置、尺度、旋转和不透明度从第 1 步起共同优化。材质发射率固定；Stage1 的 RGB SH 固定。位置也参与 SH 方向的梯度。为保持高斯与材质标签的一一对应，本版没有新增或剪除高斯。可用 `--lambda_rgb` 添加固定 RGB SH 的 RGB 图像约束，默认值为 0。

SH 零初始化；每通道残差有平滑幅度约束（默认 `--rgb_sh_bound 0.5`），正则权重默认 `--lambda_rgb_sh 0.001`。系数仍组成一阶、二阶线性 SH 展开。合成颜色先限制在 `[0,1]`，再进入原高斯渲染器。日志记录梯度和截断比例。位置学习率为 Stage1 量级的约十分之一，其余几何学习率也调低，适合已有几何的联合微调。

RoadBlock 当前 thermal 图为伪彩 RGB，因此命令显式使用 `pseudocolor_rgb`。这次优化针对图像外观和 PSNR；温度仅作为未标定模型变量，不能据此报告真实温度精度。若改用原始辐射测量数据，可选择 `raw_rjpeg` 并先准备通过校验的 radiometric manifest。

正式训练前准备两个输入：

- `data/RGBT-Scenes/RGBT-Scenes/RoadBlock/material_masks/`：每张图对应一个同名、单通道 PNG，像素值为材质 ID `0,1,...`；未知区域为 `255`。SAM2 的对象二值 mask 需要按材质归类、合并成此标签图。仅训练 fit 视角的 mask 用于标签融合，不使用验证或测试 mask。
- `material_config.json`：使用项目原有的材料配置格式，材质 ID 连续且与标签图一致，填写你确认的 `epsilon0` 与 `confirmed: true`。测试使用的虚拟发射率不是 RoadBlock 的真实材料值，请勿复制到正式实验。

内部划分为 56 个 fit、6 个 validation、9 个 published test 视角；best checkpoint 依据内部 validation loss 选择，test 不用于训练或选择。正式训练命令（两个输入准备好之后运行）：

```bash
conda activate 3dgs
cd /home/liupengyu/graduate
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nohup python -u train_thermal_stage2_rgbsh.py \
  -s data/RGBT-Scenes/RGBT-Scenes/RoadBlock \
  -m output/RoadBlock_stage2_rgbsh24_joint_run1 \
  --geometry_model output/RoadBlock_3dgs_run1 \
  --geometry_iteration 30000 \
  --material_mask_dir data/RGBT-Scenes/RGBT-Scenes/RoadBlock/material_masks \
  --material_config material_config.json \
  --observation_domain pseudocolor_rgb \
  --eval --resolution 1 --steps 10000 --seed 0 \
  > logs/RoadBlock_stage2_rgbsh24_joint_run1.log 2>&1 &

tail -f logs/RoadBlock_stage2_rgbsh24_joint_run1.log
```

输出目录必须为空或不存在；重复实验用新的 `-m` 和日志名。默认沿用 Huber 数据损失、温度 TV 和环境先验。如果要实验 3DGS 常用的 L1+SSIM，显式增加 `--photometric_loss l1_ssim`，比较时各组保持一致。

训练保存 `thermal_stage2_joint_best.pt`、`thermal_stage2_joint_final.pt`、每 1000 步 checkpoint、`training.jsonl` 和 `training_summary.json`。每个 checkpoint 包含完整的联合几何、Stage1 RGB SH、温度与环境项、固定材料表、SH24 及相机划分。用新的评估入口恢复这些内容：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_stage2_rgbsh.py \
  --checkpoint output/RoadBlock_stage2_rgbsh24_joint_run1/thermal_stage2_joint_best.pt \
  --split test --save_images
```

默认在 checkpoint 目录下新建评估目录，写入 `metrics.json`，图像依次拼接 GT、预测、绝对误差。PSNR 是每视角 RGB MSE 得出的 PSNR 的平均值，图像范围 `[0,1]`。此 checkpoint 有独立格式标记，不能直接交给旧冻结 Stage2 的 C/K/R 或 D 入口；若后续还要跑这些实验，需要显式适配联合几何及 SH24。

验证已覆盖：SH 与标准 3DGS 球谐约定一致、零初始化、三通道独立、幅度边界、随视角变化、位置方向梯度、checkpoint 恢复与损坏校验；原单通道 SH 的 12 项回归测试也通过。RoadBlock 219,796 个高斯的 20 步 GPU 冒烟训练，七个优化组均获得非零梯度，位置、尺度、旋转、不透明度确实更新。独立进程恢复后的验证指标完全一致。另完成 640×480 分辨率的短训练，覆盖 TV 和可选 RGB 约束。这些使用隔离的虚拟材料 mask，仅验证程序，不代表正式实验跑分。

新增代码：`train_thermal_stage2_rgbsh.py`、`utils/thermal_rgb_sh.py`、`tools/evaluate_stage2_rgbsh.py`、`tools/test_stage2_rgbsh.py`。原 `train_thermal_physics.py` 仅增加可选 parser 扩展和新观察域的归一化兼容；原入口行为保持原样。Stage1 的 `train.py`、GaussianModel 和渲染器文件哈希均未改变。旧 Stage2 文件备份在 `/home/liupengyu/.cache/graduate-setup/backups/stage2-rgbsh-20261012-160032/`。
