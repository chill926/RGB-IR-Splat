# RoadBlock Stage3: frozen D C / R / K pilot

2026-10-14 更新：已增加独立学习率入口及验证集超参搜索。
推荐先按 `README_STAGE3_TUNING.md` 启动搜索，再用生成的参数文件启动完整训练。

第一版从 `output/RoadBlock_joint_D_3090_repro_2080_run1/thermal_joint_D_selected.pt`
导出共同起点。完整保留 D 训练后的几何、不透明度、RGB SH、标量热 SH8、T、E、
材料身份、unknown 及固定 epsilon0；没有重新划分材料或修改旧 checkpoint。

## 模型和比较协议

- C: 只训练每个 Gaussian 的 T 和全局 E。
- R: 训练 T、E 和显式开启材料的共享 delta_epsilon。
- K: 训练 T、E 和同一批材料的共享 k；epsilon(T)=epsilon0+k*(T-Tref)。
- 固定 epsilon0、几何、opacity、RGB SH、thermal SH；unknown 保留固定 fallback epsilon0。
- 共用 D 的 fit 56 / validation 6 / test 9 划分、RJPEG 坐标修正、归一化、
  fit-only signal loss scale、固定 display calibration 和 signal+display 损失权重。
- RGB 仅评价。训练和 checkpoint 选择不读取测试集信号。
- 每个 seed / 预算 / 正则配置有独立 comparison lock，C/R/K 参数必须一致。
- K-UQ、Oracle/GT 监督和合成数据生成暂不包含在此入口中。

## 运行

在 2080 执行，默认绝对路径 Python 是 3dgs 环境，避免系统 Python 2.7：

```bash
export GS_WORK=/home/liupengyu/graduate
CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_roadblock_stage3.sh"
```

默认材料明确为 `matte_plastic`（沿用 D 的开关），其余三类固定。
若需要其他材料参与 K/R，在新实验中显式指定，例如：

```bash
export STAGE3_RUN=RoadBlock_stage3_D_allmaterials_run1
export STAGE3_LEARN_MATERIALS=matte_plastic,reflective_sheet,metal_screw,metal_assumed
CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_roadblock_stage3.sh"
```

导出入口要求 `--learn_materials`；`preserve` 明确复制 D 中匹配的 K/R 开关。
共同起点和已有训练目录不会覆盖；改配置时使用新的 `STAGE3_RUN`。
脚本依次训练 C/R/K，当前低 LR 试跑默认各 2000 步，然后对最佳 checkpoint 评价所有划分的 IR 伪彩和 RGB。
训练使用的学习率和正则值写入 `experiment_plan.json`，它们是试跑值，未经合成真值调参。

当前默认材料 raw LR=1e-4 -> 1e-6；lambda_k=lambda_delta_epsilon=1e-4。
正则是各材料相对于原有 sigma_k / sigma_delta 的标准化平方偏差，k_prior 沿用 D。
shell 试跑默认 T LR=1e-4→1e-6、E LR=1e-5→1e-7、hold=0.1；环境先验强度沿用 D。
直接 Python 入口不指定 T/E LR 和 hold 时继承 D；实际训练值写入实验协议和日志。环境正则锚定共同起点 E。
验证选择 signal+display objective，初始 step 0 也参与选择，避免保存更差的更新结果。

## 输出和检查

`*_common/thermal_stage3_common.pt` 及 JSON 保存逐类开关、数量、epsilon0 和来源 hash。
各分支有 `stage3_training.jsonl`、`thermal_{branch}_best.pt`、final、summary。
日志包含数据损失、环境/材料正则、材料数据梯度、k/delta、温度分布和边界比例。
每次保存校验几何、RGB/thermal SH、epsilon0 和全部材料 buffer 未变。
评估入口 `tools/evaluate_thermal_physics.py` 已支持直接读取 RJPEG 物理信号，
不把伪彩转灰度当成物理目标；原始热图仅为尺寸信息和固定 display 监督目标。

真实数据没有表面 T / k / epsilon / E 真值。PSNR/SSIM 增益表示重建改善；
日志中的 apparent_temperature_MAE 是相机黑体等效温度误差，不能解释为表面温度恢复误差。
固定 thermal SH 可能吸收物理外观误差，后续用合成参数恢复与 SH 消融研究。

回归测试：`PYTHONPATH="$GS_WORK" "$THERMAL_PYTHON" -B tests/test_thermal_stage3.py`。
短程 GPU 检查是流程验证，不是正式 C/R/K 训练结果。
