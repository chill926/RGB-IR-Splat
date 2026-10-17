# Stage3 hyperparameter tuning on 2080

模型与材料标签沿用 D 共同起点；固定 epsilon0、geometry、opacity、RGB/thermal SH。
新增 Stage3 独立 T/E LR、结束 LR、lr_hold_fraction；实际值写入 checkpoint、comparison lock 和日志。
不修改原 D、旧 Stage3 模型或共同起点。

## 推荐：两阶段搜索

在 2080 的终端运行：

```bash
export GS_WORK=/home/liupengyu/graduate
CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python -B -u \
  "$GS_WORK/tools/tune_stage3.py" \
  --output "$GS_WORK/output/RoadBlock_stage3_tune_run1"
```

依次跑 12 个试验，每个 2000 步、每 100 步验证：

1. C：T LR={3e-5,1e-4,3e-4} × E LR={1e-5,1e-4}，终值为起始值的 1%。
2. 在选出的 T/E 设置上分别搜索 R 和 K 的 lambda={1e-3,1e-4,1e-5}。

所有试验 material raw LR=1e-4→1e-6，hold fraction=0.1，seed=0。
默认使用已经验证的 `output/RoadBlock_stage3_D_pilot_run1_common/thermal_stage3_common.pt`，
仅 matte_plastic 开启 K/R，unknown 保留。可以用 --common 指定另一个已导出的共同起点。

搜索只读取 fit/validation 信号，不评价测试集。每个试验 log 在搜索目录，
结果写入 trial_results.json、selection.json、best_settings.env。
首先按 best validation objective 选择；若 best 都为 step0，按 final validation objective 打破平局，
并明确记录 C_all_best_steps_zero。这表示选择较稳定的试跑配置，不能声称已超过 D。
R/K 每个正则档共用比较锁；各档 C 的材料正则恒为零，因此不重复训练 C。
最终正式 C/R/K 会使用统一的两种选定正则参数和完整预算作匹配比较。

### 搜索后完整训练

```bash
source "$GS_WORK/output/RoadBlock_stage3_tune_run1/best_settings.env"
CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_roadblock_stage3.sh"
```

生成的设置指定 C/R/K 各 10000 步；输出默认为
`output/RoadBlock_stage3_tune_run1_full_{C,R,K}`。训练后评价 best 的全部 IR/RGB 划分。
参数文件只生成启动设置，搜索结束不会自动启动完整训练。

### 预览和恢复

在搜索命令后加 `--dry_run` 可以校验输入并打印搜索计划，不创建输出或训练。
加 `--resume` 可以复用同参数、同源码、已完整完成的试验。
存在中断的半成品时会停止并保留文件，应使用新搜索目录，不会自动覆盖。
如果仅用户终端显示过程并希望后台持续运行，可在 tmux 会话里执行上述命令。

## 可选：只试一组低 LR

```bash
export GS_WORK=/home/liupengyu/graduate
export STAGE3_RUN=RoadBlock_stage3_D_lrpilot_run1
CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_roadblock_stage3.sh"
```

新的 shell 默认是各 2000 步、eval100、T=1e-4→1e-6、E=1e-5→1e-7、
material=1e-4→1e-6、hold=0.1、lambda_k=lambda_delta=1e-4。
用 STAGE3_T_LR、STAGE3_T_LR_FINAL、STAGE3_E_LR、STAGE3_E_LR_FINAL、
STAGE3_MATERIAL_LR、STAGE3_MATERIAL_LR_FINAL、STAGE3_HOLD、
STAGE3_LAMBDA_K、STAGE3_LAMBDA_DELTA、STAGE3_STEPS、STAGE3_EVAL_EVERY 覆盖。
已有输出会被拒绝，下一次使用新的 STAGE3_RUN。

## 直接调用训练入口

```bash
/home/liupengyu/miniconda3/envs/3dgs/bin/python -B -u train_thermal_stage3.py \
  --common output/RoadBlock_stage3_D_pilot_run1_common/thermal_stage3_common.pt \
  --branch K --comparison_protocol output/my_stage3_comparison.json \
  -m output/my_stage3_K --steps 2000 --eval_every 100 \
  --temperature_lr 1e-4 --temperature_lr_final 1e-6 \
  --environment_lr 1e-5 --environment_lr_final 1e-7 \
  --material_lr 1e-4 --material_lr_final 1e-6 \
  --lr_hold_fraction 0.1 --lambda_k 1e-4 --lambda_delta_epsilon 1e-4
```

直接入口不指定 T/E LR 或 hold 时继承 common，以保留已有调用的语义。
直接入口的材料 LR 与 lambda 默认已降低。
超参搜索可以改善优化稳定性，不能保证解决自由温度与发射率的补偿退化。
