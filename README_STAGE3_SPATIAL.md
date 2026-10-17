# RoadBlock Stage3: local temperature prior and float signal handoff

Stage2/D → common → Stage3 直接传递 T/E/material/geometry/SH 浮点检查点。
未保存伪彩图再反解 DN。渲染域是归一化 Q，Q=DN+PlanckO；
三通道渲染器接收同一标量信号复制三次，并不是调色板 RGB。
固定显示映射只参与已有 display 辅助损失和伪彩评价，不作为下一阶段物理输入。
本次没有重跑或更改 Stage2 模型，已有共同起点可直接使用。

## 温度空间先验

六个局部近邻，去重为无向边；仅两端均有原共同起点 fit support 的边有效。
超过正距离中位数三倍的连接被移除。
权重为距离高斯权重 × 固定 D RGB DC 相似度 × sqrt(opacity_i*opacity_j) × 材料权重。
RGB sigma=0.15；已知异类边权重0.1；任一端unknown边权重0.1。
unknown不会被视为统一材料；全部权重基于冻结状态，不参与优化。

损失为 `sum(w*SmoothL1((Ti-Tj)/10K,beta=1))/sum(w)`。
无真实温度监督，没有合成数据或 K-UQ；冻结 epsilon0、geometry、opacity、RGB/thermal SH。
训练信号与现有 display 权重不变，验证选择不加入训练温度正则项。
每个 checkpoint/日志记录实际 lambda_spatial、图 hash、图过滤数量、温度正则和梯度。

## 搜索并运行匹配的 C/R/K 试跑

```bash
export GS_WORK=/home/liupengyu/graduate
CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python -B -u \
  "$GS_WORK/tools/tune_stage3_spatial.py" \
  --output "$GS_WORK/output/RoadBlock_stage3_spatial_run1" --evaluate_selected
```

默认先跑 C 的 lambda_spatial={0,1e-4,1e-3,1e-2}，每组2000步，eval100。
随后 R/K 使用选出的同一 lambda 和比较锁，各跑2000步，再评价 C/R/K best 的全部 IR/RGB 划分。
超参继承已完成的 RoadBlock_stage3_tune_run1：
T=3e-5→3e-7，E=1e-4→1e-6，material=1e-4→1e-6，
lambda_k=lambda_delta=1e-5，hold=0.1，seed=0。
选参数只依据验证集；测试集在选定并完成三分支训练后才评价。
若最佳强度为0或best仍为step0，会照实报告，不强行选择非零先验。

支持 --dry_run、--resume。恢复时跳过已完成的匹配试验，
将未完成的试验/日志另存为 interrupted_* 后从共同起点重跑；不会覆盖旧模型。
search.lock 防止同一目录同时跑两个搜索。

完整预算可在试跑后执行：

```bash
source "$GS_WORK/output/RoadBlock_stage3_spatial_run1/best_settings.env"
CUDA_VISIBLE_DEVICES=0 bash "$GS_WORK/scripts/run_roadblock_stage3.sh"
```

完整三分支均使用同一组超参，输出目录以 RoadBlock_stage3_spatial_run1_full 开头。
也可在普通启动脚本中通过 STAGE3_SPATIAL 指定强度；其默认值仍为0，旧启动语义保留。

需要持续后台运行时可使用 tmux，或把 stdout/stderr 重定向到一个新日志。
各组训练进度见搜索目录的 C_spatial_00.log 至03.log、R_selected_spatial.log、K_selected_spatial.log。
