# graduate：2080 服务器部署

部署日期：2026-10-11（北京时间）。项目位置：`/home/liupengyu/graduate`，服务器 SSH 别名：`2080`。

本地原目录中的代码、数据、Git 信息、已有掩码和训练输出已上传；仅排除了 macOS 的 `.DS_Store` 文件。8,595 个文件、5,252,426,794 字节通过 SHA-256 校验。本地原目录未修改。

## 两个独立环境

| 环境 | Python | PyTorch / TorchVision | CUDA 工具链 | 用途 |
|---|---|---|---|---|
| `3dgs` | 3.10.22 | 1.13.1+cu116 / 0.14.1+cu116 | `/usr/local/cuda-11.6` | 3DGS 主干及热场代码 |
| `sam2` | 3.10.22 | 2.5.1+cu118 / 0.20.1+cu118 | `/usr/local/cuda-11.8` | 图像、视频分割掩码 |

3DGS 按项目 `requirements.server.txt` 安装，并补齐 imagecodecs、TensorBoard；项目提供的 simple-knn 和定制光栅器已编译。SAM2 使用项目的 `third_party/sam2` 源码和原目录已有的 SAM2.1 Hiera Tiny 权重，CUDA 后处理扩展已编译。

两个环境的 `CUDA_HOME` 已分别保存，`TORCH_CUDA_ARCH_LIST=7.5` 对应服务器 RTX 2080 Ti。服务器的系统 Ninja 1.10.0 可供后续编译使用。若需查看编译器版本，激活环境后运行 `"$CUDA_HOME/bin/nvcc" --version`。

[SAM2 官方安装说明](https://github.com/facebookresearch/sam2#installation)要求较新的 PyTorch，因此与项目固定的 3DGS 依赖隔离安装。CUDA 版 PyTorch 对照[官方版本安装说明](https://pytorch.org/get-started/previous-versions/)。

## 登录和激活

在本机终端：

```bash
ssh -b 100.79.94.121 2080
```

`100.79.94.121` 是部署时 VPN 的源地址；VPN 重新连接后若地址变化，应替换为当前地址。普通 `ssh 2080` 能连通时也可直接使用。

服务器的 Bash 已初始化 Conda，新登录可直接激活。已有终端可先执行 `source ~/.bashrc`。

分别在两个服务器终端中使用：

```bash
cd /home/liupengyu/graduate
conda activate 3dgs
```

```bash
cd /home/liupengyu/graduate
conda activate sam2
```

## 3DGS 主干示例

以现有 Truck 场景的 RGB 几何主干为例，在 `3dgs` 环境中运行：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s data/RGBT-Scenes/RGBT-Scenes/Truck \
  -m output/Truck_stage1_server \
  --data_branch rgbt --rgb_geometry_stage --eval --iterations 30000
```

数据目录保留了原来的双层 `RGBT-Scenes/RGBT-Scenes`。使用新的输出目录可与已有结果分别保存。正式训练由你启动，本次只进行了环境和 CUDA 算子验证。

原有部分批量 worker 脚本包含 `/home/imglab/...` 等旧服务器路径，批量使用前需要将其项目路径和 Python 路径改为本次部署位置。上述单场景命令已使用当前目录。

## SAM2 分割掩码示例

在 `sam2` 环境中，准备 `material_config.json` 和 `material_prompts.json`，可参照仓库中的同名 `.example.json` 文件。提示 JSON 的帧键应使用实际图像文件名的 stem，例如 `FLIR3055`，并填入你标注的框/点及材料名称。

逐图像生成材料标签 PNG 和置信度数组：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/generate_sam2_metal_masks.py \
  --image_dir data/RGBT-Scenes/RGBT-Scenes/Truck/rgb/train \
  --prompts material_prompts.json \
  --material_config material_config.json \
  --output_dir data/RGBT-Scenes/RGBT-Scenes/Truck/material_masks/train \
  --checkpoint third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --device cuda
```

有序图像序列可使用 `tools/generate_sam2_material_video_masks.py`，参数相同；为对象提供稳定的 `object_id`，可从关键帧提示传播掩码。正式分割由你提供提示后运行。

材料标签是你在配置和提示中指定的类别，SAM2 负责分割区域。未知像素默认标记为 255。原有 `masks_sam2` 已保留。

## 验证记录与版本清单

已通过：

- 8,595 个原文件的 SHA-256 校验。
- 两个环境的 `pip check`。
- 3DGS CUDA KNN、定制光栅器前向及反向梯度、训练入口导入与 `--help`。
- LPIPS/VGG GPU 评估；VGG 权重已缓存，并核对官方文件名中的哈希前缀。
- SAM2 CUDA 连通域算子、Tiny 权重加载、GPU 图像预测。
- 项目图像和视频分割脚本在合成测试帧上生成标签 PNG、置信度数组，输出尺寸、类别和有限数值检查通过。

服务器记录位于 `/home/liupengyu/.cache/graduate-setup/`，包含安装日志、验证脚本、环境 YAML 和 pip 完整版本清单。此次没有执行正式训练或批量修改数据集。

SAM2 Tiny 权重：`third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt`，SHA-256：`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。

## 后续需要联网安装包

服务器本次无法直连包仓库。临时隧道已关闭；需要再次联网安装依赖时，在本机终端运行随附脚本：

```bash
python3 /Users/liupengyu/Documents/Codex/2026-10-11/qing/outputs/graduate-2080联网代理.py
```

脚本会建立只监听回环地址的 HTTPS 包代理，通过 SSH 反向端口映射提供给服务器，并打开带临时 `http_proxy`/`https_proxy` 的服务器终端。可在该终端激活环境后使用 pip；退出 SSH 会自动关闭代理。

VPN 地址改变时添加 `--bind-address 当前地址`；无需绑定源地址时可使用 `--bind-address ''`。远端 18080 被占用时可添加 `--remote-port 18081`。


## 2026-10-12：RoadBlock RGB Stage1 加载修复

已修复服务器 `scene/__init__.py` 和 `scene/dataset_readers.py`：启用 `--rgb_geometry_stage` 时只读取 RGB 图像，不解码热图、不创建伪造的物理信号。默认热图加载的伪彩检查仍保留。

5 项回归检查通过；RoadBlock 原分辨率下的 20 步 GPU 短训、训练/测试视图评估及模型保存通过。正式 30,000 步任务尚未代为启动，原先的训练命令可重新执行。

修改前代码备份：`/home/liupengyu/.cache/graduate-setup/backups/stage1-rgb-only-20261012-132257/`。本地 Desktop 原代码未修改；可通过随附 `RGB-Stage1热图加载修复.patch` 同步修复，避免后续上传旧版文件覆盖服务器修复。

RoadBlock 当前 `thermal/` 图像被检测为伪彩，Stage2 仍需有效的单通道热信号或有依据的辐射预处理。材料 mask 不会使伪彩图变成辐射观测。
