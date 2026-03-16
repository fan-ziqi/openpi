# π0.5 Subtask 真机数采、微调训练与部署完整指南

> 参考论文: [π0.5: Efficient and Generalizable VLA via Next-Token Autoregression and Flow Matching](https://arxiv.org/abs/2504.16054)

本文档详细介绍如何在真实机器人上使用 π0.5 Subtask 版本进行数据采集、微调训练和部署推理。

---

## 目录

1. [概述与背景](#一概述与背景)
2. [模型架构详解](#二模型架构详解)
3. [数据采集与处理](#三数据采集与处理)
4. [模型训练](#四模型训练)
5. [部署与推理](#五部署与推理)
6. [附录](#六附录)

---

## 一、概述与背景

### 1.1 为什么需要 π0.5？

π0.5 是 Physical Intelligence 提出的分层 Vision-Language-Action (VLA) 模型，专门解决**开放世界泛化**问题——即在训练分布之外的环境（新房间、新光照、长周期任务）中依然能稳定工作。

- **System 2 (语义推理)**: 从 VLM 推断高层语义子任务
- **System 1 (运动控制)**: 基于子任务生成连续动作

### 1.2 核心创新

π0.5 的核心创新是将动作分布分解为:

```
πθ(at:t+H, ℓ̂ | ot, ℓ) = πθ(at:t:t+H | ot, ℓ̂) × πθ(ℓ̂ | ot, ℓ)
```

这种分解使动作分布依赖于预测的高层子任务，而非原始指令，实现了类似思维链的推理能力。

### 1.3 训练策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **联合训练** | 同时训练子任务预测、FAST token 预测和 Flow Matching | 小数据集 |
| **知识隔离** | 两阶段训练: 先微调 VLM，再微调动作专家 | 大数据集、更好泛化 |

---

## 二、模型架构详解

### 2.1 整体架构

π0.5 由以下组件构成:

- **主干网络**: PaliGemma (SigLIP 视觉编码器 + Gemma 2B 语言模型)
- **动作专家**: ~311M 的流匹配生成器 (Gemma 300M)

### 2.2 输入输出

**输入类型**:
- 文本 token: 任务指令、状态信息
- 图像 patch: 多视角相机图像
- Flow matching 中间去噪值

**输出**:
- 文本 token: 高层子任务描述
- 连续动作: 机械臂/夹爪/基座的目标姿态

### 2.3 注意力机制

- 图像/文本/动作 token 使用**双向注意力**
- 动作 token 内部使用**因果注意力** (阶梯状)

### 2.4 两阶段训练流水线 (官方)

**阶段 1: 预训练 (280,000 步)**
- 使用 FAST action tokenizer
- α = 0 (禁用 Flow Matching)
- 所有任务用离散 token 表示

**阶段 2: 后训练 (80,000 步)**
- 联合训练: 下一 token 预测 + Flow Matching
- α = 10.0
- 添加动作专家用于连续动作生成
- 使用 `stop_gradient_flow_to_prefix` 阻止梯度反向传播到 VLM

**联合损失函数**:
```
E[ H(x1:M, fθℓ(ot,ℓ)) + α||ω - at:t+H - fθa(at:t+Hτ,ω, ot,ℓ)||² ]
```

### 2.5 训练数据来源

| 数据集 | 描述 |
|--------|------|
| **MM** | ~400 小时移动操作数据，~100 个家庭环境 |
| **ME** | 非移动机器人数据，多样化家庭环境 |
| **CE** | 跨本体实验室数据 (OXE 数据集) |
| **HL** | 高层子任务预测数据，带语义标注 |
| **WD** | 网络数据 (图像 caption、VQA、物体定位) |
| **VI** | 口头指令数据 - 专家用户提供的语言演示 |

### 2.6 泛化测试

**测试环境**:
- 模拟家庭环境 (定量对比)
- 3 个真实家庭 (训练时未见)
- 任务: 水槽盘子、抽屉物品、洗衣篮、整理床铺

**性能扩展**: 10⁴ 位置时性能接近直接在测试家庭训练的模型

### 2.7 控制参数

- **推理频率**: 50 Hz
- **控制方式**: 直接输出目标姿态 + PD 控制器跟踪，无需轨迹规划

### 2.8 相机配置

- **高层推理**: 使用全部 4 个相机
- **低层推理**: 使用腕部和前向相机

### 2.9 推理流水线

```
┌─────────────────────────────────────────────────────────────┐
│  阶段 1: PREFILL                                            │
│  输入: 图像 + 任务指令                                       │
│  处理: 一次性前向传播，初始化 KV Cache                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段 2: 自回归子任务解码                                     │
│  输入: KV Cache                                             │
│  处理: 逐词预测子任务描述                                     │
│  输出: "move arm to cup"                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段 3: Flow Matching 动作去噪                              │
│  输入: 机器人状态 + 噪声动作                                  │
│  处理: 10 次迭代去噪                                          │
│  输出: 连续动作序列                                          │
└─────────────────────────────────────────────────────────────┘
```

**Flow Matching 公式详解**:

```
训练时:
  time ~ Beta(1.5, 1) * 0.999 + 0.001
  x_t = time * noise + (1 - time) * action
  v_t = model(x_t, time)  # 预测速度
  loss = MSE(v_t, action - noise)

推理时:
  x_0 = noise
  dt = -1.0 / num_steps
  for step in range(num_steps):
      v_t = model(x_t, time)
      x_{t+1} = x_t + dt * v_t
```

### 2.10 关键概念

| 概念 | 说明 |
|------|------|
| **KV Cache** | 键值缓存，避免重复计算；预分配机制 `_init_cache()` |
| **FAST Tokens** | 离散化动作令牌，连接语言和动作 |
| **状态离散化** | 256 bins 离散化，编码为字符串放入 prompt |
| **Action Horizon** | 动作时界，决定动作块长度 (默认 20-50) |
| **分位数归一化** | 使用 1%/99% 分位数归一化动作到 [-1, 1] |

---

## 三、数据采集与处理

### 3.1 硬件准备

**ALOHA 机器人**:
- 2 × WidowX 机械臂 (6 DoF + gripper)
- 4 × RealSense 相机
- 控制柜和电源

**官方移动操作平台**:
- 4 个相机 (前向、后向、双腕部)
- 2 个机械臂 (6 DoF + 平行夹爪)
- 轮式全向基座
- 躯干提升机构 (18-19 DoF)

### 3.2 数据采集脚本

#### 3.2.1 安装 ALOHA 依赖

```bash
# 初始化子模块
git submodule update --init --recursive

# 创建虚拟环境
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate

# 安装依赖
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client
```

#### 3.2.2 数据采集脚本

```python
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

class AlohaDataCollector:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0

    def start_episode(self, task_description: str):
        self.current_episode = {
            'task_description': task_description,
            'timestamp': datetime.now().isoformat(),
            'observations': [],
            'actions': [],
            'subtasks': []  # 子任务标注
        }

    def add_step(self, observation, action, subtask: str = None):
        self.current_episode['observations'].append(observation)
        self.current_episode['actions'].append(action)
        if subtask:
            self.current_episode['subtasks'].append(subtask)

    def end_episode(self):
        episode_file = self.save_dir / f"episode_{self.episode_count:05d}.hdf5"
        with h5py.File(episode_file, 'w') as f:
            f.attrs['task_description'] = self.current_episode['task_description']
            # 保存观察和动作数据...
            if self.current_episode['subtasks']:
                f.create_dataset('subtask',
                    data=np.array(self.current_episode['subtasks'], dtype='S'))
        self.episode_count += 1
```

### 3.3 数据格式要求

```python
{
    # 观察数据
    'observation/state': (T, 14),      # 关节状态
    'observation/cam_high': (T, H, W, 3),
    'observation/cam_low': (T, H, W, 3),
    'observation/cam_left_wrist': (T, H, W, 3),
    'observation/cam_right_wrist': (T, H, W, 3),

    # 动作数据
    'action': (T, 14),                 # 目标关节位置

    # 子任务标注 (Subtask 版本)
    'task': "Pick up the cup...",      # 高层任务
    'subtask': "Move arm to cup",      # 低层子任务
}
```

### 3.4 转换为 LeRobot 格式

```bash
python examples/aloha_real/convert_aloha_data_to_lerobot.py \
    --raw-dir /path/to/hdf5/data \
    --repo-id your_username/your_dataset \
    --video-backend cuda
```

---

## 四、模型训练

### 4.1 环境准备

```bash
# 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --dev

# 下载 FAST Tokenizer
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="physical-intelligence/fast")
PY

# 下载预训练模型
python - <<'PY'
from openpi.training import config as _config
from openpi.shared import download
config = _config.get_config("pi05_base")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
PY
```

### 4.2 训练配置

#### 方案 A: 联合训练 (推荐小数据集)

```python
TrainConfig(
    name="my_task_pi05_subtask_hybrid",
    model=pi05_config.Pi05Config(
        action_horizon=20,
        max_token_len=192,
        subtask_loss_weight=0.15,
        fast_token_loss_weight=0.15,
        flow_matching_loss_weight=1.0,
        fast_tokenizer_path="physical-intelligence/fast",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/path/to/pi05_base/params"
    ),
    data=LeRobotDataConfig(
        repo_id="your_username/your_dataset",
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1000,
        peak_lr=1e-4,
        decay_steps=50000,
        decay_lr=1e-5,
    ),
    num_train_steps=30000,
    batch_size=32,
    fsdp_devices=8,
)
```

#### 方案 B: 知识隔离训练

**Stage 1: 微调 VLM**
```python
TrainConfig(
    name="my_task_pi05_subtask_fast",
    model=pi05_config.Pi05Config(
        subtask_loss_weight=10.0,
        fast_token_loss_weight=1.0,
        flow_matching_loss_weight=0.0,  # 禁用
    ),
    # ...
)
```

**Stage 2: 微调动作专家**
```python
TrainConfig(
    name="my_task_pi05_action_expert",
    model=pi05_config.Pi05Config(
        flow_matching_loss_weight=1.0,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/path/to/stage1/checkpoint"
    ),
    freeze_filter=nnx.All(
        nnx.Param,
        nnx_utils.PathRegex(".*llm.*"),
        nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
    ),
)
```

### 4.3 开始训练

```bash
# 联合训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py my_task_pi05_subtask_hybrid \
    --exp-name=my_experiment --overwrite

# 知识隔离 - Stage 1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py my_task_pi05_subtask_fast \
    --exp-name=my_experiment --overwrite

# 知识隔离 - Stage 2
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py my_task_pi05_action_expert \
    --exp-name=my_experiment --overwrite
```

### 4.4 训练监控

| 指标 | 含义 |
|------|------|
| `loss/total` | 总损失 |
| `loss/subtask` | 子任务预测损失 |
| `loss/fast_token` | FAST token 损失 |
| `loss/flow_matching` | Flow matching 损失 |
| `metrics/accuracy/subtask` | 子任务准确率 |

---

## 五、部署与推理

### 5.1 部署架构

| 模式 | 说明 | 优势 |
|------|------|------|
| **本地部署** | 模型和机器人运行在同一台机器 | 低延迟 |
| **远程部署 (推荐)** | 模型在 GPU 服务器，通过 WebSocket 与机器人通信 | 更强 GPU、隔离依赖 |

### 5.2 服务器启动

```bash
# 方式 1: 使用通用脚本
uv run scripts/serve_policy.py --env=ALOHA --port=8000

# 方式 2: 指定自定义检查点
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid \
    --port=8000

# 方式 3: 异步推理服务器
python scripts/async_pi05/async_pi05_websocket_server.py \
    --config my_task_pi05_action_expert \
    --checkpoint /path/to/checkpoint \
    --port 8765
```

### 5.3 客户端连接

```python
from openpi_client import websocket_client_policy

# 连接服务器
client = websocket_client_policy.WebsocketClientPolicy(
    host="192.168.1.100",
    port=8765
)

# 构建观察数据
observation = {
    "state": np.ones((14,)),
    "images": {
        "cam_high": img,
        "cam_low": img,
        "cam_left_wrist": img,
        "cam_right_wrist": img,
    },
    "prompt": "do something",
}

# 获取动作
action = client.infer(observation)["actions"]
```

### 5.4 观察数据格式

**ALOHA**:
```python
{
    "state": (14,),           # 关节状态
    "images": {
        "cam_high": (3, 224, 224),    # uint8
        "cam_low": (3, 224, 224),
        "cam_left_wrist": (3, 224, 224),
        "cam_right_wrist": (3, 224, 224),
    },
    "prompt": "...",
}
```

**DROID**:
```python
{
    "observation/exterior_image_1_left": (224, 224, 3),
    "observation/wrist_image_left": (224, 224, 3),
    "observation/joint_position": (7,),
    "observation/gripper_position": (1,),
    "prompt": "...",
}
```

**LIBERO**:
```python
{
    "observation/state": (8,),           # 8 维状态
    "observation/image": (224, 224, 3),
    "observation/wrist_image": (224, 224, 3),
    "prompt": "...",
}
```

### 5.5 推理 API 返回格式

```python
{
    "actions": np.array,  # (action_horizon, action_dim)
    "server_timing": {
        "infer_ms": 100.0,      # 推理耗时 (ms)
        "prev_total_ms": 105.0, # 上次总耗时 (ms)
    }
}
```

**Subtask 版本额外输出**:
```python
{
    "actions": np.array,           # 预测的连续动作
    "subtask": "Move arm to cup",  # 生成的子任务描述
    "action_tokens": [...],        # 生成的 FAST action tokens
}
```

### 5.6 GPU 要求

| 场景 | GPU 要求 |
|------|----------|
| 推理 | ≥8GB VRAM (RTX 4090) |
| LoRA 微调 | ≥22.5GB VRAM |
| 全参数微调 | ≥70GB VRAM (A100/H100) |

**性能优化**:
```bash
# JAX 内存优化
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

- 使用异步推理服务器减少延迟
- 客户端预处理图像减少带宽
- 图像尺寸 224×224，uint8 格式

### 5.7 Jetson Orin 端侧部署

> 基于社区分享的部署经验整理

Jetson Orin 可以作为端侧部署设备，推理延迟约 **1.2 秒**。

**环境配置**:
- JetPack 6.1, Ubuntu 22.04 (aarch64)
- Python 3.10
- CUDA 12.6, cuDNN 9.3, TensorRT 10.3

**详细步骤**:

1. **拉取仓库**
```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
```

2. **修改 Python 版本**
```bash
echo "3.10" > .python-version
# 修改 pyproject.toml: requires-python >=3.10, target-version py310
```

3. **修复 download.py 兼容性**
```python
# src/openpi/shared/download.py 第 169 行
date = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
```

4. **安装依赖**
```bash
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

5. **安装 Jetson PyTorch**
```bash
# 下载 PyTorch 2.3 aarch64 版本
uv pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
uv pip install torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
uv pip install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```

6. **替换 transformers**
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.10/site-packages/transformers/
```

7. **转换 JAX → PyTorch**
```bash
.venv/bin/python3.10 examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --config_name pi05_droid \
    --output_path torch_pi05_droid/

cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets/ torch_pi05_droid/
```

8. **禁用 Torch Compile** (关键)
```python
import torch
torch._dynamo.config.suppress_errors=True
```

**推理延迟**: ~1.2 秒 (Jetson Orin 64GB)

**性能统计**:
| 指标 | 平均值 | P50 | P90 |
|------|--------|-----|-----|
| 推理延迟 | ~1170ms | ~1200ms | ~1220ms |

### 5.8 Docker 部署

```yaml
version: '3.8'
services:
  inference-server:
    build:
      context: .
      dockerfile: Dockerfile.inference
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - OPENPI_DATA_HOME=/data
    ports:
      - "8765:8765"
    volumes:
      - ./checkpoints:/data/checkpoints
```

```bash
sudo xhost +local:docker
docker compose up --build
```

---

## 六、附录

### A. 官方模型清单

**基础模型 (微调)**:
| 模型 | 路径 |
|------|------|
| π0 | gs://openpi-assets/checkpoints/pi0_base |
| π0-FAST | gs://openpi-assets/checkpoints/pi0_fast_base |
| π0.5 | gs://openpi-assets/checkpoints/pi05_base |

**微调模型 (推理)**:
| 模型 | 路径 |
|------|------|
| π0-FAST-DROID | gs://openpi-assets/checkpoints/pi0_fast_droid |
| π0.5-DROID | gs://openpi-assets/checkpoints/pi05_droid |
| π0.5-LIBERO | gs://openpi-assets/checkpoints/pi05_libero |

### B. 常见问题

**训练问题**:
| 问题 | 解决方案 |
|------|----------|
| Loss 不下降 | 调整学习率 |
| 子任务准确率低 | 检查数据标注 |
| GPU OOM | 减小 batch_size |

**推理问题**:
| 问题 | 解决方案 |
|------|----------|
| 动作抖动 | 检查相机质量 |
| 延迟过高 | 使用异步服务器 |
| 任务失败 | 增加训练数据多样性 |

**模型下载问题**:
| 问题 | 解决方案 |
|------|----------|
| 下载失败 | 使用 `gsutil` 手动下载 |
| 速度慢 | 配置国内镜像或使用网盘 |

```bash
# 手动下载检查点
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base /local/path/
```

### C. 参考资源

- [π0.5 论文](https://arxiv.org/abs/2504.16054)
- [Physical Intelligence 网站](https://www.pi.website)
- [FAST Tokenizer](https://huggingface.co/physical-intelligence/fast)
- [LeRobot 数据集](https://github.com/huggingface/lerobot)
- [ALOHA 硬件](https://github.com/tonyzhaozh/aloha)
- [π0.5 Subtask 复现代码](https://github.com/Ke-Wang1017/openpi_subtask)
