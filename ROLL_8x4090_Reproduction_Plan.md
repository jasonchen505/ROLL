# ROLL 框架 8×4090 完整复现学习计划

> 基于 8 张 RTX 4090 (24GB) 的算力限制，制定分阶段、可执行的 ROLL 框架学习与复现计划

---

## 目录

1. [硬件资源评估与限制分析](#1-硬件资源评估与限制分析)
2. [整体复现计划概览](#2-整体复现计划概览)
3. [Phase 1: 环境搭建与单机验证](#3-phase-1-环境搭建与单机验证)
4. [Phase 2: Agentic Pipeline 复现](#4-phase-2-agentic-pipeline-复现)
5. [Phase 3: RLVR Pipeline 复现](#5-phase-3-rlvr-pipeline-复现)
6. [Phase 4: 算法对比实验](#6-phase-4-算法对比实验)
7. [Phase 5: 工程优化与性能调优](#7-phase-5-工程优化与性能调优)
8. [关键配置参数对照表](#8-关键配置参数对照表)
9. [常见问题与解决方案](#9-常见问题与解决方案)
10. [学习成果检验清单](#10-学习成果检验清单)

---

## 1. 硬件资源评估与限制分析

### 1.1 4090 vs A100 对比

| 参数 | RTX 4090 | A100 80GB | 影响 |
|------|----------|-----------|------|
| **显存** | 24GB | 80GB | 限制模型大小和batch size |
| **FP16算力** | 82.6 TFLOPS | 312 TFLOPS | 训练速度慢3-4倍 |
| **卡间通信** | PCIe 4.0 | NVLink | 多卡并行效率低 |
| **显存带宽** | 1008 GB/s | 2039 GB/s | 推理速度慢2倍 |

### 1.2 4090 上的限制

```
模型大小限制：
- 0.5B 模型：可正常训练，batch size 可以较大
- 1.5B 模型：可训练，需要减小 batch size
- 3B 模型：需要 gradient checkpointing + CPU offload
- 7B 模型：需要 FSDP2 + CPU offload，batch size 很小
- 8B+ 模型：基本不可行（显存不足）

训练速度限制：
- 单步训练时间约为 A100 的 3-4 倍
- 推理速度约为 A100 的 2-3 倍
- 总训练时间需要相应延长

多卡并行限制：
- 无 NVLink，卡间通信走 PCIe
- 数据并行效率约为 70-80%（vs NVLink 的 95%+）
- 不适合大规模模型并行（TP/PP）
```

### 1.3 推荐模型选择

| 模型 | 参数量 | 4090 可行性 | 推荐配置 |
|------|--------|-------------|----------|
| Qwen2.5-0.5B | 0.5B | ✅ 推荐 | 8卡，batch size 16-32 |
| Qwen2.5-1.5B | 1.5B | ✅ 可行 | 8卡，batch size 8-16 |
| Qwen3-4B | 4B | ⚠️ 困难 | 8卡，需要 CPU offload |
| Qwen3-8B | 8B | ⚠️ 很困难 | 8卡，需要大量优化 |
| Qwen3-30B | 30B | ❌ 不可行 | 显存不足 |

**推荐：使用 Qwen2.5-0.5B 作为入门模型，后续可尝试 1.5B**

---

## 2. 整体复现计划概览

### 2.1 五个阶段

```
Phase 1: 环境搭建与单机验证 (1-2天)
├── Docker 环境搭建
├── 依赖安装
├── 单卡验证基础功能
└── 理解配置系统

Phase 2: Agentic Pipeline 复现 (2-3天)
├── FrozenLake 环境训练
├── 理解 Agentic RL 流程
├── 调整配置适应 4090
└── 观察训练曲线

Phase 3: RLVR Pipeline 复现 (2-3天)
├── 数学推理任务训练
├── 理解 RLVR 流程
├── 配置多域奖励
└── 分析训练结果

Phase 4: 算法对比实验 (3-4天)
├── GRPO vs PPO 对比
├── 超参调优实验
├── 消融实验
└── 结果分析

Phase 5: 工程优化与性能调优 (2-3天)
├── 显存优化
├── 训练速度优化
├── 多卡效率优化
└── 性能基准测试
```

### 2.2 时间线

```
Week 1: Phase 1 + Phase 2
  - 完成环境搭建
  - 跑通 Agentic Pipeline
  - 理解核心流程

Week 2: Phase 3 + Phase 4
  - 跑通 RLVR Pipeline
  - 完成算法对比
  - 分析实验结果

Week 3: Phase 5 + 总结
  - 性能优化
  - 撰写学习笔记
  - 准备面试材料
```

---

## 3. Phase 1: 环境搭建与单机验证

### 3.1 Docker 环境搭建

```bash
# Step 1: 安装 Docker 和 NVIDIA Container Toolkit
curl -fsSL https://github.com/alibaba/ROLL/blob/main/scripts/install_docker_nvidia_container_toolkit.sh | sudo bash

# Step 2: 启动 Docker 容器
sudo docker run -dit \
  --gpus all \
  -p 9001:22 \
  --ipc=host \
  --shm-size=10gb \
  -v /data/home/yizhou:/workspace \
  roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084 \
  /bin/bash

# Step 3: 进入容器
sudo docker exec -it <container_id> /bin/bash

# Step 4: 验证 GPU
nvidia-smi
```

### 3.2 安装 ROLL

```bash
# 进入工作目录
cd /workspace

# 克隆 ROLL 仓库
git clone https://github.com/alibaba/ROLL.git
cd ROLL

# 安装依赖
pip install -r requirements_torch260_vllm.txt -i https://mirrors.aliyun.com/pypi/simple/

# 安装 ROLL（开发模式）
pip install -e .
```

### 3.3 单卡验证

```bash
# 使用最简单的配置验证
# 创建单卡配置文件
cat > examples/4090_test/single_gpu_test.yaml << 'EOF'
hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "4090_single_gpu_test"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
system_envs:
  USE_MODELSCOPE: '1'

track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard

checkpoint_config:
  type: file_system
  output_dir: ./output/checkpoints

num_gpus_per_node: 1

max_steps: 10
save_steps: 10000
logging_steps: 1
eval_steps: 5
resume_from_checkpoint: false

rollout_batch_size: 4
val_batch_size: 4
sequence_length: 2048

advantage_clip: 0.2
ppo_epochs: 1
adv_estimator: "grpo"
init_kl_coef: 0.0
whiten_advantages: true
entropy_loss_coef: 0
max_grad_norm: 1.0

pretrain: Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct

actor_train:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 4
    warmup_steps: 2
    lr_scheduler_type: cosine
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,1))
  infer_batch_size: 1

actor_infer:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: 64
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: 1
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.6
      block_size: 16
      load_format: auto
  device_mapping: list(range(0,1))
  infer_batch_size: 1

reference:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(0,1))
  infer_batch_size: 1

reward_normalization:
  grouping: tags
  method: identity

train_env_manager:
  max_env_num_per_worker: 2
  num_env_groups: 1
  group_size: 1
  tags: [FrozenLake]
  num_groups_partition: [1]
  generating_args:
    max_new_tokens: 32
    top_p: 0.99
    top_k: 100
    temperature: 0.99
    num_return_sequences: 1

val_env_manager:
  max_env_num_per_worker: 1
  num_env_groups: 1
  group_size: 1
  tags: [FrozenLake]
  num_groups_partition: [1]
  generating_args:
    max_new_tokens: 32
    top_p: 0.99
    top_k: 100
    temperature: 0.2
    num_return_sequences: 1

max_tokens_per_step: 32

custom_envs:
  FrozenLake:
    ${custom_env.FrozenLake}
EOF

# 运行测试
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"
python examples/start_agentic_pipeline.py --config_path examples/4090_test --config_name single_gpu_test
```

### 3.4 学习目标

```
✅ 掌握 ROLL 的环境配置
✅ 理解 YAML 配置系统的结构
✅ 跑通单卡 Agentic Pipeline
✅ 观察训练日志和 TensorBoard
✅ 理解各配置参数的含义
```

---

## 4. Phase 2: Agentic Pipeline 复现

### 4.1 目标

在 8×4090 上完整运行 FrozenLake Agentic RL 训练，理解：
- Agentic Pipeline 的工作流程
- 环境管理和轨迹收集
- GRPO 算法在 Agent 任务中的应用

### 4.2 配置文件

```yaml
# examples/4090_config/agentic_frozen_lake_8x4090.yaml
defaults:
  - ../config/traj_envs@_here_

hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "agentic_frozen_lake_8x4090"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
render_save_dir: ./output/render
system_envs:
  USE_MODELSCOPE: '1'

track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard/agentic_frozen_lake

checkpoint_config:
  type: file_system
  output_dir: ./output/checkpoints/${exp_name}

num_gpus_per_node: 8

max_steps: 500
save_steps: 100
logging_steps: 1
eval_steps: 20
resume_from_checkpoint: false

# 4090 优化：减小 batch size
rollout_batch_size: 64
val_batch_size: 64
sequence_length: 2048

advantage_clip: 0.2
ppo_epochs: 1
adv_estimator: "grpo"
init_kl_coef: 0.0
whiten_advantages: true
entropy_loss_coef: 0
max_grad_norm: 1.0

pretrain: Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct

actor_train:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    # 4090 优化：减小 per_device batch size
    per_device_train_batch_size: 2
    # 增加 gradient accumulation 补偿
    gradient_accumulation_steps: 32
    warmup_steps: 20
    lr_scheduler_type: cosine
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,8))
  infer_batch_size: 2

actor_infer:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    # 4090 优化：减小生成长度
    max_new_tokens: 64
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: 1
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: vllm
    strategy_config:
      # 4090 优化：减小显存占用
      gpu_memory_utilization: 0.6
      block_size: 16
      load_format: auto
  device_mapping: list(range(0,8))
  infer_batch_size: 1

reference:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(0,8))
  infer_batch_size: 2

reward_normalization:
  grouping: traj_group_id
  method: mean_std

train_env_manager:
  max_env_num_per_worker: 8
  # 4090 优化：减小环境组数
  num_env_groups: 16
  group_size: 4
  tags: [FrozenLake]
  num_groups_partition: [16]
  generating_args:
    max_new_tokens: 64
    top_p: 0.99
    top_k: 100
    temperature: 0.99
    num_return_sequences: 1

val_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 64
  group_size: 1
  tags: [FrozenLake]
  num_groups_partition: [64]
  generating_args:
    max_new_tokens: 64
    top_p: 0.99
    top_k: 100
    temperature: 0.2
    num_return_sequences: 1

max_tokens_per_step: 64

custom_envs:
  FrozenLake:
    ${custom_env.FrozenLake}
```

### 4.3 运行命令

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

# 运行 Agentic Pipeline
python examples/start_agentic_pipeline.py \
  --config_path examples/4090_config \
  --config_name agentic_frozen_lake_8x4090
```

### 4.4 观察指标

```
核心指标（TensorBoard）：
- val/score/mean: 验证得分，应逐渐上升
- actor/total_loss: 训练损失，应逐渐下降
- actor/kl_loss: KL散度，应保持稳定
- system/tps: 吞吐量，观察训练效率

预期结果：
- 训练 500 步后，FrozenLake 成功率应达到 50-70%
- 单步训练时间约 10-20 秒
- 总训练时间约 2-3 小时
```

### 4.5 学习目标

```
✅ 理解 Agentic Pipeline 的完整流程
✅ 掌握环境管理器的工作机制
✅ 理解 GRPO 算法在 Agent 任务中的应用
✅ 学会观察和分析训练指标
✅ 理解配置参数对训练的影响
```

---

## 5. Phase 3: RLVR Pipeline 复现

### 5.1 目标

在 8×4090 上运行数学推理 RLVR 训练，理解：
- RLVR Pipeline 的工作流程
- 多域奖励系统的配置
- 数学推理任务的训练特点

### 5.2 配置文件

```yaml
# examples/4090_config/rlvr_math_8x4090.yaml
hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "rlvr_math_8x4090"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
system_envs:
  USE_MODELSCOPE: '1'

track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard/rlvr_math

checkpoint_config:
  type: file_system
  output_dir: ./output/checkpoints/${exp_name}

num_gpus_per_node: 8

max_steps: 300
save_steps: 50
logging_steps: 1
eval_steps: 10
resume_from_checkpoint: false

# 4090 优化配置
rollout_batch_size: 32
prompt_length: 512
response_length: 1024
num_return_sequences_in_group: 4

ppo_epochs: 1
adv_estimator: "grpo"

# clip
value_clip: 0.5
reward_clip: 10
advantage_clip: 2.0
dual_clip_loss: true

# normalize
norm_mean_type: ~
norm_std_type: ~

# data mask
max_len_mask: true
difficulty_mask: true
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95

# advantage
whiten_advantages: true

pretrain: Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct

validation:
  data_args:
    template: qwen2_5
    file_name:
      - data/math_benchmarks.jsonl
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.6
    top_k: 50
    num_beams: 1
    temperature: 0.6
    num_return_sequences: 1

actor_train:
  model_args:
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 16
    warmup_steps: 10
    num_train_epochs: 50
  data_args:
    template: qwen2_5
    file_name:
      - data/math_deepmath_deal.jsonl
    domain_interleave_probs:
      math_rule: 1.0
    dataset_dir: data
    messages: messages
    interleave_probs: "1.0"
    preprocessing_num_workers: 4
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      use_distributed_optimizer: true
      recompute_granularity: full
  device_mapping: list(range(0,8))
  infer_batch_size: 2

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: ${num_return_sequences_in_group}
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.6
      block_size: 16
      max_model_len: 2048
  device_mapping: list(range(0,8))
  infer_batch_size: 1

reference:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(0,8))
  infer_batch_size: 2

rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    model_args:
      model_name_or_path: ${reward_pretrain}
    data_args:
      template: qwen2_5
    tag_included: [deepmath_103k, aime]
    world_size: 8
    infer_batch_size: 1
```

### 5.3 数据准备

```bash
# 下载数学数据集（如果需要）
cd /workspace/ROLL

# 检查数据目录
ls -la data/

# 如果数据不存在，可以从 HuggingFace 下载
# 或者使用 ROLL 自带的示例数据
```

### 5.4 运行命令

```bash
cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

# 运行 RLVR Pipeline
python examples/start_rlvr_pipeline.py \
  --config_path examples/4090_config \
  --config_name rlvr_math_8x4090
```

### 5.5 学习目标

```
✅ 理解 RLVR Pipeline 的完整流程
✅ 掌握多域奖励系统的配置
✅ 理解数学推理任务的训练特点
✅ 学会配置和调优训练参数
✅ 分析 RLVR 训练曲线
```

---

## 6. Phase 4: 算法对比实验

### 6.1 实验设计

| 实验 | 算法 | 配置差异 | 预期结果 |
|------|------|----------|----------|
| Exp 1 | GRPO | num_return=4 | 基线 |
| Exp 2 | GRPO | num_return=8 | 方差更小 |
| Exp 3 | Reinforce | num_return=4 | 更简单 |
| Exp 4 | PPO (GAE) | 需要 Critic | 更精细 |

### 6.2 GRPO vs Reinforce 对比配置

```yaml
# GRPO 配置
adv_estimator: "grpo"
num_return_sequences_in_group: 4
use_kl_loss: true
kl_loss_coef: 0.001

# Reinforce 配置
adv_estimator: "reinforce"
num_return_sequences_in_group: 4
use_kl_loss: true
kl_loss_coef: 0.001
```

### 6.3 实验脚本

```bash
#!/bin/bash
# examples/4090_scripts/run_algorithm_comparison.sh

set -e

cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

# 实验 1: GRPO (baseline)
echo "Running GRPO baseline..."
python examples/start_rlvr_pipeline.py \
  --config_path examples/4090_config \
  --config_name rlvr_math_grpo_baseline

# 实验 2: GRPO (more samples)
echo "Running GRPO with more samples..."
python examples/start_rlvr_pipeline.py \
  --config_path examples/4090_config \
  --config_name rlvr_math_grpo_more_samples

# 实验 3: Reinforce
echo "Running Reinforce..."
python examples/start_rlvr_pipeline.py \
  --config_path examples/4090_config \
  --config_name rlvr_math_reinforce

echo "All experiments completed!"
```

### 6.4 结果分析

```python
# scripts/analyze_results.py
import os
import json
import matplotlib.pyplot as plt

def load_tensorboard_logs(log_dir):
    """加载 TensorBoard 日志"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    metrics = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]
    
    return metrics

def plot_comparison(results, output_dir):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 训练损失对比
    ax = axes[0, 0]
    for name, metrics in results.items():
        if 'actor/total_loss' in metrics:
            steps, values = zip(*metrics['actor/total_loss'])
            ax.plot(steps, values, label=name)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    
    # 2. 验证得分对比
    ax = axes[0, 1]
    for name, metrics in results.items():
        if 'val/score/mean' in metrics:
            steps, values = zip(*metrics['val/score/mean'])
            ax.plot(steps, values, label=name)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Score')
    ax.set_title('Validation Score')
    ax.legend()
    
    # 3. KL 散度对比
    ax = axes[1, 0]
    for name, metrics in results.items():
        if 'actor/approxkl' in metrics:
            steps, values = zip(*metrics['actor/approxkl'])
            ax.plot(steps, values, label=name)
    ax.set_xlabel('Steps')
    ax.set_ylabel('KL')
    ax.set_title('KL Divergence')
    ax.legend()
    
    # 4. 吞吐量对比
    ax = axes[1, 1]
    for name, metrics in results.items():
        if 'system/tps' in metrics:
            steps, values = zip(*metrics['system/tps'])
            ax.plot(steps, values, label=name)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Tokens/s')
    ax.set_title('Throughput')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'))
    plt.show()

if __name__ == '__main__':
    results = {
        'GRPO_baseline': load_tensorboard_logs('output/tensorboard/rlvr_grpo_baseline'),
        'GRPO_more_samples': load_tensorboard_logs('output/tensorboard/rlvr_grpo_more_samples'),
        'Reinforce': load_tensorboard_logs('output/tensorboard/rlvr_reinforce'),
    }
    
    plot_comparison(results, 'output/analysis')
```

### 6.5 学习目标

```
✅ 掌握控制变量法设计实验
✅ 理解不同算法的优缺点
✅ 学会分析和对比实验结果
✅ 理解超参对训练的影响
✅ 培养实验分析能力
```

---

## 7. Phase 5: 工程优化与性能调优

### 7.1 显存优化策略

#### 7.1.1 Gradient Checkpointing

```yaml
actor_train:
  model_args:
    disable_gradient_checkpointing: false  # 启用
```

**效果**：减少约 30% 显存，增加约 20% 训练时间

#### 7.1.2 CPU Offload

```yaml
strategy_args:
  strategy_name: fsdp2_train
  strategy_config:
    offload_policy: true  # 启用 CPU offload
```

**效果**：减少约 50% 显存，增加约 50% 训练时间

#### 7.1.3 减小 Batch Size

```yaml
actor_train:
  training_args:
    per_device_train_batch_size: 1  # 最小 batch size
    gradient_accumulation_steps: 64  # 增加 accumulation 补偿
```

#### 7.1.4 混合精度训练

```yaml
actor_train:
  model_args:
    dtype: bf16  # 使用 BF16
```

### 7.2 训练速度优化

#### 7.2.1 vLLM 推理优化

```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.7  # 适当提高
      block_size: 16
      max_model_len: 2048
      # 启用连续批处理
      max_num_batched_tokens: 4096
```

#### 7.2.2 数据加载优化

```yaml
actor_train:
  data_args:
    preprocessing_num_workers: 4  # 多进程数据加载
```

### 7.3 多卡效率优化

#### 7.3.1 FSDP2 配置

```yaml
actor_train:
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 8  # 8卡 FSDP
      param_dtype: bf16
      reduce_dtype: float32
      reshard_after_forward: true
      offload_policy: false
```

#### 7.3.2 通信优化

```yaml
# 4090 没有 NVLink，通信效率较低
# 可以通过以下方式优化：
# 1. 减少通信频率（增大 gradient accumulation）
# 2. 使用梯度压缩
# 3. 异步通信
```

### 7.4 性能基准测试

```bash
#!/bin/bash
# examples/4090_scripts/benchmark.sh

set -e

cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"

echo "=== Performance Benchmark ==="

# 测试 1: 单卡训练速度
echo "Test 1: Single GPU training speed"
python examples/start_rlvr_pipeline.py \
  --config_path examples/4090_config \
  --config_name benchmark_single_gpu

# 测试 2: 8卡训练速度
echo "Test 2: 8 GPU training speed"
python examples/start_rlvr_pipeline.py \
  --config_path examples/4090_config \
  --config_name benchmark_8_gpu

# 测试 3: 显存使用
echo "Test 3: Memory usage"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo "Benchmark completed!"
```

### 7.5 学习目标

```
✅ 掌握显存优化技巧
✅ 理解训练速度优化方法
✅ 学会配置 FSDP2
✅ 理解多卡并行的效率问题
✅ 能够进行性能基准测试
```

---

## 8. 关键配置参数对照表

### 8.1 4090 vs A100 配置差异

| 参数 | 4090 推荐值 | A100 默认值 | 说明 |
|------|-------------|-------------|------|
| `rollout_batch_size` | 32-64 | 64-128 | 减小以节省显存 |
| `per_device_train_batch_size` | 1-2 | 1-4 | 减小以节省显存 |
| `gradient_accumulation_steps` | 32-64 | 16-32 | 增大以补偿 batch size |
| `num_return_sequences_in_group` | 4-8 | 8-16 | 减小以节省显存 |
| `gpu_memory_utilization` | 0.6-0.7 | 0.8 | 减小以避免 OOM |
| `max_new_tokens` | 64-128 | 128-512 | 减小以节省显存 |
| `response_length` | 512-1024 | 2048-4096 | 减小以节省显存 |
| `num_env_groups` | 8-16 | 32-128 | 减小以节省资源 |
| `group_size` | 4-8 | 8-16 | 减小以节省资源 |

### 8.2 模型大小配置建议

| 模型 | 4090 配置 | 显存占用 | 训练速度 |
|------|-----------|----------|----------|
| 0.5B | 8卡，batch=2，acc=32 | ~15GB/卡 | ~15s/step |
| 1.5B | 8卡，batch=1，acc=64 | ~20GB/卡 | ~30s/step |
| 3B | 8卡，CPU offload | ~22GB/卡 | ~60s/step |
| 7B | 8卡，CPU offload + FSDP | ~24GB/卡 | ~120s/step |

---

## 9. 常见问题与解决方案

### 9.1 显存不足 (OOM)

**问题**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
```yaml
# 方案 1: 减小 batch size
rollout_batch_size: 16
per_device_train_batch_size: 1

# 方案 2: 启用 gradient checkpointing
actor_train:
  model_args:
    disable_gradient_checkpointing: false

# 方案 3: 减小生成长度
response_length: 512
max_new_tokens: 64

# 方案 4: 使用 CPU offload
strategy_args:
  strategy_config:
    offload_policy: true
```

### 9.2 训练速度慢

**问题**：单步训练时间过长

**解决方案**：
```yaml
# 方案 1: 使用 Megatron 而不是 FSDP
strategy_args:
  strategy_name: megatron_train

# 方案 2: 优化 vLLM 配置
actor_infer:
  strategy_args:
    strategy_config:
      gpu_memory_utilization: 0.7
      max_num_batched_tokens: 4096

# 方案 3: 减少环境并行数
train_env_manager:
  num_env_groups: 8
```

### 9.3 多卡通信效率低

**问题**：8卡并行效率只有 60-70%

**解决方案**：
```yaml
# 方案 1: 使用 FSDP2 而不是 DDP
strategy_args:
  strategy_name: fsdp2_train
  strategy_config:
    fsdp_size: 8

# 方案 2: 增大 gradient accumulation
gradient_accumulation_steps: 64

# 方案 3: 使用梯度压缩
# （需要在代码中实现）
```

### 9.4 模型下载失败

**问题**：
```
ConnectionError: Failed to download model
```

**解决方案**：
```bash
# 方案 1: 使用国内镜像
export USE_MODELSCOPE=1

# 方案 2: 手动下载
# 从 ModelScope 或 HuggingFace 镜像下载模型
# 然后修改配置中的模型路径
```

### 9.5 Ray 初始化失败

**问题**：
```
Ray connection failed
```

**解决方案**：
```bash
# 方案 1: 检查端口
netstat -tlnp | grep 6379

# 方案 2: 重启 Ray
ray stop
ray start --head

# 方案 3: 检查防火墙
sudo ufw status
```

---

## 10. 学习成果检验清单

### 10.1 基础能力

```
□ 能够独立搭建 ROLL 开发环境
□ 理解 YAML 配置系统的结构
□ 能够运行单卡 Agentic Pipeline
□ 能够运行单卡 RLVR Pipeline
□ 理解各配置参数的含义
```

### 10.2 算法理解

```
□ 理解 GRPO 算法的原理和实现
□ 理解 PPO 算法的原理和实现
□ 理解 Reinforce 算法的原理和实现
□ 能够对比不同算法的优缺点
□ 理解 Advantage 估计的方法
```

### 10.3 工程能力

```
□ 能够配置 FSDP2 训练
□ 能够配置 Megatron 训练
□ 能够配置 vLLM 推理
□ 掌握显存优化技巧
□ 能够进行性能基准测试
```

### 10.4 实验能力

```
□ 能够设计对比实验
□ 能够分析训练曲线
□ 能够调优超参数
□ 能够撰写实验报告
□ 能够复现论文结果
```

### 10.5 面试准备

```
□ 能够清晰介绍 ROLL 框架
□ 能够解释 RLHF/GRPO/PPO 的原理
□ 能够描述项目中遇到的问题和解决方案
□ 能够讨论工程优化的方法
□ 能够分析业务场景的需求
```

---

## 附录 A: 快速启动脚本

### A.1 环境初始化脚本

```bash
#!/bin/bash
# scripts/setup_4090_env.sh

set -e

echo "=== Setting up ROLL environment for 8x4090 ==="

# 1. 检查 GPU
echo "Checking GPUs..."
nvidia-smi

# 2. 克隆 ROLL
if [ ! -d "/workspace/ROLL" ]; then
    echo "Cloning ROLL..."
    cd /workspace
    git clone https://github.com/alibaba/ROLL.git
fi

cd /workspace/ROLL

# 3. 安装依赖
echo "Installing dependencies..."
pip install -r requirements_torch260_vllm.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -e .

# 4. 设置环境变量
echo "Setting environment variables..."
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"
export USE_MODELSCOPE=1

# 5. 验证安装
echo "Verifying installation..."
python -c "import roll; print('ROLL installed successfully!')"

echo "=== Setup completed! ==="
```

### A.2 一键运行脚本

```bash
#!/bin/bash
# scripts/run_all_experiments.sh

set -e

cd /workspace/ROLL
export PYTHONPATH="/workspace/ROLL:$PYTHONPATH"
export USE_MODELSCOPE=1

echo "=== Running all experiments on 8x4090 ==="

# Phase 1: 单卡验证
echo "Phase 1: Single GPU verification..."
python examples/start_agentic_pipeline.py \
  --config_path examples/4090_config \
  --config_name single_gpu_test

# Phase 2: Agentic Pipeline
echo "Phase 2: Agentic Pipeline..."
python examples/start_agentic_pipeline.py \
  --config_path examples/4090_config \
  --config_name agentic_frozen_lake_8x4090

# Phase 3: RLVR Pipeline
echo "Phase 3: RLVR Pipeline..."
python examples/start_rlvr_pipeline.py \
  --config_path examples/4090_config \
  --config_name rlvr_math_8x4090

# Phase 4: 算法对比
echo "Phase 4: Algorithm comparison..."
bash examples/4090_scripts/run_algorithm_comparison.sh

echo "=== All experiments completed! ==="
```

---

## 附录 B: 学习资源

### B.1 官方文档

- ROLL GitHub: https://github.com/alibaba/ROLL
- ROLL 文档: https://alibaba.github.io/ROLL/
- ROLL 论文: https://arxiv.org/abs/2506.06122

### B.2 相关论文

- PPO: https://arxiv.org/abs/1707.06347
- GRPO: https://arxiv.org/abs/2402.03300
- DPO: https://arxiv.org/abs/2305.18290
- RLHF: https://arxiv.org/abs/2203.02155

### B.3 工具文档

- Ray: https://docs.ray.io/
- vLLM: https://docs.vllm.ai/
- FSDP2: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM

---

## 总结

### 8×4090 复现 ROLL 的关键点

```
1. 模型选择：使用 0.5B-1.5B 小模型
2. 配置优化：减小 batch size，增加 gradient accumulation
3. 显存管理：使用 gradient checkpointing，必要时 CPU offload
4. 训练策略：先跑通流程，再优化性能
5. 实验设计：控制变量，多维度评估
```

### 预期成果

```
Week 1: 跑通 Agentic Pipeline，理解核心流程
Week 2: 跑通 RLVR Pipeline，完成算法对比
Week 3: 性能优化，准备面试材料
```

> **提示**：4090 虽然算力有限，但足以理解 ROLL 的核心架构和算法原理。重点是理解"为什么这么设计"，而不是追求大规模训练的效果。
