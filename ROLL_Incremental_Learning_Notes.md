# ROLL 框架增量学习笔记

> 基于前两轮分析，补充学习到的新知识点和深入理解
> 
> 记录从"知道概念"到"理解本质"的增量认知

---

## 目录

1. [配置系统深度理解](#1-配置系统深度理解)
2. [FSDP2 vs Megatron 选型指南](#2-fsdp2-vs-megatron-选型指南)
3. [4090 实际限制与优化策略](#3-4090-实际限制与优化策略)
4. [Agentic Pipeline 工程细节](#4-agentic-pipeline-工程细节)
5. [RLVR Pipeline 工程细节](#5-rlvr-pipeline-工程细节)
6. [模型更新机制深入理解](#6-模型更新机制深入理解)
7. [性能优化实战经验](#7-性能优化实战经验)
8. [调试与问题定位技巧](#8-调试与问题定位技巧)
9. [业务场景适配思考](#9-业务场景适配思考)

---

## 1. 配置系统深度理解

### 1.1 Hydra 配置系统的层次结构

**前两轮认知**：知道 ROLL 使用 YAML 配置文件

**增量理解**：配置系统有明确的层次结构

```yaml
# 层次结构：
# 1. 基础配置（config/ 目录）
# 2. Pipeline 配置（具体 yaml 文件）
# 3. CLI 覆盖（命令行参数）

# 示例：配置继承
defaults:
  - ../config/traj_envs@_here_  # 继承基础环境配置

# CLI 覆盖
# python examples/start_agentic_pipeline.py rollout_batch_size=32
```

### 1.2 配置参数的映射关系

**前两轮认知**：知道配置参数的含义

**增量理解**：配置参数如何映射到代码

```python
# YAML 配置 → Python 对象
# roll/pipeline/agentic/agentic_config.py

@dataclass
class AgenticConfig:
    rollout_batch_size: int = 1024
    sequence_length: int = 8192
    adv_estimator: str = "grpo"
    # ...

# 使用 dacite 转换
from dacite import from_dict
ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))
```

### 1.3 关键配置参数的深层含义

**前两轮认知**：知道 `rollout_batch_size` 是每步的 prompt 数量

**增量理解**：它与训练 batch size 的关系

```python
# 实际训练 batch size 的计算
# Megatron 后端：
global_batch_size = (
    per_device_train_batch_size * 
    gradient_accumulation_steps * 
    world_size / 
    tensor_model_parallel_size / 
    pipeline_model_parallel_size / 
    context_parallel_size
)

# 关键约束：
# rollout_batch_size * num_return_sequences_in_group 必须是 global_batch_size 的整数倍
```

---

## 2. FSDP2 vs Megatron 选型指南

### 2.1 两种后端的核心差异

**前两轮认知**：知道 ROLL 支持 FSDP2 和 Megatron 两种训练后端

**增量理解**：选型的关键考量

```
FSDP2 特点：
- PyTorch 原生，兼容性好
- 配置简单，学习成本低
- 支持 CPU offload
- 适合中小规模训练（<16卡）

Megatron 特点：
- NVIDIA 优化，性能更好
- 支持 5D 并行（TP/PP/CP/EP/DP）
- 配置复杂，学习成本高
- 适合大规模训练（>16卡）

选型建议：
- 8×4090：优先 FSDP2（简单、够用）
- 16×A100：可以考虑 Megatron（性能更好）
- 32+GPU：必须 Megatron（需要模型并行）
```

### 2.2 FSDP2 的关键配置

**增量理解**：FSDP2 配置的深层含义

```yaml
strategy_args:
  strategy_name: fsdp2_train
  strategy_config:
    # fsdp_size 的含义
    # - fsdp_size >= world_size: 纯 FSDP 模式
    # - fsdp_size < world_size: HSDP 模式（有 DDP 副本）
    fsdp_size: 8
    
    # reshard_after_forward 的权衡
    # - true: 节省显存，但训练稍慢
    # - false: 显存占用多，但训练更快
    reshard_after_forward: true
    
    # offload_policy 的权衡
    # - true: 节省显存，但训练更慢
    # - false: 显存占用多，但训练更快
    offload_policy: false
```

### 2.3 Megatron 的关键配置

**增量理解**：Megatron 并行策略的选择

```yaml
strategy_args:
  strategy_name: megatron_train
  strategy_config:
    # TP (Tensor Parallelism)
    # - 适合：大模型、单机多卡
    # - 限制：需要高带宽通信（NVLink）
    tensor_model_parallel_size: 1  # 4090 不适合 TP
    
    # PP (Pipeline Parallelism)
    # - 适合：超大模型、多机
    # - 限制：有 pipeline bubble 开销
    pipeline_model_parallel_size: 1
    
    # CP (Context Parallelism)
    # - 适合：长序列训练
    # - 优势：可以处理超长序列
    context_parallel_size: 1
    
    # EP (Expert Parallelism)
    # - 适合：MoE 模型
    # - 限制：只对 MoE 有效
    expert_model_parallel_size: 1
```

---

## 3. 4090 实际限制与优化策略

### 3.1 显存限制的实际影响

**前两轮认知**：知道 4090 有 24GB 显存

**增量理解**：显存的具体分配

```
模型参数（0.5B）：
- FP16: ~1GB
- BF16: ~1GB

优化器状态（Adam）：
- 参数副本: ~2GB
- 梯度: ~1GB
- 动量: ~2GB
- 方差: ~2GB
总计: ~7GB

激活值（取决于 batch size 和序列长度）：
- batch=1, seq=2048: ~4GB
- batch=2, seq=2048: ~8GB
- batch=4, seq=2048: ~16GB

总显存需求：
- 最小配置: ~12GB
- 推荐配置: ~18GB
- 最大配置: ~24GB
```

### 3.2 4090 的优化策略

**增量理解**：针对 4090 的具体优化

```yaml
# 策略 1: 使用 gradient checkpointing
actor_train:
  model_args:
    disable_gradient_checkpointing: false
# 效果：减少 ~30% 激活值显存，增加 ~20% 训练时间

# 策略 2: 使用 CPU offload（FSDP2）
strategy_args:
  strategy_config:
    offload_policy: true
# 效果：减少 ~50% 显存，增加 ~50% 训练时间

# 策略 3: 使用 distributed optimizer（Megatron）
strategy_args:
  strategy_config:
    use_distributed_optimizer: true
# 效果：将优化器状态分片到多卡，减少 ~60% 优化器显存

# 策略 4: 减小 batch size，增加 gradient accumulation
actor_train:
  training_args:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 64
# 效果：减少显存占用，保持有效 batch size
```

### 3.3 4090 的性能瓶颈

**增量理解**：4090 的主要性能瓶颈

```
瓶颈 1: 显存带宽
- 4090: 1008 GB/s
- A100: 2039 GB/s
- 影响：推理速度慢 ~2 倍

瓶颈 2: 卡间通信
- 4090: PCIe 4.0 (~32 GB/s)
- A100: NVLink (~600 GB/s)
- 影响：多卡并行效率低

瓶颈 3: 计算能力
- 4090: 82.6 TFLOPS (FP16)
- A100: 312 TFLOPS (FP16)
- 影响：训练速度慢 ~3-4 倍
```

---

## 4. Agentic Pipeline 工程细节

### 4.1 环境管理器的工作机制

**前两轮认知**：知道 EnvManager 负责环境交互

**增量理解**：EnvManager 的具体工作流程

```python
# EnvManager 的生命周期
class BaseEnvManager:
    def run_rollout_loop(self, data: DataProto):
        """
        1. 从 scheduler 获取 episode_id
        2. 调用 reset() 初始化环境
        3. 循环执行：
           - make_decision() 获取模型输出
           - step() 执行动作
           - 检查终止条件
        4. 调用 formulate_rollouts() 构建训练数据
        5. 通过 output_queue 提交轨迹
        """
        while self.running:
            # 获取 episode
            episode_id = self.scheduler.get_episode_id(self.group_id)
            if episode_id is None:
                break
            
            # 初始化环境
            rollout_cache = self.reset()
            
            # 多轮交互
            while not done:
                lm_output = self.make_decision(rollout_cache)
                rollout_cache = self.step(lm_output)
            
            # 提交轨迹
            rollout = self.formulate_rollouts(rollout_cache)
            self.output_queue.put(self.group_id, episode_id, rollout)
```

### 4.2 环境组的配置策略

**增量理解**：环境组配置的深层含义

```yaml
train_env_manager:
  # max_env_num_per_worker: 每个 worker 的最大环境数
  # 影响：内存占用、并行度
  max_env_num_per_worker: 16
  
  # num_env_groups: 环境组总数
  # 影响：训练数据的多样性
  num_env_groups: 128
  
  # group_size: 每组的环境数
  # 影响：GRPO 的采样数
  group_size: 8
  
  # 配置建议：
  # - 小显存：减小 max_env_num_per_worker
  # - 快速验证：减小 num_env_groups
  # - GRPO 训练：group_size >= 4
```

### 4.3 轨迹收集与过滤

**增量理解**：轨迹过滤的实现细节

```python
# roll/pipeline/agentic/agentic_pipeline.py
class GroupFilter:
    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        """
        过滤逻辑：
        1. 检查 drop_flag
        2. 检查轨迹长度
        3. 检查奖励值
        """
        for data in group:
            if data.meta_info["drop_flag"]:
                return True  # 过滤该轨迹
        return False
```

---

## 5. RLVR Pipeline 工程细节

### 5.1 多域奖励系统的实现

**前两轮认知**：知道 ROLL 支持多域奖励

**增量理解**：多域奖励的具体实现

```yaml
# 域配置示例
rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    tag_included: [deepmath_103k, aime]
    world_size: 8
    
  code_sandbox:
    worker_cls: roll.pipeline.rlvr.rewards.code_sandbox_reward_worker.CodeSandboxRewardWorker
    tag_included: [KodCode]
    world_size: 8

# 域采样概率
data_args:
  domain_interleave_probs:
    math_rule: 0.4
    code_sandbox: 0.3
    llm_judge: 0.1
```

### 5.2 奖励计算的流程

**增量理解**：奖励计算的具体流程

```python
# 奖励计算流程
def compute_rewards(data: DataProto):
    # 1. 解析响应文本
    response_text = tokenizer.batch_decode(data.batch["responses"])
    
    # 2. 获取 ground truth
    ground_truths = data.non_tensor_batch["ground_truth"]
    
    # 3. 计算奖励
    rewards = []
    for pred, gt in zip(response_text, ground_truths):
        if verify_math_answer(pred, gt):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    # 4. 奖励后处理
    rewards = reward_postprocess(rewards, config)
    
    return rewards
```

### 5.3 奖励归一化策略

**增量理解**：不同归一化方法的适用场景

```yaml
reward_normalization:
  # grouping: 归一化的分组方式
  # - tags: 按环境类型分组
  # - traj_group_id: 按轨迹组分组
  # - batch: 按 batch 分组
  grouping: traj_group_id
  
  # method: 归一化方法
  # - identity: 不归一化
  # - mean_std: 均值-标准差归一化
  # - asym_clip: 非对称裁剪
  method: mean_std
```

---

## 6. 模型更新机制深入理解

### 6.1 训练-推理参数同步

**前两轮认知**：知道需要同步训练和推理模型的参数

**增量理解**：同步的具体实现

```python
# roll/distributed/executor/model_update_group.py
class ModelUpdateGroup:
    def model_update(self, step=None):
        """
        同步流程：
        1. 训练集群调用 start_model_update()
        2. 收集所有训练 worker 的参数
        3. 广播到推理集群
        4. 推理集群调用 finish_model_update()
        """
        # 1. 收集训练参数
        dataprotos = ray.get([
            train_worker.start_model_update.remote()
            for train_worker in self.src_cluster.workers
        ])
        
        # 2. 广播到推理集群
        ray.get([
            infer_worker.finish_model_update.remote(dataproto)
            for infer_worker, dataproto in zip(
                self.dst_cluster.workers, dataprotos
            )
        ])
```

### 6.2 Colocate vs Separate 模式

**增量理解**：两种部署模式的权衡

```yaml
# Colocate 模式：训练和推理共享 GPU
actor_train:
  device_mapping: list(range(0,8))
actor_infer:
  device_mapping: list(range(0,8))  # 相同

# Separate 模式：训练和推理使用不同 GPU
actor_train:
  device_mapping: list(range(0,4))
actor_infer:
  device_mapping: list(range(4,8))  # 不同

# 权衡：
# Colocate：
# - 优点：节省 GPU
# - 缺点：需要时间分复用，训练速度慢
#
# Separate：
# - 优点：训练和推理并行，速度快
# - 缺点：需要更多 GPU
```

---

## 7. 性能优化实战经验

### 7.1 吞吐量优化

**增量理解**：影响吞吐量的关键因素

```
因素 1: 推理效率
- 使用 vLLM/SGLang 而不是 HuggingFace
- 适当增大 gpu_memory_utilization
- 使用连续批处理

因素 2: 训练效率
- 使用 gradient checkpointing
- 使用混合精度训练
- 优化数据加载

因素 3: 通信效率
- 使用 FSDP2 而不是 DDP
- 增大 gradient accumulation
- 使用异步通信
```

### 7.2 显存优化

**增量理解**：显存优化的具体方法

```yaml
# 方法 1: 减小模型显存
actor_train:
  model_args:
    dtype: bf16  # 使用 BF16 而不是 FP32

# 方法 2: 减小优化器显存
strategy_args:
  strategy_config:
    use_distributed_optimizer: true  # 分片优化器状态

# 方法 3: 减小激活值显存
actor_train:
  model_args:
    disable_gradient_checkpointing: false

# 方法 4: CPU offload
strategy_args:
  strategy_config:
    offload_policy: true
```

### 7.3 训练稳定性优化

**增量理解**：保证训练稳定性的方法

```yaml
# 方法 1: 梯度裁剪
max_grad_norm: 1.0

# 方法 2: KL 约束
init_kl_coef: 0.1
use_kl_loss: true
kl_loss_coef: 0.001

# 方法 3: 奖励裁剪
reward_clip: 10

# 方法 4: 优势归一化
whiten_advantages: true
```

---

## 8. 调试与问题定位技巧

### 8.1 调试工具

**增量理解**：ROLL 提供的调试工具

```bash
# 1. Ray Debugger
export RAY_DEBUG=legacy
# 然后可以使用 pdb 调试

# 2. TensorBoard
track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard

# 3. 性能分析
system_envs:
  RAY_PROFILING: "1"
profiler_output_dir: ./output/profile
```

### 8.2 常见问题定位

**增量理解**：问题定位的系统方法

```
问题 1: OOM
定位：检查显存使用
nvidia-smi --query-gpu=memory.used --format=csv

解决：
1. 减小 batch size
2. 启用 gradient checkpointing
3. 使用 CPU offload

问题 2: 训练速度慢
定位：检查时间分布
查看 time/* 指标

解决：
1. 优化推理效率
2. 减少通信开销
3. 使用更快的后端

问题 3: 训练不稳定
定位：检查 loss 和 KL 曲线
查看 actor/* 指标

解决：
1. 减小学习率
2. 增加 KL 约束
3. 使用梯度裁剪
```

---

## 9. 业务场景适配思考

### 9.1 不同场景的配置策略

**增量理解**：根据场景调整配置

```
场景 1: 快速验证算法
- 减小模型大小（0.5B）
- 减小 batch size
- 减少训练步数
- 目标：跑通流程，验证算法

场景 2: 追求最佳效果
- 使用较大模型（1.5B+）
- 增大 batch size
- 增加训练步数
- 目标：达到最佳性能

场景 3: 资源受限
- 使用最小模型（0.5B）
- 使用 CPU offload
- 使用 gradient checkpointing
- 目标：在有限资源下完成训练
```

### 9.2 从实验到生产的考量

**增量理解**：实验与生产的差异

```
实验环境：
- 关注：算法效果
- 数据：小规模、干净
- 评估：标准 benchmark

生产环境：
- 关注：业务价值
- 数据：大规模、有噪声
- 评估：用户满意度、转化率

关键差异：
1. 数据质量：生产数据需要更多清洗
2. 系统稳定性：生产需要容错和监控
3. 延迟要求：生产有实时性要求
4. 成本控制：生产需要考虑资源成本
```

---

## 10. 总结：从知道到理解

### 10.1 认知升级

| 维度 | 前两轮认知 | 增量理解 |
|------|------------|----------|
| **配置系统** | 知道 YAML 配置 | 理解层次结构和参数映射 |
| **训练后端** | 知道 FSDP2 和 Megatron | 理解选型依据和配置细节 |
| **4090 限制** | 知道显存有限 | 理解具体限制和优化策略 |
| **Agentic RL** | 知道环境交互 | 理解 EnvManager 工作机制 |
| **RLVR** | 知道多域奖励 | 理解奖励计算和归一化 |
| **模型同步** | 知道需要同步 | 理解同步机制和部署模式 |
| **性能优化** | 知道需要优化 | 理解具体优化方法和权衡 |
| **调试技巧** | 知道有调试工具 | 理解问题定位的系统方法 |
| **业务适配** | 知道需要适配 | 理解不同场景的配置策略 |

### 10.2 关键收获

```
1. 配置系统：不是简单的参数设置，而是有层次、有约束的系统
2. 训练后端：不是简单的性能对比，而是有场景、有权衡的选择
3. 4090 限制：不是简单的"显存不够"，而是需要系统性的优化
4. Agentic RL：不是简单的环境交互，而是有复杂的状态管理
5. RLVR：不是简单的奖励计算，而是有多域、有归一化的系统
6. 模型同步：不是简单的参数复制，而是有模式、有效率的权衡
7. 性能优化：不是简单的"加速"，而是需要平衡多个维度
8. 调试技巧：不是简单的"看日志"，而是有系统、有方法的过程
9. 业务适配：不是简单的"调参"，而是需要理解场景和需求
```

### 10.3 面试准备建议

```
基础问题：能够清晰解释概念
- 什么是 GRPO？
- PPO 和 GRPO 的区别？

深入问题：能够解释原理和设计
- 为什么 GRPO 不需要 Critic？
- 这样设计有什么优缺点？

实战问题：能够描述问题和解决
- 你在 4090 上遇到了什么问题？
- 你是怎么解决的？

开放问题：能够分析场景和权衡
- 这个方案适合什么场景？
- 如果资源有限，怎么优化？
```

---

## 附录：学习资源推荐

### 官方文档

- ROLL GitHub: https://github.com/alibaba/ROLL
- ROLL 文档: https://alibaba.github.io/ROLL/
- ROLL 论文: https://arxiv.org/abs/2506.06122

### 相关论文

- PPO: https://arxiv.org/abs/1707.06347
- GRPO: https://arxiv.org/abs/2402.03300
- DPO: https://arxiv.org/abs/2305.18290
- FSDP2: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

### 工具文档

- Ray: https://docs.ray.io/
- vLLM: https://docs.vllm.ai/
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- TensorBoard: https://www.tensorflow.org/tensorboard
