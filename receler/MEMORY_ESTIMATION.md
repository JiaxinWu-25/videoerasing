# 显存需求估算指南

## 概述

在训练大型模型（如 CogVideoX-5b）时，提前估算显存需求可以避免 OOM（Out of Memory）错误。本工具提供了自动显存估算功能。

## 快速使用

### 1. 独立估算工具

使用 `estimate_memory.py` 脚本进行显存估算：

```bash
# 基本估算（使用默认参数）
python receler/estimate_memory.py --model_path ./CogVideoX-5b

# 指定 batch_size 和配置
python receler/estimate_memory.py \
    --model_path ./CogVideoX-5b \
    --batch_size 4 \
    --num_frames 17 \
    --dtype float16

# 根据可用显存推荐配置
python receler/estimate_memory.py \
    --model_path ./CogVideoX-5b \
    --recommend \
    --device_id 2
```

### 2. 训练脚本自动检查

训练脚本 `train_unlearning_prompt_only.py` 会在训练前自动检查显存：

```bash
CUDA_VISIBLE_DEVICES=2 python receler/train_unlearning_prompt_only.py \
    --model_path ./CogVideoX-5b \
    --target_concept "nudity" \
    --batch_size 4 \
    --use_fp16 \
    --use_cpu_offload  # 如果显存不足，会自动建议添加此参数
```

## 显存组成

训练时的显存主要由以下部分组成：

### 1. 模型参数（Model Parameters）

- **Transformer**: ~5B 参数
  - FP32: ~20 GB
  - FP16: ~10 GB
- **Text Encoder (T5)**: ~220M 参数
  - FP32: ~0.88 GB
  - FP16: ~0.44 GB

### 2. 梯度（Gradients）

- 大小 = 模型参数大小
- FP32: ~20 GB
- FP16: ~10 GB

### 3. 优化器状态（Optimizer States）

- **AdamW**: 需要 momentum + variance = 2倍参数大小
- FP32: ~40 GB
- FP16: ~20 GB

### 4. 激活值（Activations）

- 取决于 batch_size、序列长度、模型深度
- 粗略估算：batch_size × num_frames × latent_size × 3-5倍

### 5. Batch 数据

- Latent: `batch_size × channels × height × width × num_frames`
- 例如：`4 × 16 × 32 × 32 × 17` = ~0.14 GB (FP16)

### 6. 两个模型实例

训练时需要：
- **Original Model**（冻结，不需要梯度）: ~10 GB (FP16)
- **Unlearned Model**（需要梯度）: ~10 GB (FP16) + 梯度 + 优化器

## 显存需求计算公式

### 单模型推理
```
显存 = 模型参数 × 数据类型大小
```

### 训练（单模型）
```
显存 = 模型参数 × (1 + 1 + 2) × 数据类型大小
     = 模型参数 × 4 × 数据类型大小
```
其中：
- 1 = 模型参数
- 1 = 梯度
- 2 = 优化器状态（AdamW）

### 训练（双模型：original + unlearned）
```
显存 = 模型参数 × (1 + 1 + 1 + 2) × 数据类型大小
     = 模型参数 × 5 × 数据类型大小
```
其中：
- 1 = original 模型（冻结）
- 1 = unlearned 模型参数
- 1 = unlearned 模型梯度
- 2 = unlearned 模型优化器状态

### 加上激活值和 Batch 数据
```
总显存 = 模型显存 + 激活值 + Batch 数据
```

## 实际示例

### CogVideoX-5b (FP16)

假设：
- Transformer: 5B 参数 × 2 bytes = 10 GB
- Text Encoder: 220M 参数 × 2 bytes = 0.44 GB
- Batch Size: 4
- Num Frames: 17

**模型基础显存**:
- Original Model: 10.44 GB
- Unlearned Model: 10.44 GB
- Gradients: 10.44 GB
- Optimizer States: 20.88 GB
- **小计**: 52.2 GB

**训练时额外显存**:
- Batch Latents: ~0.14 GB
- Text Encoder Outputs: ~0.01 GB
- Activations: ~1-2 GB
- **小计**: ~2 GB

**总显存需求**: ~54 GB

### 使用 CPU Offload

如果使用 `--use_cpu_offload`，模型参数会动态加载到 GPU，显存需求大幅降低：

- 仅当前使用的层在 GPU 上
- 估算显存需求: ~15-20 GB（取决于 batch_size）

## 显存优化建议

### 1. 使用 FP16

```bash
--use_fp16
```
- 显存节省: ~50%
- 注意：需要 GPU 支持（计算能力 >= 7.0）

### 2. 使用 CPU Offload

```bash
--use_cpu_offload
```
- 显存节省: ~60-70%
- 速度: 可能稍慢（CPU-GPU 数据传输）

### 3. 减小 Batch Size

```bash
--batch_size 1  # 默认 4
```
- 显存节省: 线性减少
- 注意：可能影响训练稳定性

### 4. 使用梯度累积

如果 batch_size=1 仍然 OOM，可以使用梯度累积：
- 多次前向传播，累积梯度
- 最后统一更新参数
- 等效于更大的 batch_size，但显存需求不变

### 5. 使用多 GPU

```bash
--multi_gpu
```
- 模型并行，GPU上
- 需要多个 GPU（通过 `CUDA_VISIBLE_DEVICES` 指定）

## 常见问题

### Q: 为什么估算的显存和实际使用不一致？

A: 估算值基于理论计算，实际使用可能因以下因素有所不同：
- PyTorch 内存碎片
- 框架开销
- 激活值大小（取决于实际序列长度）
- 其他进程占用

建议保留 10-20% 的显存缓冲。

### Q: 如何查看实际显存使用？

A: 使用 `nvidia-smi` 或 PyTorch 内置函数：

```python
import torch
print(f"已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"已保留: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

### Q: 单 GPU 训练 CogVideoX-5b 需要多少显存？

A: 
- **不使用 CPU Offload**: ~54 GB（FP16）或 ~108 GB（FP32）
- **使用 CPU Offload**: ~15-20 GB（FP16）

### Q: 为什么训练时需要两个模型？

A: 
- **Original Model**: 用于计算保留概念的损失（Concept Preservation Loss）
- **Unlearned Model**: 用于计算去学习损失（Unlearning Loss）

两个模型都需要在内存中，但 original 模型是冻结的（不需要梯度）。

## 工具输出示例

```
================================================================
显存检查
================================================================

GPU 0 显存状态:
  总显存: 46.07 GB
  已分配: 0.00 GB
  已保留: 0.00 GB
  可用显存: 46.07 GB

估算训练显存需求...
正在加载模型以估算显存需求...
  模型路径: ./CogVideoX-5b
  数据类型: torch.float16 (2 字节/参数)
  Transformer: 5.00B 参数 = 10.00 GB
  Text Encoder: 0.22B 参数 = 0.44 GB

梯度显存: 10.44 GB
优化器状态 (AdamW): 20.88 GB

Batch Latents (4x17): 0.14 GB
Text Encoder 输出: 0.0001 GB
激活值（估算）: 1.50 GB

================================================================
显存需求汇总
================================================================
  model_parameters          :    10.44 GB
  gradients                 :    10.44 GB
  optimizer_states          :    20.88 GB
  batch_latents             :     0.14 GB
  text_encoder_outputs      :     0.00 GB
  activations               :     1.50 GB
  original_model            :    10.44 GB
  unlearned_model           :    10.44 GB
------------------------------------------------------------
  总显存需求: 63.78 GB
================================================================

显存对比:
  估算需求: 63.78 GB
  可用显存: 46.07 GB

⚠️  警告：估算显存需求 (63.78 GB) 接近或超过可用显存 (46.07 GB)
   建议:
   1. 添加 --use_cpu_offload 参数以使用 CPU offload
   2. 减小 --batch_size（当前: 4）
   3. 使用 --use_fp16 以节省显存（如果尚未启用）

   自动启用 CPU offload 以避免 OOM...
================================================================
```

## 参考

- [PyTorch 显存管理文档](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Diffusers CPU Offload 文档](https://huggingface.co/docs/diffusers/optimization/fp16#memory-efficient-attention)

