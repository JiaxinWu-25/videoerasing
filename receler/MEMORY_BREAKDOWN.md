# 训练显存需求详细分解

## 问题 1: 为什么 Adapter 只有 8.4M 参数，但训练显存需要 192.46 GB？

### 关键误解

**Adapter 的 8.4M 参数只是训练显存的一小部分！**

训练显存的主要组成部分：

### 1. 两个完整模型（最大部分）

训练时需要加载**两个完整的 CogVideoX-5b 模型**：

```
Original Model (冻结):     ~10 GB (FP16)
Unlearned Model (冻结参数): ~10 GB (FP16)
─────────────────────────────────────────
小计:                        ~20 GB
```

**为什么需要两个模型？**
- **Original Model**: 用于计算 Concept Preservation Loss（保留概念的损失）
- **Unlearned Model**: 用于计算 Unlearning Loss（去学习损失）

**为什么 Unlearned Model 也需要完整模型？**
- ✅ **前向传播需要**：Adapter 不是独立模型，需要完整 Transformer 计算注意力
- ✅ **损失计算需要**：需要 unlearned 模型的完整输出
- ✅ **激活值需要**：反向传播需要存储完整模型的激活值
- ⚠️ **虽然只更新 Adapter 参数，但前向传播需要完整模型**

**训练后保存什么？**
- ✅ **只保存 Adapter 权重**（~12.3M 参数，类似 LoRA）
- ❌ **不保存完整模型**（可以从原始模型加载）

详见 `TRAINING_VS_INFERENCE.md` 了解训练和推理的区别。

### 2. 梯度显存

虽然 Original Model 冻结（不需要梯度），但 Unlearned Model 需要梯度：

```
Unlearned Model 梯度:      ~10 GB (FP16)
Adapter 梯度:               ~0.05 GB (FP16, 很小)
─────────────────────────────────────────
小计:                        ~10 GB
```

### 3. 优化器状态（AdamW）

优化器需要为每个可训练参数存储 momentum 和 variance：

```
Unlearned Model 优化器:    ~20 GB (FP16, 2倍参数)
Adapter 优化器:             ~0.1 GB (FP16, 很小)
─────────────────────────────────────────
小计:                        ~20 GB
```

### 4. 激活值（Activations）

前向传播和反向传播需要存储中间激活值：

```
激活值（估算）:             ~1-2 GB (取决于 batch_size)
```

### 5. Batch 数据

```
Batch Latents:              ~0.1-0.5 GB (取决于 batch_size)
Text Encoder 输出:          ~0.001 GB
```

### 6. Adapter 参数（很小）

```
Adapter 参数:               ~0.05 GB (FP16)
```

### 显存需求汇总

| 组件 | 显存 (GB) | 占比 |
|------|----------|------|
| Original Model | 10.0 | ~5% |
| Unlearned Model | 10.0 | ~5% |
| Unlearned Model 梯度 | 10.0 | ~5% |
| Unlearned Model 优化器 | 20.0 | ~10% |
| **激活值** | **~100-150** | **~50-75%** |
| Batch 数据 | 0.1-0.5 | <1% |
| Adapter 参数 | 0.05 | <0.1% |
| **总计** | **~150-200 GB** | **100%** |

### 关键发现

**激活值占用了大部分显存！**

激活值的大小取决于：
- Batch Size
- 序列长度（num_frames × latent_height × latent_width）
- 模型深度（41 层）
- 隐藏维度（1152）

对于 CogVideoX-5b：
- 41 层 Transformer Blocks
- Hidden dim = 1152
- Batch size = 4, num_frames = 24, latent = 32×32

激活值估算：
```
激活值 ≈ batch_size × num_frames × latent_size × hidden_dim × num_layers × multiplier
       ≈ 4 × 24 × (32×32) × 1152 × 41 × 2-3
       ≈ 100-150 GB
```

### 为什么 Adapter 显存很小？

Adapter 只有 8.4M 参数，显存占用：
- 参数: ~0.05 GB
- 梯度: ~0.05 GB  
- 优化器: ~0.1 GB
- **总计: ~0.2 GB**（相对于 192 GB 可以忽略）

## 问题 2: 为什么有 41 个 Transformer Blocks？

### CogVideoX-5b 架构

CogVideoX-5b 实际有 **41 个 Transformer Blocks**，而不是我之前说的 28 层。

### 验证方法

训练时会打印：
```
changing:  transformer.blocks.0
changing:  transformer.blocks.1
...
changing:  transformer.blocks.40
```

这说明有 41 个 blocks（索引 0-40）。

### Adapter 参数量重新计算

假设：
- `dim` = 1152（CogVideoX hidden dimension）
- `rank` = 128
- `num_blocks` = 41

每个 AdapterEraser 的参数量：
```
参数量 = 2 × dim × rank + rank + dim
       = 2 × 1152 × 128 + 128 + 1152
       ≈ 295,424 参数
       ≈ 0.3M 参数
```

**总 Adapter 参数量**：
```
总参数量 = 41 × 0.3M
         = 12.3M 参数
```

### 显存占用（41 层）

| Rank | 每个 Adapter | 41 层总参数量 | FP16 训练显存 |
|------|-------------|--------------|--------------|
| 64   | ~0.15M      | ~6.2M        | ~37 MB       |
| 128  | ~0.30M      | ~12.3M       | ~74 MB       |
| 256  | ~0.59M      | ~24.2M       | ~145 MB      |

**注意**：即使有 41 层，Adapter 的显存占用仍然很小（~74 MB），相对于模型本身的显存（~20 GB）可以忽略。

## 显存优化策略

### 1. 使用 CPU Offload（最重要）

```bash
--use_cpu_offload
```

**效果**：
- 模型参数动态加载到 GPU
- 显存需求从 ~150-200 GB 降至 ~15-25 GB
- **节省 ~85-90% 显存**

### 2. 使用 FP16

```bash
--use_fp16
```

**效果**：
- 参数显存减半：20 GB → 10 GB
- 激活值也减半：~100 GB → ~50 GB
- **节省 ~50% 显存**

### 3. 减小 Batch Size

```bash
--batch_size 1  # 从 4 降到 1
```

**效果**：
- 激活值线性减少：~100 GB → ~25 GB
- **节省 ~75% 激活值显存**

### 4. 降低 Rank（效果有限）

```bash
--eraser_rank 64  # 从 128 降到 64
```

**效果**：
- Adapter 显存：74 MB → 37 MB
- **节省 ~37 MB**（相对于 192 GB 可以忽略）

## 实际显存需求对比

### 不使用 CPU Offload

| 配置 | 显存需求 |
|------|---------|
| FP32, batch_size=4 | ~300-400 GB |
| FP16, batch_size=4 | ~150-200 GB |
| FP16, batch_size=2 | ~75-100 GB |
| FP16, batch_size=1 | ~40-50 GB |

### 使用 CPU Offload

| 配置 | 显存需求 |
|------|---------|
| FP16, batch_size=4 | ~20-30 GB |
| FP16, batch_size=2 | ~15-20 GB |
| FP16, batch_size=1 | ~10-15 GB |

## 总结

### 为什么 Adapter 只有 8.4M（实际 12.3M）但需要 192 GB？

1. ✅ **两个完整模型**：~20 GB
2. ✅ **梯度 + 优化器**：~30 GB
3. ✅ **激活值**：~100-150 GB（最大部分！）
4. ✅ **Adapter**：~0.1 GB（可以忽略）

**激活值是显存的主要消耗者**，而不是模型参数本身。

### 为什么有 41 个 Blocks？

- CogVideoX-5b 实际有 **41 个 Transformer Blocks**
- 每个 Block 的 attn1 后都注入一个 AdapterEraser
- 总 Adapter 参数量：**12.3M**（rank=128）

### 显存优化优先级

1. 🥇 **CPU Offload**（节省最多）
2. 🥈 **FP16**（节省 50%）
3. 🥉 **减小 Batch Size**（线性减少）
4. ⚠️ **降低 Rank**（效果有限，不推荐）

