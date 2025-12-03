# 训练 vs 推理：为什么训练时需要完整模型？

## 核心问题

**Q: 训练后只保存 Adapter（类似 LoRA），为什么训练时需要加载完整的 unlearned model？**

## 关键理解

### Adapter 不是独立的模型，而是注入到完整模型中的

```
完整模型 = 原始 Transformer + Adapter（注入）
```

训练时需要完整模型进行**前向传播**，虽然只更新 Adapter 参数。

## 训练时的流程

### 1. 模型加载

```python
# 加载两个完整的模型
pipe_original = CogVideoXPipeline.from_pretrained(...)  # 完整模型
pipe_unlearned = CogVideoXPipeline.from_pretrained(...)  # 完整模型

# 在 unlearned 模型中注入 Adapter
erasers = setup_cogvideo_adapter_eraser(
    model=pipe_unlearned.transformer,  # 完整模型
    eraser_rank=128
)
```

### 2. 前向传播（需要完整模型）

```python
# 计算 unlearned 模型的输出
def model_wrapper_unlearned(x_t, t, cond):
    output = transformer_unlearned(  # 完整模型 + Adapter
        hidden_states,
        timestep=t,
        encoder_hidden_states=cond
    )
    return output

# 计算 original 模型的输出（用于对比）
def model_wrapper_original(x_t, t, cond):
    with torch.no_grad():
        output = transformer_original(  # 完整模型（冻结）
            hidden_states,
            timestep=t,
            encoder_hidden_states=cond
        )
    return output
```

### 3. 前向传播过程

当调用 `transformer_unlearned()` 时：

```
输入: hidden_states
  ↓
Transformer Block 0:
  ├─ attn1 (原始注意力)
  ├─ Adapter (注入) ← 在这里修改输出
  └─ 输出: hidden_states + adapter(hidden_states)
  ↓
Transformer Block 1:
  ├─ attn1 (原始注意力)
  ├─ Adapter (注入) ← 在这里修改输出
  └─ 输出: hidden_states + adapter(hidden_states)
  ↓
...
Transformer Block 40:
  ├─ attn1 (原始注意力)
  ├─ Adapter (注入) ← 在这里修改输出
  └─ 输出: hidden_states + adapter(hidden_states)
  ↓
最终输出
```

**关键点**：
- 每一层都需要原始 Transformer 计算注意力
- Adapter 只是修改输出（残差连接）
- **没有完整模型，无法进行前向传播**

### 4. 损失计算

```python
# 需要两个模型的输出进行对比
loss = loss_fn(
    model_original=model_wrapper_original,    # 原始模型输出
    model_unlearned=model_wrapper_unlearned,  # 去学习模型输出
    ...
)
```

### 5. 反向传播（只更新 Adapter）

```python
# 冻结原始模型参数
for param in pipe_unlearned.transformer.parameters():
    param.requires_grad = False  # 不更新

# 只优化 Adapter 参数
for eraser in erasers.values():
    for param in eraser.parameters():
        param.requires_grad = True  # 只更新这个

# 反向传播
loss.backward()  # 只计算 Adapter 的梯度
optimizer.step()  # 只更新 Adapter 参数
```

### 6. 保存（只保存 Adapter）

```python
# 只保存 Adapter 权重，不保存完整模型
save_cogvideo_eraser_from_transformer(
    output_dir,
    pipe_unlearned.transformer
)
# 保存的文件：
# - eraser_weights.pt (只有 Adapter 权重，~12.3M 参数)
# - eraser_config.json (配置信息)
```

## 推理时的流程

### 1. 加载原始模型

```python
pipe = CogVideoXPipeline.from_pretrained(
    model_path,  # 原始模型路径
    torch_dtype=torch.float16
)
```

### 2. 注入训练好的 Adapter

```python
# 加载 Adapter 权重
eraser_ckpt = torch.load("eraser_weights.pt")
eraser_config = json.load("eraser_config.json")

# 注入到原始模型中
inject_eraser(
    transformer=pipe.transformer,
    eraser_ckpt=eraser_ckpt,
    eraser_rank=eraser_config['eraser_rank']
)
```

### 3. 推理

```python
# 现在 pipe.transformer 已经包含 Adapter
video = pipe(prompt="...")
```

## 为什么训练时需要完整模型？

### 原因 1: 前向传播需要完整模型

Adapter 不是独立的模型，它需要：
- 原始 Transformer 计算注意力
- Adapter 修改输出（残差连接）

```
output = transformer_block(hidden_states) + adapter(hidden_states)
         ↑                        ↑
    需要完整模型            需要完整模型作为输入
```

### 原因 2: 损失计算需要两个模型的输出

```python
# Unlearning Loss: 需要 unlearned 模型的输出
v_unlearned = model_unlearned(x_t, t, cond_target)

# Preservation Loss: 需要 original 模型的输出（对比）
v_original = model_original(x_t, t, cond_preserve)

loss = ||v_unlearned - v_negative||² + ||v_unlearned - v_original||²
```

### 原因 3: 激活值需要完整模型

前向传播时，每一层的激活值都需要存储（用于反向传播）：
- Transformer 的激活值：~100-150 GB
- Adapter 的激活值：很小，可以忽略

即使只更新 Adapter，也需要存储完整模型的激活值。

## 显存占用对比

### 训练时（需要完整模型）

| 组件 | 显存 | 说明 |
|------|------|------|
| Original Model | ~10 GB | 完整模型（冻结） |
| Unlearned Model | ~10 GB | 完整模型（冻结参数，但需要前向传播） |
| Adapter 参数 | ~0.1 GB | 很小 |
| 激活值 | ~100-150 GB | 完整模型的激活值 |
| **总计** | **~120-170 GB** | |

### 推理时（只需要 Adapter 权重）

| 组件 | 显存 | 说明 |
|------|------|------|
| 原始模型 | ~10 GB | 加载一次 |
| Adapter 权重 | ~0.1 GB | 注入到模型中 |
| 激活值 | ~10-20 GB | 推理时不需要存储所有激活值 |
| **总计** | **~20-30 GB** | |

## 类比：LoRA 训练

### LoRA 训练也需要完整模型

```python
# LoRA 训练时
base_model = load_model()  # 完整模型
lora = LoRALayer()  # LoRA 层
# 注入
model = base_model + lora

# 前向传播
output = model(input)  # 需要完整模型

# 反向传播
loss.backward()  # 只更新 LoRA 参数

# 保存
save_lora_weights()  # 只保存 LoRA 权重
```

**关键点**：LoRA 和 Adapter 一样，训练时都需要完整模型进行前向传播。

## 优化：能否只加载 Adapter？

### 理论上可以，但实现复杂

如果只加载 Adapter：
- ✅ 显存节省：~10 GB（不需要 unlearned model）
- ❌ 无法计算损失（需要完整模型输出）
- ❌ 无法进行前向传播（Adapter 依赖 Transformer）

### 实际方案：使用 CPU Offload

```bash
--use_cpu_offload
```

**效果**：
- 模型参数动态加载到 GPU
- 显存需求从 ~120-170 GB 降至 ~15-25 GB
- **这是最实用的优化方案**

## 总结

### 为什么训练时需要完整 unlearned model？

1. ✅ **前向传播需要**：Adapter 不是独立模型，需要完整 Transformer 计算
2. ✅ **损失计算需要**：需要 unlearned 模型的完整输出
3. ✅ **激活值需要**：反向传播需要存储完整模型的激活值

### 训练后保存什么？

- ✅ **只保存 Adapter 权重**（~12.3M 参数）
- ❌ **不保存完整模型**（可以从原始模型加载）

### 推理时如何使用？

1. 加载原始模型
2. 注入训练好的 Adapter 权重
3. 推理

### 显存优化

- 🥇 **CPU Offload**：最有效（节省 ~85-90%）
- 🥈 **FP16**：节省 ~50%
- 🥉 **减小 Batch Size**：线性减少激活值

**关键理解**：Adapter 是"修改器"，不是"替代器"。训练时需要完整模型来"修改"，但保存时只需要"修改器"本身。

