# T2VUnlearning Eraser 架构说明

## 1. 微调部分

### 只微调 AdapterEraser，不微调原始模型

T2VUnlearning **只微调 AdapterEraser（适配器）部分**，原始 Transformer 模型**完全冻结**：

```python
# 冻结 transformer，只训练 eraser
for param in pipe_unlearned.transformer.parameters():
    param.requires_grad = False

# 只优化 eraser 参数
for eraser in erasers.values():
    eraser.train()
    for param in eraser.parameters():
        param.requires_grad = True
```

### 为什么只微调 Adapter？

1. **参数效率**：Adapter 参数量远小于完整模型（rank=128 时，每个 adapter 只有 ~0.1M 参数）
2. **避免灾难性遗忘**：冻结原始模型参数，只通过 adapter 进行局部修改
3. **显存友好**：只需要存储和更新 adapter 参数，不需要存储原始模型的梯度

## 2. Adapter 注入位置

### 每个 CogVideoXBlock 的 attn1（自注意力）后都会接 AdapterEraser

```python
# 代码位置: receler/erasers/cogvideo_erasers.py

def setup_cogvideo_adapter_eraser(model, eraser_rank, device, dtype):
    def replace_transformer_block(model):
        for name, module in model.named_modules():
            if isinstance(module, CogVideoXBlock):
                print("changing: ",name)
                original_attention = module.attn1  # 自注意力层
                modified_attention = CogVideoXWithEraser(original_attention, eraser_rank)
                module.attn1 = modified_attention  # 替换为带 adapter 的版本
```

### CogVideoXBlock 结构

```
CogVideoXBlock
├── attn1 (自注意力) ← AdapterEraser 注入在这里
├── attn2 (交叉注意力，如果有)
├── norm1, norm2
└── mlp (前馈网络)
```

### AdapterEraser 的工作方式

```python
class CogVideoXWithEraser(nn.Module):
    def forward(self, hidden_states, ...):
        # 1. 原始注意力计算
        hidden_states, encoder_hidden_states = self.attn(...)
        
        # 2. Adapter 残差连接
        if self.adapter.use_eraser:
            hidden_states = hidden_states + self.adapter(hidden_states)
        
        return hidden_states, encoder_hidden_states
```

**关键点**：
- Adapter 的输出通过**残差连接**添加到原始注意力输出
- 训练时，adapter 学习如何修改 hidden_states 以消除目标概念
- 推理时，可以通过 `adapter.use_eraser = False` 禁用 adapter

## 3. AdapterEraser 结构

### 架构

```python
class AdapterEraser(nn.Module):
    def __init__(self, dim, mid_dim):  # mid_dim = eraser_rank
        super().__init__()
        self.down = nn.Linear(dim, mid_dim)      # 降维: dim -> rank
        self.act = nn.GELU()                      # 激活函数
        self.up = zero_module(nn.Linear(mid_dim, dim))  # 升维: rank -> dim (初始化为0)
```

### 参数量计算

假设：
- `dim` = hidden_dim（Transformer 的隐藏维度，CogVideoX 通常是 1152）
- `rank` = eraser_rank（默认 128）

每个 AdapterEraser 的参数量：
```
参数量 = dim × rank + rank + rank × dim + dim
       = dim × rank × 2 + rank + dim
       ≈ 2 × dim × rank  (当 rank << dim 时)
```

对于 CogVideoX（dim=1152, rank=128）：
```
参数量 ≈ 2 × 1152 × 128 = 294,912 ≈ 0.3M 参数
```

### 总参数量

**CogVideoX-5b 实际有 41 个 CogVideoXBlock**（不是 28 层）：

```
总 Adapter 参数量 = N × 0.3M
                 = 41 × 0.3M
                 ≈ 12.3M 参数
```

**注意**：训练时会打印 "changing: transformer.blocks.0" 到 "transformer.blocks.40"，共 41 个 blocks。

## 4. Rank 参数对显存的影响

### Rank 与参数量关系

| Rank | 每个 Adapter 参数量 | 41 层总参数量 | FP16 显存 (参数) | FP16 显存 (训练) |
|------|-------------------|--------------|-----------------|----------------|
| 64   | ~0.15M            | ~6.2M        | ~12 MB          | ~37 MB         |
| 128  | ~0.30M            | ~12.3M       | ~25 MB          | ~74 MB         |
| 256  | ~0.59M            | ~24.2M       | ~48 MB          | ~145 MB        |
| 512  | ~1.18M            | ~48.4M       | ~97 MB          | ~290 MB        |

**训练时显存** = 参数 + 梯度 + 优化器状态 ≈ 参数 × 4

### Rank 对性能的影响

- **Rank 太小（如 32-64）**：
  - ✅ 显存占用最小
  - ⚠️ 可能表达能力不足，去学习效果较差
  - ⚠️ 可能需要更多训练步数

- **Rank 适中（128-256）**：
  - ✅ 平衡显存和性能
  - ✅ 通常能获得良好的去学习效果
  - ✅ 推荐范围

- **Rank 太大（512+）**：
  - ✅ 表达能力最强
  - ⚠️ 显存占用增加
  - ⚠️ 可能过拟合，影响其他概念

### 推荐 Rank 值

| GPU 显存 | 推荐 Rank | 说明 |
|---------|----------|------|
| < 24 GB | 64-96    | 最小配置 |
| 24-40 GB | 128      | 默认值，平衡性能 |
| 40-80 GB | 128-256  | 可以适当增加 |
| > 80 GB | 256-512  | 可以尝试更大值 |

## 5. 如何调整 Rank

### 命令行参数

```bash
python receler/train_unlearning_prompt_only.py \
    --model_path ./CogVideoX-5b \
    --target_concept "nudity" \
    --eraser_rank 64 \  # 从默认 128 降低到 64
    --batch_size 2 \
    --use_fp16 \
    --use_cpu_offload
```

### 显存节省估算

从 `rank=128` 降低到 `rank=64`：
- 参数量减少：~50%
- 训练显存节省：~25 MB（对于 28 层）
- 虽然节省不多，但可以配合其他优化措施

### 实际建议

**如果显存紧张**，优先考虑：
1. ✅ **使用 CPU Offload**（节省最多，~60-70%）
2. ✅ **使用 FP16**（节省 ~50%）
3. ✅ **减小 Batch Size**（线性减少）
4. ⚠️ **降低 Rank**（节省较少，~25 MB，但可能影响效果）

**如果显存充足**，可以：
- 保持 `rank=128`（默认值，性能最佳）
- 或者尝试 `rank=256`（如果显存允许）

## 6. 代码位置

### 关键文件

1. **Adapter 定义**: `receler/erasers/utils.py`
   - `class AdapterEraser`

2. **Adapter 注入**: `receler/erasers/cogvideo_erasers.py`
   - `setup_cogvideo_adapter_eraser()`
   - `class CogVideoXWithEraser`

3. **训练脚本**: `receler/train_unlearning_prompt_only.py`
   - `--eraser_rank` 参数

### 查看有多少层被注入 Adapter

训练时会打印：
```
changing:  transformer.blocks.0
changing:  transformer.blocks.1
...
changing:  transformer.blocks.40
```

每个 "changing" 对应一个被注入 Adapter 的 CogVideoXBlock。

**CogVideoX-5b 有 41 个 Transformer Blocks**（索引 0-40），所以会看到 41 个 "changing" 输出。

## 7. 总结

### T2VUnlearning 微调了什么？

✅ **只微调 AdapterEraser**（~8.4M 参数，rank=128）
❌ **不微调原始 Transformer**（~5B 参数，完全冻结）

### Adapter 注入位置？

✅ **每个 CogVideoXBlock 的 attn1（自注意力）后**

### Rank 可以调小吗？

✅ **可以**，但：
- 显存节省有限（~25 MB for rank 128→64）
- 可能影响去学习效果
- 建议优先使用 CPU Offload 和 FP16

### 推荐配置

**显存紧张时**：
```bash
--eraser_rank 64 \
--use_fp16 \
--use_cpu_offload \
--batch_size 1
```

**显存充足时**：
```bash
--eraser_rank 128 \  # 默认值，性能最佳
--use_fp16 \
--batch_size 2-4
```

