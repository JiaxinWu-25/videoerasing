# 负引导速度损失函数实现说明

## 概述

本实现提供了 T2VUnlearning 方法中的**负引导速度预测损失**（Negatively-guided Velocity Prediction Loss）和**概念保留损失**（Concept Preservation Loss）。

## 文件说明

- `unlearning_losses.py`: 核心损失函数实现
- `train_unlearning_example.py`: 训练示例代码
- `UNLEARNING_LOSS_README.md`: 本文档

## 核心组件

### 1. MaskLocalizationLoss（Mask-based Localization 损失）

**理论依据**：
- 根据公式：`L_loc = (1/L) * sum_{l=1}^{L} ||o^l ⊙ (1 - M)||_2^2`
- 目标：迫使 eraser 在非目标概念区域（背景区域，即 1-M）输出为 0
- 将擦除"局限"到目标概念出现的视觉区域，避免对上下文产生过度遗忘

**实现原理**：
```python
# 1. 对每个 eraser 层 l，计算其在背景区域的输出
masked_output = eraser_output * (1 - M)  # ⊙ 表示元素级乘法

# 2. 计算 L2 范数平方
layer_loss = ||masked_output||_2^2

# 3. 对所有层求平均
L_loc = (1/L) * sum_{l=1}^{L} layer_loss
```

**关键参数**：
- `eraser_outputs`: 字典，包含所有 eraser 层的输出
- `mask`: 注意力掩码 M，1 表示目标概念区域，0 表示背景区域
- `loss_type`: 损失类型，"l2"（默认）或 "l1"

### 2. NegativelyGuidedVelocityLoss（负引导速度损失）

**理论依据**：
- 从概率角度：目标是在扩散时刻降低生成含目标概念 c 的概率
- 基于 Bayes 定理和 score → velocity 的等价关系
- 通过负引导 velocity 实现去学习

**实现原理**：
```python
# 1. 计算原模型在目标概念 c 上的 velocity
v_c = model(x_t, t, cond_c)

# 2. 计算无条件 velocity（降低概念 c 的概率）
v_uncond = model(x_t, t, uncond)

# 3. 计算负引导 velocity
v_negative = v_uncond + guidance_scale * (v_uncond - v_c)

# 4. 损失：使预测的 velocity 接近负引导 velocity
loss = ||v_predicted - v_negative||²
```

**关键参数**：
- `guidance_scale`: 引导强度，控制负引导的程度（默认 7.5）
- `loss_type`: 损失类型，"l2" 或 "l1"
- `mask`: 可选的空间掩码，用于 mask-based localization

### 3. ConceptPreservationLoss（概念保留损失）

**理论依据**：
- 防止灾难性遗忘：保持模型在相关非目标概念上的生成能力
- 例如：去除 nudity 时，保留 "person" 概念

**实现原理**：
```python
# 1. 原模型在保留概念上的 velocity
v_original = model_original(x_t, t, cond_preserve)

# 2. 去学习后模型在保留概念上的 velocity
v_unlearned = model_unlearned(x_t, t, cond_preserve)

# 3. 损失：保持原模型的能力
loss = ||v_unlearned - v_original||²
```

### 4. T2VUnlearningLoss（总损失函数）

结合负引导速度损失、概念保留损失和 Mask-based localization 损失：

```python
total_loss = unlearning_weight * unlearning_loss + 
             preservation_weight * preservation_loss +
             localization_weight * localization_loss
```

## 使用方法

### 基本使用

```python
from receler.unlearning_losses import T2VUnlearningLoss
import torch

# 1. 准备模型和配置
model_original = ...  # 原始模型（冻结）
model_unlearned = ...  # 去学习模型（可训练）
alphas_cumprod = model_unlearned.alphas_cumprod  # 从模型获取

# 2. 初始化损失函数
loss_fn = T2VUnlearningLoss(
    alphas_cumprod=alphas_cumprod,
    unlearning_weight=1.0,
    preservation_weight=0.5,
    guidance_scale=7.5,
    loss_type="l2"
)

# 3. 准备数据
x_t = ...  # 噪声样本 (B, C, H, W, F)
t = ...    # 时间步 (B,)
cond_target = ...  # 目标概念的条件嵌入（要去除的，如 "nudity"）
cond_preserve = ...  # 保留概念的条件嵌入（如 "person"）

# 4. 计算损失
total_loss, loss_dict = loss_fn(
    model_original=model_original,
    model_unlearned=model_unlearned,
    x_t=x_t,
    t=t,
    cond_target=cond_target,
    cond_preserve=cond_preserve
)

# 5. 反向传播
total_loss.backward()
optimizer.step()
```

### 结合 Mask-based Localization

```python
from receler.concept_reg_cogvideo import AttnMapsCapture, get_mask, EraserOutputsCapture

# 1. 提取注意力掩码
attn_maps = {}
with AttnMapsCapture(model_unlearned, attn_maps):
    _ = model_unlearned(x_t, t, cond_target)

# 2. 生成掩码
masks = get_mask(
    attn_maps=attn_maps,
    word_indices=word_indices,  # 目标概念的 token 索引
    thres=0.5,
    height=H,
    width=W,
    head_num=30,
    text_len=cond_target.shape[1]
)
mask = masks["average_mask"]

# 3. 捕获 eraser 输出（用于 localization 损失）
eraser_outputs = {}
with EraserOutputsCapture(model_unlearned, erasers, eraser_outputs):
    _ = model_unlearned(x_t, t, cond_target)

# 4. 在损失函数中使用掩码和 eraser 输出
total_loss, loss_dict = loss_fn(
    model_original=model_original,
    model_unlearned=model_unlearned,
    x_t=x_t,
    t=t,
    cond_target=cond_target,
    cond_preserve=cond_preserve,
    mask=mask,  # 传入掩码
    eraser_outputs=eraser_outputs  # 传入 eraser 输出
)
```

## 参数说明

### T2VUnlearningLoss 参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `alphas_cumprod` | torch.Tensor | 累积 alpha 值 (num_timesteps,) | 必需 |
| `unlearning_weight` | float | 去学习损失的权重 | 1.0 |
| `preservation_weight` | float | 保留损失的权重 | 0.5 |
| `localization_weight` | float | Mask-based localization 损失的权重 | 1.0 |
| `guidance_scale` | float | 负引导强度 | 7.5 |
| `loss_type` | str | 损失类型，"l2" 或 "l1" | "l2" |

### forward 方法参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_original` | nn.Module | 原始模型（冻结） |
| `model_unlearned` | nn.Module | 去学习后的模型（可训练） |
| `x_t` | torch.Tensor | 噪声样本 (B, C, H, W, ...) |
| `t` | torch.Tensor | 时间步 (B,) |
| `cond_target` | torch.Tensor | 目标概念的条件嵌入（要去除的概念） |
| `cond_preserve` | torch.Tensor (可选) | 保留概念的条件嵌入 |
| `cond_neg` | torch.Tensor (可选) | 负提示词的条件嵌入 |
| `uncond` | torch.Tensor (可选) | 无条件嵌入 |
| `mask` | torch.Tensor (可选) | 空间掩码，用于 mask-based localization |
| `eraser_outputs` | Dict[str, torch.Tensor] (可选) | eraser 输出字典，用于 mask-based localization |

## 损失字典说明

`loss_fn` 返回的 `loss_dict` 包含以下键：

- `total_loss`: 总损失
- `unlearning_loss_weighted`: 加权后的去学习损失
- `preservation_loss_weighted`: 加权后的保留损失
- `localization_loss_weighted`: 加权后的 localization 损失
- `unlearning/unlearning_loss`: 去学习损失（未加权）
- `unlearning/v_pred_norm`: 预测 velocity 的范数
- `unlearning/v_neg_norm`: 负引导 velocity 的范数
- `preservation/preservation_loss`: 保留损失（未加权）
- `preservation/v_orig_norm`: 原模型 velocity 的范数
- `preservation/v_unlearn_norm`: 去学习模型 velocity 的范数
- `localization/total_loss`: localization 损失（未加权）
- `localization/num_layers`: eraser 层数
- `localization/{eraser_name}/loss`: 每个 eraser 层的损失

## 注意事项

1. **模型接口要求**：
   - 模型需要支持 `model(x_t, t, cond)` 的调用方式
   - 模型需要返回 velocity（v-prediction 模式）

2. **条件嵌入**：
   - `cond_target`: 目标概念的条件嵌入（要去除的）
   - `cond_preserve`: 保留概念的条件嵌入（防止遗忘）
   - `cond_neg`: 负提示词的条件嵌入（可选）
   - `uncond`: 无条件嵌入（可选，如果模型支持）

3. **Mask-based Localization**：
   - 掩码形状需要匹配损失计算的维度
   - 掩码值：1 表示目标概念区域，0 表示背景区域
   - 使用掩码可以限制更新范围，避免过度遗忘

4. **优化器设置**：
   - 建议只优化 eraser（AdapterEraser）的参数
   - 保持原始模型冻结

## 示例场景

### 场景 1: 去除 nudity，保留 person

```python
loss_fn = T2VUnlearningLoss(
    alphas_cumprod=alphas_cumprod,
    unlearning_weight=1.0,
    preservation_weight=0.5,  # 保留 person 概念
    guidance_scale=7.5
)

# 目标概念：nudity
cond_target = text_encoder("nudity")

# 保留概念：person
cond_preserve = text_encoder("person")
```

### 场景 2: 使用负提示词

```python
# 使用负提示词增强去学习效果
cond_neg = text_encoder("not nudity, safe content")

total_loss, loss_dict = loss_fn(
    model_original=model_original,
    model_unlearned=model_unlearned,
    x_t=x_t,
    t=t,
    cond_target=cond_target,
    cond_preserve=cond_preserve,
    cond_neg=cond_neg  # 使用负提示词
)
```

## 理论背景

### 负引导速度预测

在 v-prediction 模式下，模型预测 velocity：
- `v = α_t * ε - σ_t * x_0`
- 其中 `α_t = sqrt(alphas_cumprod[t])`, `σ_t = sqrt(1 - alphas_cumprod[t])`

负引导的目标是降低 `P(x_t | c)`，即增加 `P(x_t | ¬c)`。

通过 classifier-free guidance 的方式：
```
v_negative = v_uncond + guidance_scale * (v_uncond - v_c)
```

这相当于引导模型远离目标概念 c。

### 概念保留

为防止灾难性遗忘，引入保留正则化：
- 选择语义相关的保留概念 `c_pre`
- 生成 `c_pre` 的伪数据
- 最小化去学习后模型与原模型在该概念上的 velocity 差异

## 参考

- T2VUnlearning 论文
- Receler 项目
- Classifier-free Guidance
- v-prediction 扩散模型

