# T2VUnlearning 训练指南

## 核心思想

**T2VUnlearning 的训练不需要真实视频数据！**

只需要：
1. **Prompt**：文本提示词
2. **目标概念定义**：要消除的概念（如 "nudity"）
3. **保留概念定义**：要保留的概念（如 "person", "face"）

## 为什么不需要真实视频？

### 1. 负引导速度损失（Negatively-guided Velocity Loss）

负引导 velocity `v_neg` 的计算只需要：
- 随机噪声 latent `x_t`
- 随机时间步 `t`
- 目标概念的 prompt

```python
# v_neg 的计算过程：
v_c = model(x_t, t, cond_target)      # 目标概念上的 velocity
v_uncond = model(x_t, t, uncond)      # 无条件 velocity
v_negative = v_uncond + scale * (v_uncond - v_c)  # 负引导 velocity
```

**不需要真实视频**，因为：
- `x_t` 是随机噪声，可以从 `torch.randn()` 生成
- `v_c` 和 `v_uncond` 都是模型预测，只需要 prompt

### 2. 概念保留损失（Concept Preservation Loss）

概念保留损失需要：
- 随机噪声 latent `x_t`
- 保留概念的 prompt
- 原模型和去学习后模型的 velocity 预测

```python
v_original = model_original(x_t, t, cond_preserve)
v_unlearned = model_unlearned(x_t, t, cond_preserve)
loss = ||v_unlearned - v_original||²
```

**不需要真实视频**，因为：
- `x_t` 是随机噪声
- velocity 预测只需要 prompt

### 3. Mask-based Localization 损失

Mask 的提取需要：
- 一次前向传播来捕获注意力映射
- 目标概念的 prompt

```python
# 提取 mask
with AttnMapsCapture(model, attn_maps):
    _ = model(x_t, t, cond_target)
mask = get_mask(attn_maps, word_indices, ...)
```

**不需要真实视频**，因为：
- 注意力映射的提取只需要 prompt 和前向传播
- `x_t` 可以是随机噪声

## 训练流程

### 步骤 1: 准备 Prompt 数据

创建 prompt 文件（每行一个 prompt）：

```txt
# prompts.txt
nudity
a scene with nudity
video containing nudity
explicit content
...
```

或者使用评估数据中的 prompts：

```python
import pandas as pd
df = pd.read_csv("evaluation/data/nudity_cogvideox.csv")
prompts = df["prompt"].tolist()
```

### 步骤 2: 定义概念

创建保留概念文件：

```txt
# preserved_concepts.txt
person
face
clothing
background
scene
```

### 步骤 3: 运行训练

```bash
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/pretrained/cogvideox \
    --target_concept "nudity" \
    --preserved_concepts_file receler/preserved_concepts.txt \
    --prompts_file prompts.txt \
    --eraser_rank 128 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --output_dir ./output
```

## 训练数据格式

### PromptOnlyDataset

```python
dataset = PromptOnlyDataset(
    prompts=["nudity", "a scene with nudity", ...],
    target_concept="nudity",
    preserve_concepts=["person", "face"],
    num_samples=1000  # 可选：重复数据集
)
```

每个样本包含：
```python
{
    "prompt": "nudity",
    "target_concept": "nudity",
    "preserve_concepts": ["person", "face"]
}
```

### 训练时的数据生成

训练时，每个 batch 会：

1. **生成随机噪声：
```python
x_start = torch.randn(B, C, H, W, F)  # 随机噪声
```

2. **采样时间步**：
```python
t = torch.randint(0, 1000, (B,))
```

3. **添加噪声**：
```python
x_t = sqrt(alpha_t) * x_start + sqrt(1 - alpha_t) * noise
```

4. **编码 prompt**：
```python
cond_target = text_encoder("nudity")
cond_preserve = text_encoder("person, face")
```

5. **计算损失**：
```python
loss = unlearning_loss + preservation_loss + localization_loss
```

## 关键参数说明

### 损失权重

- `--unlearning_weight` (默认 1.0)：去学习损失权重
- `--preservation_weight` (默认 0.5)：保留损失权重
- `--localization_weight` (默认 1.0)：Mask-based localization 损失权重

### Latent 参数

需要根据实际模型调整：
- `--latent_channels` (默认 16)：Latent 通道数
- `--latent_height` (默认 10)：Latent 高度
- `--latent_width` (默认 10)：Latent 宽度
- `--num_frames` (默认 49)：帧数

### 训练参数

- `--batch_size` (默认 4)：批次大小
- `--learning_rate` (默认 1e-4)：学习率
- `--num_epochs` (默认 10)：训练轮数
- `--eraser_rank` (默认 128)：Eraser rank

## 训练过程

### 每个训练步骤

1. **生成随机 latent** `x_start`（不需要真实视频）
2. **采样时间步** `t`
3. **添加噪声**得到 `x_t`
4. **编码 prompt**得到条件嵌入
5. **提取 mask**（用于 localization）
6. **捕获 eraser 输出**（用于 localization）
7. **计算损失**：
   - 负引导速度损失
   - 概念保留损失
   - Mask-based localization 损失
8. **反向传播**更新 eraser 参数

### 损失计算

```python
# 1. 负引导速度损失
v_predicted = model_unlearned(x_t, t, cond_target)
v_negative = compute_negative_guided_velocity(...)
loss_unlearning = ||v_predicted - v_negative||²

# 2. 概念保留损失
v_original = model_original(x_t, t, cond_preserve)
v_unlearned = model_unlearned(x_t, t, cond_preserve)
loss_preservation = ||v_unlearned - v_original||²

# 3. Mask-based localization 损失
loss_localization = (1/L) * sum ||eraser_output * (1 - mask)||²

# 总损失
total_loss = w1 * loss_unlearning + w2 * loss_preservation + w3 * loss_localization
```

## 优势

### 1. 无需收集真实视频
- 不需要包含目标概念的真实视频数据
- 只需要 prompt 文本

### 2. 数据生成简单
- 随机噪声 latent 可以从 `torch.randn()` 生成
- 不需要 VAE 编码真实视频

### 3. 训练效率高
- 不需要加载和预处理视频数据
- 训练速度快

### 4. 灵活性高
- 可以轻松调整 prompt
- 可以快速测试不同的概念组合

## 注意事项

### 1. Latent 形状

需要根据实际模型调整 latent 的形状参数：
- `latent_channels`
- `latent_height`
- `latent_width`
- `num_frames`

### 2. Transformer 接口

训练代码中的 `model_wrapper` 需要根据实际模型的接口调整：

```python
def model_wrapper(x_t, t, cond):
    # 需要根据实际模型接口实现
    output = transformer(x_t, t, cond)
    return output
```

### 3. 注意力映射提取

Mask 提取需要正确的模型结构和注意力层：
- 确保模型有 cross-attention 层
- 确保可以注册 forward hook

### 4. Eraser 输出捕获

需要正确捕获 eraser 的输出：
- 确保 eraser 已正确注入到模型中
- 确保 hook 可以正确捕获输出

## 示例：训练去除 nudity 概念

```bash
# 1. 准备 prompts
echo "nudity" > prompts.txt
echo "a scene with nudity" >> prompts.txt
echo "explicit content" >> prompts.txt

# 2. 准备保留概念
cat > preserved_concepts.txt << EOF
person
face
clothing
background
EOF

# 3. 运行训练
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/cogvideox \
    --target_concept "nudity" \
    --preserved_concepts_file preserved_concepts.txt \
    --prompts_file prompts.txt \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --output_dir ./output
```

## 总结

T2VUnlearning 的训练**完全不需要真实视频数据**，只需要：

1. ✅ **Prompt 文本**（定义要处理的内容）
2. ✅ **目标概念**（要消除的）
3. ✅ **保留概念**（要保留的）
4. ✅ **随机噪声 latent**（训练时生成）

这使得训练过程非常灵活和高效！

