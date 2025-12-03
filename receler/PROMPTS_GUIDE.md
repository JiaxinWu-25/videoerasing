# Prompts.txt 文件格式指南

## 文件格式

`prompts.txt` 是一个简单的文本文件，每行一个 prompt：

```
nudity
naked body
explicit content
a scene with nudity
```

## Prompt 类型

### 1. 简单概念词（推荐用于训练）

最简单的格式，直接描述目标概念：

```
nudity
naked
explicit
adult content
```

**优点**：
- 简洁明了
- 训练速度快
- 适合快速迭代

**示例文件**：
```txt
# prompts_simple.txt
nudity
naked body
explicit content
adult content
nude
bare
unclothed
```

### 2. 完整视频描述（可选，用于更精确训练）

包含目标概念的完整视频描述：

```
A woman, free from clothing, stands in a dimly lit room
A scene containing explicit nudity
A video with adult content
```

**优点**：
- 更精确的描述
- 可以包含上下文信息
- 训练效果可能更好

**示例文件**：
```txt
# prompts_detailed.txt
A woman, free from clothing, stands in a dimly lit room
A scene containing explicit nudity and inappropriate content
A video with adult content showing naked bodies
A couple, naked and embracing, in an intimate setting
```

### 3. 从评估数据提取（推荐用于测试）

从 `evaluation/data/` 目录下的 CSV 文件中提取：

```bash
# 提取 nudity 相关的 prompts
python create_prompts_from_csv.py \
    --csv_file evaluation/data/nudity_cogvideox.csv \
    --output_file prompts_nudity.txt \
    --column_name prompt

# 提取 face 相关的 prompts
python create_prompts_from_csv.py \
    --csv_file evaluation/data/face_cogvideox.csv \
    --output_file prompts_face.txt \
    --column_name prompt
```

## 文件格式规则

### 基本规则

1. **每行一个 prompt**：每个 prompt 占一行
2. **注释行**：以 `#` 开头的行会被忽略
3. **空行**：空行会被忽略
4. **编码**：文件应使用 UTF-8 编码

### 示例文件

```txt
# T2VUnlearning Training Prompts
# Target Concept: nudity
# Date: 2024-01-01

# Simple concepts
nudity
naked body
explicit content

# Detailed descriptions
A woman, free from clothing, stands in a dimly lit room
A scene containing explicit nudity
A video with adult content

# More examples
nude
bare
unclothed
```

## 如何准备 Prompts

### 方法 1: 手动创建（简单概念）

创建 `prompts.txt`：

```txt
nudity
naked body
explicit content
adult content
```

### 方法 2: 从评估数据提取（推荐）

```bash
# 提取 nudity prompts
python receler/create_prompts_from_csv.py \
    --csv_file evaluation/data/nudity_cogvideox.csv \
    --output_file prompts.txt

# 只提取前 50 个
python receler/create_prompts_from_csv.py \
    --csv_file evaluation/data/nudity_cogvideox.csv \
    --output_file prompts.txt \
    --max_prompts 50
```

### 方法 3: 使用 Python 生成

```python
# 生成简单的 prompts
prompts = [
    "nudity",
    "naked body",
    "explicit content",
    "adult content",
    "nude",
    "bare",
    "unclothed",
    "a scene with nudity",
    "a video containing explicit content",
]

with open("prompts.txt", "w") as f:
    for prompt in prompts:
        f.write(f"{prompt}\n")
```

## Prompt 数量建议

| 用途 | 推荐数量 | 说明 |
|------|---------|------|
| 快速测试 | 10-20 | 用于快速验证训练流程 |
| 基础训练 | 50-100 | 用于基本的去学习效果 |
| 完整训练 | 100+ | 用于最佳的去学习效果 |

**注意**：可以使用 `--num_samples` 参数重复数据集，所以即使 prompt 数量少，也可以通过重复来增加训练样本。

## 不同目标概念的 Prompts 示例

### 去除 Nudity

```txt
nudity
naked body
explicit content
adult content
nude
bare
unclothed
a scene with nudity
a video containing explicit content
```

### 去除特定人物（如 Angela Merkel）

```txt
Angela Merkel
Angela Merkel speaking
Angela Merkel walking
Angela Merkel in a formal setting
a video featuring Angela Merkel
```

### 去除特定物体（如 airplane）

```txt
airplane
aircraft
flying plane
a scene with an airplane
a video showing airplanes
```

## 使用 Prompts 文件训练

```bash
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/pretrained/cogvideox \
    --target_concept "nudity" \
    --prompts_file prompts.txt \
    --preserved_concepts_file preserved_concepts.txt \
    --batch_size 4 \
    --num_epochs 10
```

## 注意事项

1. **Prompt 质量**：Prompt 应该准确描述目标概念
2. **多样性**：使用多样化的 prompts 可以提高训练效果
3. **长度**：Prompt 长度没有严格限制，但建议保持合理（通常 1-200 词）
4. **语言**：确保 prompts 使用模型支持的语言（通常是英语）

## 常见问题

### Q: Prompt 需要多详细？

A: 对于训练，简单的概念词（如 "nudity"）就足够了。详细的描述（如评估数据中的完整视频描述）可以用于更精确的训练，但不是必需的。

### Q: 需要多少个 prompts？

A: 最少 10-20 个，推荐 50-100 个。如果数量较少，可以使用 `--num_samples` 参数重复数据集。

### Q: 可以从评估数据中提取吗？

A: 可以！使用 `create_prompts_from_csv.py` 脚本可以轻松从 CSV 文件中提取 prompts。

### Q: Prompt 中需要包含保留概念吗？

A: 不需要。保留概念在 `preserved_concepts.txt` 中单独定义。训练时，prompts 用于目标概念，保留概念用于概念保留损失。

## 示例：完整的训练数据准备

```bash
# 1. 创建 prompts.txt（从评估数据提取）
python receler/create_prompts_from_csv.py \
    --csv_file evaluation/data/nudity_cogvideox.csv \
    --output_file prompts.txt \
    --max_prompts 100

# 2. 创建 preserved_concepts.txt
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
    --prompts_file prompts.txt \
    --preserved_concepts_file preserved_concepts.txt \
    --batch_size 4 \
    --num_epochs 10
```

