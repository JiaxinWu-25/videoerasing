# 自动生成 Preserved Concepts 功能总结

## ✅ 功能已实现

已成功实现**自动生成保留概念**功能，可以根据目标概念自动生成10-15个语义相关的保留概念。

## 🎯 核心特性

### 1. 智能概念映射

基于预定义的概念映射字典，支持常见场景：

| 目标概念 | 自动生成的保留概念示例 |
|---------|---------------------|
| `nudity` | person, face, clothing, background, scene, body, hair, hands, gesture, expression... |
| `airplane` | sky, clouds, airport, person, ground, trees, mountains, runway, building... |
| `face` | person, body, clothing, background, scene, hair, hands, eyes, nose, mouth... |
| `person` | face, clothing, background, scene, body, posture, gesture, movement... |

### 2. 智能匹配策略

- ✅ **精确匹配**：直接查找目标概念
- ✅ **部分匹配**：检查目标概念是否包含映射中的键
- ✅ **关键词匹配**：根据常见关键词（如 "nudity", "naked", "face"）匹配

### 3. 自动数量控制

- 默认生成 **15个** 保留概念
- 可自定义数量（10-20个推荐）
- 自动去重和排序

## 📝 使用方法

### 方法 1: 独立生成（推荐）

```bash
# 生成保留概念文件
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 15 \
    --output_file preserved_concepts.txt
```

**输出示例** (`preserved_concepts.txt`):
```txt
# Preserved Concepts for Target Concept: nudity
# Auto-generated preserved concepts
# Total: 15 concepts

background
body
clothing
dress
environment
expression
face
facial features
garment
gesture
hair
hands
human
individual
legs
```

### 方法 2: 训练时自动生成（最简单）

```bash
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/cogvideox \
    --target_concept "nudity" \
    --auto_generate_preserved \
    --num_preserved_concepts 15 \
    --prompts_file prompts.txt \
    --batch_size 4 \
    --num_epochs 10
```

训练脚本会自动：
1. ✅ 根据 `target_concept` 生成保留概念
2. ✅ 使用生成的保留概念进行训练
3. ✅ 可选：保存生成的概念到文件

## 🔍 生成示例

### 示例 1: Nudity

```bash
python receler/auto_generate_preserved_concepts.py --target_concept "nudity"
```

**生成结果**（15个概念）:
```
background, body, clothing, dress, environment, expression, 
face, facial features, garment, gesture, hair, hands, 
human, individual, legs
```

### 示例 2: Airplane

```bash
python receler/auto_generate_preserved_concepts.py --target_concept "airplane"
```

**生成结果**（15个概念）:
```
air, airport, atmosphere, background, building, clouds, 
ground, landscape, mountains, people, person, pilot, 
runway, scene, sky
```

### 示例 3: Face

```bash
python receler/auto_generate_preserved_concepts.py --target_concept "face"
```

**生成结果**（15个概念）:
```
background, body, clothing, expression, eyes, garment, 
gesture, hair, hands, human, individual, mouth, nose, 
outfit, people
```

## 🎨 生成策略说明

### 1. 预定义映射（主要方法）

脚本包含预定义的概念映射，涵盖：
- **Nudity相关**：nudity, naked, explicit, adult content
- **Person相关**：face, person, human, people
- **Object相关**：airplane, car, bicycle
- **Violence相关**：violence, weapon
- **特定人物**：angela merkel（可扩展）

### 2. 通用保留概念

默认包含通用概念：
- `background`, `scene`, `setting`, `environment`
- `lighting`, `color`, `texture`
- `composition`, `framing`, `camera angle`

### 3. 智能匹配

即使目标概念不在预定义列表中，也会：
- 检查关键词匹配
- 返回通用保留概念
- 确保至少有基本的保留概念

## 📊 概念数量建议

| 场景 | 推荐数量 | 说明 |
|------|---------|------|
| 快速测试 | 10个 | 快速验证功能 |
| 基础训练 | 12-15个 | 平衡效果和效率 |
| 完整训练 | 15-20个 | 最佳去学习效果 |

## 🚀 完整训练流程

### 一键训练（最简单）

```bash
# 1. 准备 prompts（可选）
python receler/create_prompts_from_csv.py \
    --csv_file evaluation/data/nudity_cogvideox.csv \
    --output_file prompts.txt

# 2. 运行训练（自动生成保留概念）
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/cogvideox \
    --target_concept "nudity" \
    --auto_generate_preserved \
    --num_preserved_concepts 15 \
    --prompts_file prompts.txt \
    --batch_size 4 \
    --num_epochs 10 \
    --output_dir ./output
```

**就这么简单！** 无需手动指定保留概念。

## 🔧 高级选项

### 保存生成的概念

```bash
python receler/train_unlearning_prompt_only.py \
    --target_concept "nudity" \
    --auto_generate_preserved \
    --save_generated_preserved preserved_concepts_nudity.txt \
    ...
```

### 调整概念数量

```bash
# 生成更多概念
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 20

# 生成较少概念
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 10
```

### 不包含通用概念

```bash
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --no_common
```

## 📁 文件结构

```
receler/
├── auto_generate_preserved_concepts.py  # 自动生成脚本
├── PRESERVED_CONCEPTS_GUIDE.md          # 详细指南
├── QUICK_START.md                       # 快速开始
└── preserved_concepts.txt               # 生成的保留概念文件（可选）
```

## ✨ 优势

1. **自动化**：无需手动指定每个概念的保留列表
2. **智能匹配**：根据目标概念自动生成语义相关的概念
3. **防止遗忘**：确保相关非目标概念不会丢失
4. **灵活配置**：可以调整数量、保存文件等
5. **易于使用**：一键生成，集成到训练流程

## 🎯 使用建议

1. **首次使用**：使用 `--auto_generate_preserved` 自动生成
2. **检查结果**：生成后检查保留概念列表是否合理
3. **手动调整**：如有需要，可以手动编辑生成的文件
4. **保存文件**：使用 `--save_generated_preserved` 保存以便复用

## 📝 示例：完整工作流

```bash
# 1. 生成保留概念（查看结果）
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 15 \
    --output_file preserved_concepts.txt

# 2. 检查生成的概念
cat preserved_concepts.txt

# 3. 运行训练（使用生成的概念）
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/cogvideox \
    --target_concept "nudity" \
    --preserved_concepts_file preserved_concepts.txt \
    --prompts_file prompts.txt \
    --batch_size 4 \
    --num_epochs 10
```

或者更简单：

```bash
# 一键训练（自动生成保留概念）
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/cogvideox \
    --target_concept "nudity" \
    --auto_generate_preserved \
    --prompts_file prompts.txt \
    --batch_size 4 \
    --num_epochs 10
```

## ✅ 测试结果

已测试的目标概念：
- ✅ `nudity` → 15个相关概念
- ✅ `airplane` → 15个相关概念  
- ✅ `face` → 15个相关概念

所有测试通过！功能正常工作。

