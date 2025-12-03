# 自动生成 Preserved Concepts（保留概念）指南

## 概述

`auto_generate_preserved_concepts.py` 可以根据目标概念自动生成语义相关的保留概念列表，用于防止去学习导致的灾难性遗忘。

## 为什么需要自动生成？

在去学习过程中，如果只关注消除目标概念，可能会导致模型忘记相关的非目标概念。例如：
- 消除 "nudity" 时，可能也会影响 "person"、"face"、"clothing" 等概念
- 消除 "airplane" 时，可能也会影响 "sky"、"clouds"、"airport" 等概念

通过自动生成保留概念，可以：
1. **防止灾难性遗忘**：保持模型在相关概念上的能力
2. **提高训练效率**：无需手动指定每个概念的保留列表
3. **确保语义相关性**：生成的保留概念与目标概念逻辑相关

## 使用方法

### 基本使用

```bash
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 15
```

这会生成 `preserved_concepts_nudity.txt` 文件。

### 指定输出文件

```bash
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --output_file my_preserved_concepts.txt \
    --num_concepts 15
```

### 调整概念数量

```bash
# 生成更多概念（20个）
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 20

# 生成较少概念（10个）
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 10
```

### 不包含通用概念

```bash
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 15 \
    --no_common
```

## 生成策略

### 1. 预定义概念映射（主要方法）

脚本包含预定义的概念映射字典，涵盖常见场景：

| 目标概念 | 保留概念示例 |
|---------|------------|
| `nudity` | person, face, clothing, background, scene, body, hair, hands |
| `airplane` | sky, clouds, airport, person, ground, trees, mountains |
| `face` | person, body, clothing, background, scene, hair, hands |
| `person` | face, clothing, background, scene, body, posture, gesture |
| `violence` | person, face, clothing, background, scene, body, environment |

### 2. 智能匹配

脚本会进行智能匹配：
- **精确匹配**：直接查找目标概念
- **部分匹配**：检查目标概念是否包含映射中的键
- **关键词匹配**：根据常见关键词（如 "nudity", "naked", "face"）匹配

### 3. 通用保留概念

默认包含通用保留概念：
- `background`, `scene`, `setting`, `environment`
- `lighting`, `color`, `texture`
- `composition`, `framing`, `camera angle`

## 示例输出

### 示例 1: Nudity

```bash
python receler/auto_generate_preserved_concepts.py --target_concept "nudity"
```

输出文件 `preserved_concepts_nudity.txt`:

```txt
# Preserved Concepts for Target Concept: nudity
# Auto-generated preserved concepts
# Total: 15 concepts
# Format: one concept per line

arms
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
individual
legs
movement
outfit
person
people
posture
scene
setting
skin
```

### 示例 2: Airplane

```bash
python receler/auto_generate_preserved_concepts.py --target_concept "airplane"
```

输出文件 `preserved_concepts_airplane.txt`:

```txt
# Preserved Concepts for Target Concept: airplane
# Auto-generated preserved concepts
# Total: 15 concepts
# Format: one concept per line

air
airport
atmosphere
background
building
clouds
ground
landscape
mountains
people
person
pilot
road
runway
scene
sky
trees
```

## 在训练中使用

生成保留概念后，可以在训练脚本中使用：

```bash
# 1. 生成保留概念
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --output_file preserved_concepts.txt

# 2. 使用生成的保留概念训练
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/cogvideox \
    --target_concept "nudity" \
    --preserved_concepts_file preserved_concepts.txt \
    --prompts_file prompts.txt \
    --batch_size 4 \
    --num_epochs 10
```

## 自定义概念映射

如果需要添加自定义的概念映射，可以修改 `auto_generate_preserved_concepts.py` 中的 `concept_mapping` 字典：

```python
self.concept_mapping = {
    # 添加你的自定义映射
    "your_target_concept": [
        "preserved_concept_1",
        "preserved_concept_2",
        "preserved_concept_3",
        # ...
    ],
    # ...
}
```

## 集成到训练流程

可以在训练脚本中自动生成保留概念：

```python
from receler.auto_generate_preserved_concepts import PreservedConceptsGenerator

# 自动生成保留概念
generator = PreservedConceptsGenerator()
preserved_concepts = generator.generate(
    target_concept="nudity",
    num_concepts=15
)

# 使用生成的保留概念
# ...
```

## 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--target_concept` | str | 目标概念（要消除的） | 必需 |
| `--output_file` | str | 输出文件路径 | `preserved_concepts_{target}.txt` |
| `--num_concepts` | int | 期望的保留概念数量 | 15 |
| `--use_llm` | flag | 使用LLM生成（如果可用） | False |
| `--no_common` | flag | 不包含通用保留概念 | False |

## 支持的目标概念

当前支持的目标概念包括：

### 已预定义映射的概念

- **Nudity相关**：`nudity`, `naked`, `explicit`, `adult content`
- **Person相关**：`face`, `person`, `human`, `people`
- **Object相关**：`airplane`, `car`, `bicycle`
- **Violence相关**：`violence`, `weapon`
- **通用**：`object`, `animal`

### 智能匹配

即使目标概念不在预定义列表中，脚本也会尝试：
1. 部分匹配（检查关键词）
2. 返回通用保留概念

## 最佳实践

1. **生成后检查**：生成后检查保留概念列表，确保合理
2. **根据场景调整**：根据具体场景添加或删除概念
3. **数量适中**：推荐 10-20 个保留概念
4. **语义相关**：确保保留概念与目标概念语义相关

## 示例：完整训练流程

```bash
# 1. 准备 prompts
python receler/create_prompts_from_csv.py \
    --csv_file evaluation/data/nudity_cogvideox.csv \
    --output_file prompts.txt

# 2. 自动生成保留概念
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --output_file preserved_concepts.txt \
    --num_concepts 15

# 3. 运行训练
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/cogvideox \
    --target_concept "nudity" \
    --prompts_file prompts.txt \
    --preserved_concepts_file preserved_concepts.txt \
    --batch_size 4 \
    --num_epochs 10
```

## 注意事项

1. **概念质量**：生成的概念应该与目标概念语义相关
2. **数量平衡**：太多概念可能影响去学习效果，太少可能导致遗忘
3. **手动调整**：生成后可以手动调整列表
4. **测试验证**：训练后测试模型在保留概念上的表现

## 扩展：使用LLM生成（未来功能）

未来版本可能支持使用LLM自动生成保留概念：

```bash
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --use_llm \
    --num_concepts 15
```

这将使用LLM根据目标概念生成语义相关的保留概念。

