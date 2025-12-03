# T2VUnlearning 快速开始指南

## 完整训练流程（自动生成保留概念）

### 步骤 1: 准备 Prompts

```bash
# 从评估数据提取 prompts
python receler/create_prompts_from_csv.py \
    --csv_file evaluation/data/nudity_cogvideox.csv \
    --output_file prompts.txt \
    --max_prompts 100
```

### 步骤 2: 自动生成保留概念（推荐）

```bash
# 自动生成保留概念
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 15 \
    --output_file preserved_concepts.txt
```

或者直接在训练时自动生成：

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

### 步骤 3: 运行训练

```bash
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/pretrained/cogvideox \
    --target_concept "nudity" \
    --auto_generate_preserved \
    --num_preserved_concepts 15 \
    --prompts_file prompts.txt \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --output_dir ./output
```

## 一键训练（最简单）

```bash
python receler/train_unlearning_prompt_only.py \
    --model_path /path/to/cogvideox \
    --target_concept "nudity" \
    --auto_generate_preserved \
    --prompts_file prompts.txt \
    --batch_size 4 \
    --num_epochs 10
```

**就这么简单！** 训练脚本会自动：
1. ✅ 生成保留概念（15个，与目标概念语义相关）
2. ✅ 加载 prompts
3. ✅ 开始训练（无需真实视频）

## 支持的自动生成概念

当前支持的目标概念包括：

- `nudity`, `naked`, `explicit`, `adult content`
- `face`, `person`, `human`, `people`
- `airplane`, `car`, `bicycle`
- `violence`, `weapon`
- `angela merkel`（或其他特定人物）
- 更多...

即使目标概念不在预定义列表中，脚本也会智能匹配相关概念。

