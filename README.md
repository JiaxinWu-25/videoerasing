# T2VUnlearning

This project follows up on the work of [T2VUnlearning](https://github.com/VDIGPKU/T2VUnlearning/tree/main). We would like to express our gratitude to the original authors for their codebase.

Based on their framework, we have implemented the **training code** from scratch, which is not released by their framework, enabling the training of Erasers/Adapters for video generation models (e.g., CogVideoX).

If you find this code useful, please also consider citing/starring the original repository.

## Features

- ✅ **Concept Erasing**: Effectively remove target concepts from text-to-video models
- ✅ **Concept Preservation**: Preserve non-target concepts to avoid catastrophic forgetting
- ✅ **Mask-based Localization**: Precisely localize target concepts using attention maps
- ✅ **Prompt-only Training**: Train without real video data, only using prompts
- ✅ **Memory Efficient**: Support CPU offload and FP16 for low-memory training
- ✅ **Auto Preserved Concepts**: Automatically generate semantically related concepts to preserve

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Code Structure](#code-structure)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)

## Installation

1. **Create a Conda environment with Python 3.10:**

```bash
conda create -n eraser python=3.10
conda activate eraser
```

2. **Install the required dependencies:**

Install specific versions of `torch`, `transformers`, and `accelerate` to ensure reproducibility. While other versions may work (as long as they are compatible with `diffusers`), we recommend using the versions below:

```bash
pip install torch==2.6.0 transformers==4.48.0 accelerate==1.1.0
```

3. **Install `diffusers` from source:**

We use `diffusers==0.33.0.dev0`. Please install it with the provided source file.

```bash
cd diffusers
pip install -e .
cd examples/cogvideo
pip install -r requirements.txt
pip install opencv-python omegaconf imageio imageio-ffmpeg
```

## Quick Start

### Training

Train an eraser adapter to remove a target concept:

```bash
CUDA_VISIBLE_DEVICES=0 python receler/train_unlearning_prompt_only.py \
    --model_path ./CogVideoX-5b \
    --target_concept "nudity" \
    --auto_generate_preserved \
    --num_preserved_concepts 15 \
    --prompts_file nudity_prompts.txt \
    --batch_size 2 \
    --num_epochs 10 \
    --num_frames 24 \
    --use_fp16 \
    --use_cpu_offload \
    --output_dir ./output/nudity_unlearning
```

### Inference

Use the trained eraser adapter for inference:

```bash
CUDA_VISIBLE_DEVICES=0 python test_cogvideo.py \
    --prompt="A beautiful scene" \
    --model_path ./CogVideoX-5b \
    --eraser_path ./output/nudity_unlearning/cogvideox_nudity_eraser \
    --eraser_rank=128 \
    --num_frames=49 \
    --generate_clean \
    --output_path ./output/test \
    --seed=42
```

## Training

### Overview

T2VUnlearning uses **AdapterEraser** (similar to LoRA) to modify the model's behavior. The training process:

1. **Only trains Adapter parameters** (~12.3M parameters for rank=128)
2. **Freezes the original model** (~5B parameters)
3. **Uses prompt-only training** (no real video data needed)
4. **Applies three loss functions**:
   - Negatively-guided velocity prediction loss (unlearning)
   - Concept preservation loss (prevent forgetting)
   - Mask-based localization loss (precise targeting)

### Training Scripts

#### `train_unlearning_prompt_only.py` (Recommended)

**Main training script** that supports:
- Prompt-only training (no video data required)
- Automatic preserved concepts generation
- Memory-efficient training (CPU offload, FP16)
- Mask-based localization
- Checkpoint saving

**Key Features:**
- ✅ No video data needed - only prompts and concept definitions
- ✅ Automatic preserved concepts generation using LLM or word embeddings
- ✅ Memory optimization with CPU offload and FP16
- ✅ Attention mask extraction for precise concept localization
- ✅ Comprehensive loss tracking and checkpointing

**Usage:**

```bash
python receler/train_unlearning_prompt_only.py \
    --model_path ./CogVideoX-5b \
    --target_concept "nudity" \
    --auto_generate_preserved \
    --num_preserved_concepts 15 \
    --prompts_file nudity_prompts.txt \
    --batch_size 2 \
    --num_epochs 10 \
    --num_frames 24 \
    --use_fp16 \
    --use_cpu_offload \
    --output_dir ./output
```

**Key Arguments:**
- `--target_concept`: Concept to erase (e.g., "nudity")
- `--auto_generate_preserved`: Automatically generate preserved concepts
- `--num_preserved_concepts`: Number of preserved concepts (default: 15)
- `--prompts_file`: File containing training prompts
- `--use_cpu_offload`: Use CPU offload for memory efficiency (recommended)
- `--use_fp16`: Use FP16 precision (recommended)
- `--eraser_rank`: Adapter rank (default: 128, can reduce to 64 for lower memory)

#### `train_unlearning_example.py`

Example training script demonstrating the basic usage of loss functions and mask-based localization.

### Memory Requirements

**Without CPU Offload:**
- FP32: ~300-400 GB
- FP16: ~150-200 GB

**With CPU Offload (Recommended):**
- FP16: ~15-25 GB (batch_size=2)
- FP16: ~10-15 GB (batch_size=1)

**Memory Estimation Tool:**

```bash
python receler/estimate_memory.py \
    --model_path ./CogVideoX-5b \
    --batch_size 2 \
    --num_frames 24 \
    --dtype float16 \
    --recommend
```

See [MEMORY_ESTIMATION.md](./receler/MEMORY_ESTIMATION.md) for detailed memory breakdown.

### Training Data

#### Prompts File Format

Create a text file with one prompt per line:

```text
# nudity_prompts.txt
a scene with nudity
nudity in the background
explicit nudity content
...
```

See [PROMPTS_GUIDE.md](./receler/PROMPTS_GUIDE.md) for more details.

#### Preserved Concepts

**Option 1: Automatic Generation (Recommended)**

```bash
--auto_generate_preserved \
--num_preserved_concepts 15
```

**Option 2: Manual Specification**

Create `preserved_concepts.txt`:
```text
clothing
fashion
apparel
...
```

See [AUTO_PRESERVED_CONCEPTS_SUMMARY.md](./receler/AUTO_PRESERVED_CONCEPTS_SUMMARY.md) for details.

## Inference

### CogVideoX

```bash
CUDA_VISIBLE_DEVICES=0 python test_cogvideo.py \
    --prompt="[Test prompt]" \
    --model_path=[Path of pretrained CogVideoX diffusers weight] \
    --eraser_path=[Path of nudity erasure adapter] \
    --eraser_rank=128 \
    --num_frames=[Number of frames to generate. Default 49] \
    --generate_clean \
    --output_path=[Prefix for output videos] \
    --seed=42
```

After running the script, you should find two output videos: `[output_path]_clean.mp4` and `[output_path]_erased.mp4`, corresponding to the results from the original model and the unlearned model, respectively.

### HunyuanVideo

```bash
CUDA_VISIBLE_DEVICES=0 python test_hunyuan.py \
    --prompt="[Test prompt]" \
    --model_path=[Path of pretrained HunyuanVideo diffusers weight] \
    --eraser_path=[Path of nudity erasure adapter] \
    --eraser_rank=128 \
    --num_frames=[Number of frames to generate. Default 49] \
    --generate_clean \
    --output_path=[Prefix for output videos] \
    --seed=42
```

We also include inference script of SAFREE (`test_safree_hunyuan.sh`) and negative prompting (`test_neg_hunyuan.sh`) for HunyuanVideo.

Evaluation prompt datasets can be found in `evaluation/data`.

## Code Structure

### Core Training Components

#### `receler/train_unlearning_prompt_only.py`
**Main training script** for prompt-only unlearning training.

**Key Features:**
- Prompt-only training (no video data)
- Automatic preserved concepts generation
- Memory-efficient training with CPU offload
- Mask-based localization
- Comprehensive logging and checkpointing

#### `receler/unlearning_losses.py`
**Loss function implementations**:

- `NegativelyGuidedVelocityLoss`: Unlearning loss using negatively-guided velocity prediction
- `ConceptPreservationLoss`: Prevents catastrophic forgetting of non-target concepts
- `MaskLocalizationLoss`: Enforces zero eraser output in non-target regions
- `T2VUnlearningLoss`: Combined loss function integrating all three losses

**Mathematical Foundation:**
- Based on Bayes' theorem and score-velocity equivalence
- Derives negative-guided velocity to reduce target concept probability
- Uses velocity difference for concept preservation

#### `receler/concept_reg_cogvideo.py`
**Mask-based localization implementation**:

- `AttnMapsCapture`: Captures attention maps from transformer
- `get_mask`: Extracts attention masks from QK maps
- `EraserOutputsCapture`: Captures eraser outputs for localization loss

**How it works:**
1. Extract attention maps from full-attention QK maps
2. Threshold attention maps to identify target concept regions
3. Use masks to restrict eraser updates to target regions only

#### `receler/auto_generate_preserved_concepts.py`
**Automatic preserved concepts generation**:

- `PreservedConceptsGenerator`: Generates semantically related concepts
- Supports LLM-based generation (OpenAI API)
- Supports word embedding-based generation (WordNet, ConceptNet)

**Usage:**
```bash
python receler/auto_generate_preserved_concepts.py \
    --target_concept "nudity" \
    --num_concepts 15 \
    --output_file preserved_concepts.txt
```

#### `receler/estimate_memory.py`
**Memory estimation tool**:

- Estimates training memory requirements
- Recommends optimal configurations based on available GPU memory
- Provides detailed memory breakdown

**Usage:**
```bash
python receler/estimate_memory.py \
    --model_path ./CogVideoX-5b \
    --batch_size 2 \
    --recommend
```

### Eraser Implementation

#### `receler/erasers/cogvideo_erasers.py`
**CogVideoX eraser implementation**:

- `CogVideoXWithEraser`: Wraps attention layer with adapter eraser
- `setup_cogvideo_adapter_eraser`: Injects adapter into transformer blocks
- `save_cogvideo_eraser_from_transformer`: Saves eraser weights

**Architecture:**
- Adapter injected after `attn1` (self-attention) in each `CogVideoXBlock`
- Adapter structure: `Linear(dim -> rank) -> GELU -> Linear(rank -> dim)`
- Total: 41 transformer blocks × ~0.3M parameters = ~12.3M parameters (rank=128)

### Utilities

#### `receler/utils.py`
General utility functions for model loading, data processing, etc.

## Documentation

### Training Guides

- **[TRAINING_GUIDE.md](./receler/TRAINING_GUIDE.md)**: Comprehensive training guide
- **[QUICK_START.md](./receler/QUICK_START.md)**: Quick start guide
- **[PROMPTS_GUIDE.md](./receler/PROMPTS_GUIDE.md)**: Guide for creating prompts

### Loss Functions

- **[UNLEARNING_LOSS_README.md](./receler/UNLEARNING_LOSS_README.md)**: Detailed explanation of unlearning losses
  - Negatively-guided velocity prediction
  - Concept preservation regularization
  - Mask-based localization loss

### Memory Management

- **[MEMORY_ESTIMATION.md](./receler/MEMORY_ESTIMATION.md)**: Memory estimation guide
- **[MEMORY_BREAKDOWN.md](./receler/MEMORY_BREAKDOWN.md)**: Detailed memory breakdown
- **[TRAINING_VS_INFERENCE.md](./receler/TRAINING_VS_INFERENCE.md)**: Why training needs full models

### Architecture

- **[ERASER_ARCHITECTURE.md](./receler/ERASER_ARCHITECTURE.md)**: Adapter eraser architecture details
  - What gets fine-tuned
  - Adapter injection locations
  - Rank parameter effects

### Preserved Concepts

- **[AUTO_PRESERVED_CONCEPTS_SUMMARY.md](./receler/AUTO_PRESERVED_CONCEPTS_SUMMARY.md)**: Auto-generation guide
- **[PRESERVED_CONCEPTS_GUIDE.md](./receler/PRESERVED_CONCEPTS_GUIDE.md)**: Manual specification guide

## Key Features Added

### 1. Prompt-Only Training

**File**: `train_unlearning_prompt_only.py`

- ✅ No video data required - only prompts and concept definitions
- ✅ Random latent noise generation for training
- ✅ Efficient and scalable training process

### 2. Automatic Preserved Concepts Generation

**File**: `auto_generate_preserved_concepts.py`

- ✅ LLM-based generation (OpenAI API)
- ✅ Word embedding-based generation (WordNet, ConceptNet)
- ✅ Generates 10-15 semantically related concepts automatically

### 3. Memory-Efficient Training

**Files**: `train_unlearning_prompt_only.py`, `estimate_memory.py`

- ✅ CPU offload support (saves ~85-90% memory)
- ✅ FP16 precision support (saves ~50% memory)
- ✅ Memory estimation tool for planning
- ✅ Automatic memory checking and recommendations

### 4. Mask-Based Localization

**File**: `concept_reg_cogvideo.py`

- ✅ Attention map extraction from QK maps
- ✅ Automatic mask generation from attention maps
- ✅ Localization loss to restrict eraser updates

### 5. Comprehensive Loss Functions

**File**: `unlearning_losses.py`

- ✅ Negatively-guided velocity prediction loss
- ✅ Concept preservation loss
- ✅ Mask-based localization loss
- ✅ Combined loss with configurable weights

## Training Workflow

```
1. Prepare Prompts
   └─> Create prompts.txt with target concept prompts

2. Generate Preserved Concepts (Optional)
   └─> Auto-generate or manually specify preserved_concepts.txt

3. Estimate Memory Requirements
   └─> Run estimate_memory.py to check GPU memory

4. Train Eraser
   └─> Run train_unlearning_prompt_only.py
       ├─> Loads two models (original + unlearned)
       ├─> Injects adapter into transformer blocks
       ├─> Trains only adapter parameters
       └─> Saves eraser weights (~12.3M parameters)

5. Inference
   └─> Load original model + eraser weights
       └─> Generate videos without target concept
```

## Model Checkpoints

Download pre-trained eraser adapters from [Google Drive](https://drive.google.com/drive/folders/11r1dS2vzmbFeJZeDVZGsb2z9Tkrx64I1?usp=sharing).

## Troubleshooting

### Out of Memory (OOM)

1. **Use CPU Offload**:
   ```bash
   --use_cpu_offload
   ```

2. **Use FP16**:
   ```bash
   --use_fp16
   ```

3. **Reduce Batch Size**:
   ```bash
   --batch_size 1
   ```

4. **Reduce Rank** (less effective):
   ```bash
   --eraser_rank 64
   ```

5. **Estimate Memory First**:
   ```bash
   python receler/estimate_memory.py --model_path ./CogVideoX-5b --recommend
   ```

### Training Issues

- **Check prompts file**: Ensure `prompts_file` contains valid prompts
- **Check preserved concepts**: Use `--auto_generate_preserved` or provide `preserved_concepts_file`
- **Monitor loss**: Check `loss_dict` output for unlearning/preservation/localization losses

## Citation

If you use this code in your research, please cite:

```bibtex
@article{t2vunlearning,
  title={T2VUnlearning: A Concept Erasing Method for Text-to-Video Diffusion Models},
  author={...},
  journal={...},
  year={2024}
}
```

## Acknowledgements

This repository is built upon the excellent work of the following projects:

- [Receler](https://github.com/jasper0314-huang/Receler)
- [finetrainers](https://github.com/a-r-r-o-w/finetrainers)
- [diffusers](https://github.com/huggingface/diffusers)

We sincerely thank the authors and contributors of these projects for their valuable tools, insights, and open-source efforts.

## License

[Add your license here]

## Contact

[Add contact information here]



