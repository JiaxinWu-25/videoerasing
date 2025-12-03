"""
T2VUnlearning 训练脚本 - 仅使用 Prompt（无需真实视频）

核心思想：
1. 不需要真实视频数据，只需要随机噪声 latent 和 prompt
2. v_neg 只需要 target concept 就能计算出来
3. 训练过程在 flow matching（扩散过程）中进行
4. 只需要定义：目标概念（要消除的）和保留概念（要保留的）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict
import argparse
import os
import sys
import json

# 添加项目根目录到路径，支持直接运行脚本
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from diffusers import CogVideoXPipeline
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from receler.unlearning_losses import T2VUnlearningLoss
from receler.concept_reg_cogvideo import AttnMapsCapture, get_mask, EraserOutputsCapture
from receler.erasers.cogvideo_erasers import CogVideoXWithEraser, setup_cogvideo_adapter_eraser, save_cogvideo_eraser_from_transformer
from receler.auto_generate_preserved_concepts import PreservedConceptsGenerator


class PromptOnlyDataset(Dataset):
    """
    仅使用 Prompt 的数据集
    
    不需要真实视频，只需要：
    - prompt：文本提示词
    - target_concept：目标概念（要消除的）
    - preserve_concept：保留概念（要保留的）
    
    训练时会随机生成噪声 latent
    """
    
    def __init__(
        self,
        prompts: List[str],
        target_concept: str,
        preserve_concepts: List[str],
        num_samples: Optional[int] = None
    ):
        """
        Args:
            prompts: prompt 列表
            target_concept: 目标概念（要消除的，如 "nudity"）
            preserve_concepts: 保留概念列表（如 ["person", "face"]）
            num_samples: 如果提供，则重复数据集到这个数量
        """
        self.prompts = prompts
        self.target_concept = target_concept
        self.preserve_concepts = preserve_concepts
        
        if num_samples is not None and num_samples > len(prompts):
            # 重复数据集
            repeat_times = (num_samples // len(prompts)) + 1
            self.prompts = (self.prompts * repeat_times)[:num_samples]
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "target_concept": self.target_concept,
            "preserve_concepts": self.preserve_concepts
        }


def generate_random_latent(
    batch_size: int,
    num_frames: int,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    生成随机噪声 latent
    
    用于训练，不需要真实视频
    
    Args:
        batch_size: 批次大小
        num_frames: 帧数
        latent_channels: latent 通道数
        latent_height: latent 高度
        latent_width: latent 宽度
        device: 设备
    
    Returns:
        随机噪声 latent (B, C, H, W, F)
    """
    return torch.randn(
        batch_size, latent_channels, latent_height, latent_width, num_frames,
        device=device
    )


def train_step_prompt_only(
    transformer_original: nn.Module,
    transformer_unlearned: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: T2VUnlearningLoss,
    scheduler,
    tokenizer,
    text_encoder,
    prompts: List[str],
    target_concept: str,
    preserve_concepts: List[str],
    erasers: Dict[str, nn.Module],
    num_frames: int = 49,
    latent_channels: int = 16,
    latent_height: int = 10,
    latent_width: int = 10,
    num_timesteps: int = 1000,
    device: str = "cuda",
    text_encoder_device: Optional[torch.device] = None,
    gradient_accumulation: bool = False,
    gradient_accumulation_steps: int = 1
) -> Dict[str, float]:
    """
    执行一个训练步骤（仅使用 prompt，无需真实视频）
    
    Args:
        transformer_original: 原始 transformer（冻结）
        transformer_unlearned: 去学习后的 transformer（可训练）
        optimizer: 优化器
        loss_fn: 损失函数
        scheduler: 调度器
        tokenizer: 分词器
        text_encoder: 文本编码器
        prompts: prompt 列表
        target_concept: 目标概念（要消除的）
        preserve_concepts: 保留概念列表
        erasers: eraser 字典
        num_frames: 帧数
        latent_channels: latent 通道数
        latent_height: latent 高度
        latent_width: latent 宽度
        num_timesteps: 时间步总数
        device: 设备
    
    Returns:
        损失字典
    """
    batch_size = len(prompts)
    
    # 1. 生成随机噪声 latent（不需要真实视频！）
    x_start = generate_random_latent(
        batch_size=batch_size,
        num_frames=num_frames,
        latent_channels=latent_channels,
        latent_height=latent_height,
        latent_width=latent_width,
        device=device
    )
    
    # 2. 编码文本条件
    # 确定 text_encoder 所在的设备
    if text_encoder_device is None:
        text_encoder_device = device
    elif isinstance(text_encoder_device, str):
        text_encoder_device = torch.device(text_encoder_device)
    
    # 确定 transformer 所在的设备（用于后续移动条件嵌入）
    # 获取 transformer 的第一个参数所在的设备
    try:
        transformer_device = next(transformer_unlearned.parameters()).device
    except StopIteration:
        transformer_device = device
    
    # 检测 text_encoder 类型，确定正确的 max_length
    # CogVideoX 使用 T5EncoderModel，输出维度是 4096，max_length 是 226
    text_encoder_type = type(text_encoder).__name__
    if "CLIP" in text_encoder_type or "Clip" in text_encoder_type:
        # CLIP text_encoder 的最大位置嵌入是 77
        max_seq_length = 77
    elif "T5" in text_encoder_type or "t5" in text_encoder_type or "T5Encoder" in text_encoder_type:
        # T5 text_encoder 可以使用更长的序列（CogVideoX 标准是 226）
        max_seq_length = 226
    else:
        # 默认尝试获取模型配置中的 max_position_embeddings
        try:
            if hasattr(text_encoder, 'config') and hasattr(text_encoder.config, 'max_position_embeddings'):
                max_seq_length = text_encoder.config.max_position_embeddings
            elif hasattr(text_encoder, 'text_model') and hasattr(text_encoder.text_model, 'config'):
                max_seq_length = text_encoder.text_model.config.max_position_embeddings
            else:
                max_seq_length = 226  # CogVideoX 默认使用 T5，所以默认是 226
        except:
            max_seq_length = 226  # CogVideoX 默认使用 T5，所以默认是 226
    
    # 目标概念条件
    target_tokens = tokenizer(
        target_concept,
        return_tensors="pt",
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=True
    ).to(text_encoder_device)
    # T5 text_encoder 输出: (batch_size, seq_len, hidden_dim)
    # 注意：CogVideoX 使用的 T5 可能输出维度是 4096，而不是标准的 512
    text_output = text_encoder(target_tokens.input_ids.to(text_encoder_device))
    cond_target = text_output[0] if isinstance(text_output, tuple) else text_output.last_hidden_state
    # cond_target 形状: (1, seq_len, hidden_dim)
    
    # 检查是否需要 reshape（如果被错误地 flatten 了）
    if len(cond_target.shape) == 2:
        # 如果被 flatten 成了 (batch*seq, hidden_dim)，需要 reshape
        # 使用实际的 input_ids 长度来确定 seq_len
        actual_seq_len = target_tokens.input_ids.shape[1]
        hidden_dim = cond_target.shape[1]
        # 计算实际的 batch_size（应该是 1）
        total_elements = cond_target.shape[0]
        actual_batch = total_elements // actual_seq_len
        if actual_batch * actual_seq_len == total_elements:
            cond_target = cond_target.view(actual_batch, actual_seq_len, hidden_dim)
        else:
            # 如果无法整除，假设是单个样本被 flatten
            cond_target = cond_target.view(1, actual_seq_len, hidden_dim)
    
    # 扩展到 batch_size: (1, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
    if cond_target.shape[0] == 1:
        cond_target = cond_target.repeat(batch_size, 1, 1)
    elif cond_target.shape[0] != batch_size:
        # 如果形状不对，重新扩展
        cond_target = cond_target[:1].repeat(batch_size, 1, 1)
    # 确保条件嵌入在 transformer 所在的设备上
    cond_target = cond_target.to(transformer_device)
    
    # 保留概念条件
    preserved_text = ", ".join(preserve_concepts)
    preserve_tokens = tokenizer(
        preserved_text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=True
    ).to(text_encoder_device)
    text_output = text_encoder(preserve_tokens.input_ids.to(text_encoder_device))
    cond_preserve = text_output[0] if isinstance(text_output, tuple) else text_output.last_hidden_state
    # 检查是否需要 reshape
    if len(cond_preserve.shape) == 2:
        actual_seq_len = preserve_tokens.input_ids.shape[1]
        hidden_dim = cond_preserve.shape[1]
        total_elements = cond_preserve.shape[0]
        actual_batch = total_elements // actual_seq_len
        if actual_batch * actual_seq_len == total_elements:
            cond_preserve = cond_preserve.view(actual_batch, actual_seq_len, hidden_dim)
        else:
            cond_preserve = cond_preserve.view(1, actual_seq_len, hidden_dim)
    if cond_preserve.shape[0] == 1:
        cond_preserve = cond_preserve.repeat(batch_size, 1, 1)
    elif cond_preserve.shape[0] != batch_size:
        cond_preserve = cond_preserve[:1].repeat(batch_size, 1, 1)
    cond_preserve = cond_preserve.to(transformer_device)
    
    # 负提示词条件（可选）
    neg_prompt = f"not {target_concept}"
    neg_tokens = tokenizer(
        neg_prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=True
    ).to(text_encoder_device)
    text_output = text_encoder(neg_tokens.input_ids.to(text_encoder_device))
    cond_neg = text_output[0] if isinstance(text_output, tuple) else text_output.last_hidden_state
    # 检查是否需要 reshape
    if len(cond_neg.shape) == 2:
        actual_seq_len = neg_tokens.input_ids.shape[1]
        hidden_dim = cond_neg.shape[1]
        total_elements = cond_neg.shape[0]
        actual_batch = total_elements // actual_seq_len
        if actual_batch * actual_seq_len == total_elements:
            cond_neg = cond_neg.view(actual_batch, actual_seq_len, hidden_dim)
        else:
            cond_neg = cond_neg.view(1, actual_seq_len, hidden_dim)
    if cond_neg.shape[0] == 1:
        cond_neg = cond_neg.repeat(batch_size, 1, 1)
    elif cond_neg.shape[0] != batch_size:
        cond_neg = cond_neg[:1].repeat(batch_size, 1, 1)
    cond_neg = cond_neg.to(transformer_device)
    
    # 3. 采样时间步（确保数据类型为 long，匹配模型期望）
    t = torch.randint(0, num_timesteps, (batch_size,), device=device, dtype=torch.long)
    
    # 4. 添加噪声（扩散过程）
    noise = torch.randn_like(x_start)
    # 确保 alphas_cumprod 在正确的设备和数据类型上
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=x_start.dtype)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    t_expanded = t.view(-1, 1, 1, 1, 1)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(t_expanded.shape)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(t_expanded.shape)
    
    x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # 5. 提取注意力掩码（mask-based localization）
    # 需要先进行一次前向传播来捕获注意力映射
    # CogVideoXTransformer3D 期望输入形状: (batch_size, num_frames, channels, height, width)
    # 当前 x_t 形状: (batch_size, latent_channels, latent_height, latent_width, num_frames)
    # 需要转换为: (batch_size, num_frames, latent_channels, latent_height, latent_width)
    attn_maps = {}
    with AttnMapsCapture(transformer_unlearned, attn_maps):
        # 调整形状: (B, C, H, W, F) -> (B, F, C, H, W)
        hidden_states = x_t.permute(0, 4, 1, 2, 3).contiguous()
        # 确保 hidden_states 在 transformer 所在的设备上
        hidden_states = hidden_states.to(transformer_device)
        # 确保 timestep 是 long 类型，并在正确的设备上
        t_zero = torch.zeros(batch_size, dtype=torch.long, device=transformer_device)
        _ = transformer_unlearned(
            hidden_states,
            timestep=t_zero,
            encoder_hidden_states=cond_target
        )
    
    # 获取目标概念的 token 索引
    target_token_ids = target_tokens["input_ids"][0]
    word_indices = torch.where(target_token_ids != tokenizer.pad_token_id)[0]
    
    # 提取掩码
    masks = get_mask(
        attn_maps=attn_maps,
        word_indices=word_indices,
        thres=0.5,
        height=latent_height,
        width=latent_width,
        head_num=30,  # 根据模型配置调整
        text_len=cond_target.shape[1]
    )
    mask = masks["average_mask"]  # (B, F, H, W)
    
    # 6. 捕获 eraser 输出（用于 mask-based localization）
    eraser_outputs = {}
    with EraserOutputsCapture(transformer_unlearned, erasers, eraser_outputs):
        # 调整形状: (B, C, H, W, F) -> (B, F, C, H, W)
        hidden_states = x_t.permute(0, 4, 1, 2, 3).contiguous()
        # 确保 hidden_states 和 timestep 在 transformer 所在的设备上
        hidden_states = hidden_states.to(transformer_device)
        t_transformer = t.to(transformer_device)
        _ = transformer_unlearned(
            hidden_states,
            timestep=t_transformer,
            encoder_hidden_states=cond_target
        )
    
    # 7. 计算损失
    # 注意：这里需要将 transformer 包装成可以调用 model(x_t, t, cond) 的形式
    # 实际使用时需要根据模型接口调整
    
    # 为了计算损失，我们需要包装函数来适配 transformer 的接口
    def model_wrapper_unlearned(x_t, t, cond):
        """包装 transformer_unlearned 为损失函数需要的接口"""
        # 调整形状: (B, C, H, W, F) -> (B, F, C, H, W)
        hidden_states = x_t.permute(0, 4, 1, 2, 3).contiguous()
        # 确保所有张量在 transformer 所在的设备上
        hidden_states = hidden_states.to(transformer_device)
        t_device = t.to(transformer_device)
        cond_device = cond.to(transformer_device)
        # 调用 transformer
        output = transformer_unlearned(
            hidden_states,
            timestep=t_device,
            encoder_hidden_states=cond_device
        )
        # transformer 输出形状: (B, F, C, H, W)
        # 需要转换回: (B, C, H, W, F) 以匹配损失函数的期望
        return output.permute(0, 2, 3, 4, 1).contiguous()
    
    def model_wrapper_original(x_t, t, cond):
        """包装 transformer_original 为损失函数需要的接口"""
        # 获取 transformer_original 所在的设备
        try:
            transformer_original_device = next(transformer_original.parameters()).device
        except StopIteration:
            transformer_original_device = transformer_device
        
        # 调整形状: (B, C, H, W, F) -> (B, F, C, H, W)
        hidden_states = x_t.permute(0, 4, 1, 2, 3).contiguous()
        # 确保所有张量在 transformer_original 所在的设备上
        hidden_states = hidden_states.to(transformer_original_device)
        t_device = t.to(transformer_original_device)
        cond_device = cond.to(transformer_original_device)
        # 调用 transformer（原始模型，冻结）
        with torch.no_grad():
            output = transformer_original(
                hidden_states,
                timestep=t_device,
                encoder_hidden_states=cond_device
            )
        # transformer 输出形状: (B, F, C, H, W)
        # 需要转换回: (B, C, H, W, F) 以匹配损失函数的期望
        return output.permute(0, 2, 3, 4, 1).contiguous()
    
    # 计算损失
    total_loss, loss_dict = loss_fn(
        model_original=model_wrapper_original,  # 使用包装函数
        model_unlearned=model_wrapper_unlearned,  # 使用包装函数
        x_t=x_t,
        t=t,
        cond_target=cond_target,
        cond_preserve=cond_preserve,
        cond_neg=cond_neg,
        mask=mask,
        eraser_outputs=eraser_outputs
    )
    
    # 8. 反向传播
    if gradient_accumulation:
        # 梯度累积模式：只反向传播，不更新参数
        total_loss = total_loss / gradient_accumulation_steps  # 归一化损失
        total_loss.backward()
    else:
        # 正常模式：立即更新参数
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # 清理内存
    del x_start, noise, x_t, hidden_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}


def train_unlearning_prompt_only(
    args
):
    """
    主训练函数（仅使用 prompt）
    
    Args:
        args: 命令行参数
    """
    # 设置 PyTorch 内存管理
    if torch.cuda.is_available():
        # 设置可扩展段以避免内存碎片
        import os
        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print(f"已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # 检测可用的 GPU 数量
    if args.device == "cuda":
        num_gpus = torch.cuda.device_count()
        
        # 检测实际可见的 GPU（考虑 CUDA_VISIBLE_DEVICES）
        if num_gpus > 0:
            # 获取第一个可见 GPU 的名称（用于显示）
            gpu_name = torch.cuda.get_device_name(0)
            print(f"检测到 {num_gpus} 个可见 GPU")
            print(f"  主 GPU: {gpu_name} (逻辑设备: cuda:0)")
            if num_gpus > 1:
                for i in range(1, num_gpus):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} (逻辑设备: cuda:{i})")
            
            # 显示 GPU 内存信息
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"  GPU 0 内存: {memory_allocated:.2f} GB 已分配, {memory_reserved:.2f} GB 已保留, {memory_total:.2f} GB 总计")
        
        # 如果指定了多个 GPU，使用模型并行
        if num_gpus > 1 and args.multi_gpu:
            print(f"使用多 GPU 训练（模型并行）")
            device_map = "balanced"  # CogVideoXPipeline 只支持 "balanced"
            device = "cuda:0"  # 主设备用于数据（逻辑设备 0，对应 CUDA_VISIBLE_DEVICES 的第一个）
            print(f"  主设备: {device} (对应 CUDA_VISIBLE_DEVICES 的第一个 GPU)")
        else:
            device_map = None
            device = args.device
    else:
        device_map = None
        device = args.device
        num_gpus = 1
    
    # 1. 加载预训练模型
    print(f"加载预训练模型: {args.model_path}")
    
    # 检查 GPU 是否支持 FP16
    if args.use_fp16:
        if torch.cuda.is_available():
            # 检查 GPU 计算能力（FP16 需要 >= 7.0）
            gpu_capability = torch.cuda.get_device_capability(0)
            if gpu_capability[0] < 7:
                print(f"⚠️  警告：GPU 计算能力 {gpu_capability} < 7.0，可能不支持 FP16")
                print(f"   将使用 FP32 以避免 CUBLAS 错误")
                dtype = torch.float32
            else:
                dtype = torch.float16
                print(f"✓ GPU 支持 FP16（计算能力: {gpu_capability}）")
        else:
            dtype = torch.float32
            print("⚠️  CPU 不支持 FP16，使用 FP32")
    else:
        dtype = torch.float32
    
    # 如果使用多 GPU，提供两种方案
    if device_map:
        # 检查是否有参数强制使用 CPU offload
        use_cpu_offload = getattr(args, 'use_cpu_offload', False)
        
        if use_cpu_offload:
            print(f"⚠️  使用 CPU offload 方案（更稳定，但可能较慢）")
            print(f"   提示：如果想使用真正的多 GPU 并行，请移除 --use_cpu_offload 参数")
        else:
            print(f"尝试使用多 GPU 模型并行（device_map='balanced'）...")
            print(f"   如果遇到错误，可以添加 --use_cpu_offload 使用 CPU offload 方案")
        
        use_cpu_offload = getattr(args, 'use_cpu_offload', False)
        
        if use_cpu_offload:
            # 方案1：使用单 GPU + CPU offload（更稳定）
            print(f"加载 pipeline（transformer 使用 {dtype}，text_encoder 使用 float32）...")
            
            pipe_original = CogVideoXPipeline.from_pretrained(
                args.model_path,
                torch_dtype=dtype
            )
            
            # 确保 text_encoder 使用 FP32（避免 CUBLAS 错误）
            if hasattr(pipe_original, 'text_encoder') and args.use_fp16:
                print("将 text_encoder 转换为 FP32 以避免 CUBLAS 错误...")
                for param in pipe_original.text_encoder.parameters():
                    if param.dtype == torch.float16:
                        param.data = param.data.float()
            
            try:
                pipe_original.enable_model_cpu_offload(gpu_id=0)
                print(f"✓ 已启用 CPU offload（使用逻辑设备 cuda:0）")
            except TypeError:
                try:
                    pipe_original.enable_model_cpu_offload()
                    print(f"✓ 已启用 CPU offload（默认使用 cuda:0）")
                except Exception as e:
                    print(f"⚠️  CPU offload 失败，使用标准加载: {e}")
                    pipe_original = pipe_original.to(device)
            except Exception as e:
                print(f"⚠️  CPU offload 失败，使用标准加载: {e}")
                pipe_original = pipe_original.to(device)
            
            pipe_unlearned = CogVideoXPipeline.from_pretrained(
                args.model_path,
                torch_dtype=dtype
            )
            
            if hasattr(pipe_unlearned, 'text_encoder') and args.use_fp16:
                print("将 text_encoder 转换为 FP32 以避免 CUBLAS 错误...")
                for param in pipe_unlearned.text_encoder.parameters():
                    if param.dtype == torch.float16:
                        param.data = param.data.float()
            
            try:
                pipe_unlearned.enable_model_cpu_offload(gpu_id=0)
                print(f"✓ 已启用 CPU offload（使用逻辑设备 cuda:0）")
            except TypeError:
                try:
                    pipe_unlearned.enable_model_cpu_offload()
                    print(f"✓ 已启用 CPU offload（默认使用 cuda:0）")
                except Exception as e:
                    print(f"⚠️  CPU offload 失败，使用标准加载: {e}")
                    pipe_unlearned = pipe_unlearned.to(device)
            except Exception as e:
                print(f"⚠️  CPU offload 失败，使用标准加载: {e}")
                pipe_unlearned = pipe_unlearned.to(device)
            
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            device_map = None
        else:
            # 方案2：尝试真正的多 GPU 模型并行
            print(f"尝试使用 device_map='balanced' 进行多 GPU 模型并行...")
            print(f"   如果遇到错误，可以添加 --use_cpu_offload 使用 CPU offload 方案")
            
            try:
                # 尝试使用 device_map
                pipe_original = CogVideoXPipeline.from_pretrained(
                    args.model_path,
                    torch_dtype=dtype,
                    device_map=device_map
                )
                
                # 检查并修复 text_encoder 的设备问题
                if hasattr(pipe_original, 'text_encoder'):
                    try:
                        first_param = next(pipe_original.text_encoder.parameters())
                        if first_param.device.type == 'meta' or first_param.device.type == 'cpu':
                            print("⚠️  text_encoder 在 meta/cpu，重新加载到 GPU...")
                            from transformers import T5EncoderModel
                            pipe_original.text_encoder = T5EncoderModel.from_pretrained(
                                args.model_path,
                                subfolder="text_encoder",
                                torch_dtype=torch.float32  # text_encoder 使用 FP32
                            ).to(device)
                    except StopIteration:
                        from transformers import T5EncoderModel
                        pipe_original.text_encoder = T5EncoderModel.from_pretrained(
                            args.model_path,
                            subfolder="text_encoder",
                            torch_dtype=torch.float32
                        ).to(device)
                
                pipe_unlearned = CogVideoXPipeline.from_pretrained(
                    args.model_path,
                    torch_dtype=dtype,
                    device_map=device_map
                )
                
                if hasattr(pipe_unlearned, 'text_encoder'):
                    try:
                        first_param = next(pipe_unlearned.text_encoder.parameters())
                        if first_param.device.type == 'meta' or first_param.device.type == 'cpu':
                            print("⚠️  text_encoder 在 meta/cpu，重新加载到 GPU...")
                            from transformers import T5EncoderModel
                            pipe_unlearned.text_encoder = T5EncoderModel.from_pretrained(
                                args.model_path,
                                subfolder="text_encoder",
                                torch_dtype=torch.float32
                            ).to(device)
                    except StopIteration:
                        from transformers import T5EncoderModel
                        pipe_unlearned.text_encoder = T5EncoderModel.from_pretrained(
                            args.model_path,
                            subfolder="text_encoder",
                            torch_dtype=torch.float32
                        ).to(device)
                
                print(f"✓ 成功使用多 GPU 模型并行（device_map='balanced'）")
            except Exception as e:
                print(f"⚠️  多 GPU 模型并行失败: {e}")
                print(f"   自动切换到 CPU offload 方案...")
                # 回退到 CPU offload
                pipe_original = CogVideoXPipeline.from_pretrained(
                    args.model_path,
                    torch_dtype=dtype
                )
                pipe_original.enable_model_cpu_offload()
                
                pipe_unlearned = CogVideoXPipeline.from_pretrained(
                    args.model_path,
                    torch_dtype=dtype
                )
                pipe_unlearned.enable_model_cpu_offload()
                device_map = None
    else:
        pipe_original = CogVideoXPipeline.from_pretrained(
            args.model_path,
            torch_dtype=dtype
        ).to(device)
        
        pipe_unlearned = CogVideoXPipeline.from_pretrained(
            args.model_path,
            torch_dtype=dtype
        ).to(device)
    
    # 2. 设置 eraser
    print(f"设置 eraser (rank={args.eraser_rank})")
    # 获取 transformer 所在的设备（如果使用 device_map，可能分布在多个设备上）
    if device_map:
        # 对于多 GPU 情况，eraser 会跟随 transformer 的分布
        eraser_device = next(pipe_unlearned.transformer.parameters()).device
    else:
        eraser_device = device
    
    erasers = setup_cogvideo_adapter_eraser(
        model=pipe_unlearned.transformer,
        eraser_rank=args.eraser_rank,
        device=eraser_device,
        dtype=dtype
    )
    
    # 3. 冻结原始模型
    # Pipeline 对象没有 eval() 方法，需要调用内部组件的 eval()
    pipe_original.transformer.eval()
    for param in pipe_original.transformer.parameters():
        param.requires_grad = False
    
    # 4. 冻结 transformer，只训练 eraser
    pipe_unlearned.transformer.eval()
    for param in pipe_unlearned.transformer.parameters():
        param.requires_grad = False
    
    # 只优化 eraser 参数
    for eraser in erasers.values():
        eraser.train()
        for param in eraser.parameters():
            param.requires_grad = True
    
    # 5. 准备数据（仅 prompt，无需真实视频）
    print(f"准备训练数据（仅 prompt）...")
    
    # 从文件加载 prompts（如果提供）
    if args.prompts_file:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        # 使用默认 prompts（目标概念的变体）
        prompts = [
            f"{args.target_concept}",
            f"a scene with {args.target_concept}",
            f"video containing {args.target_concept}",
        ] * args.num_prompts
    
    # 加载或自动生成保留概念
    if args.preserved_concepts_file:
        # 从文件加载
        with open(args.preserved_concepts_file, 'r', encoding='utf-8') as f:
            preserve_concepts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"✓ 从文件加载了 {len(preserve_concepts)} 个保留概念: {args.preserved_concepts_file}")
    elif args.auto_generate_preserved:
        # 自动生成保留概念
        print(f"自动生成保留概念（目标概念: {args.target_concept}）...")
        generator = PreservedConceptsGenerator()
        preserve_concepts = generator.generate(
            target_concept=args.target_concept,
            num_concepts=args.num_preserved_concepts,
            use_llm=args.use_llm_for_preserved,
            include_common=True
        )
        
        # 保存生成的概念到文件（可选）
        if args.save_generated_preserved:
            output_file = args.save_generated_preserved
            generator.save_to_file(
                target_concept=args.target_concept,
                output_file=output_file,
                num_concepts=args.num_preserved_concepts,
                use_llm=args.use_llm_for_preserved,
                include_common=True
            )
        
        print(f"✓ 自动生成了 {len(preserve_concepts)} 个保留概念")
    else:
        # 使用默认保留概念
        preserve_concepts = ["person", "face", "clothing", "background"]
        print(f"⚠ 使用默认保留概念: {preserve_concepts}")
        print(f"  提示: 使用 --auto_generate_preserved 可以自动生成保留概念")
    
    # 创建数据集
    dataset = PromptOnlyDataset(
        prompts=prompts,
        target_concept=args.target_concept,
        preserve_concepts=preserve_concepts,
        num_samples=args.num_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 6. 初始化损失函数
    scheduler = pipe_unlearned.scheduler
    # 对于多 GPU，alphas_cumprod 放在第一个 GPU 上
    if device_map:
        alphas_cumprod = scheduler.alphas_cumprod.to(next(pipe_unlearned.transformer.parameters()).device)
    else:
        alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    loss_fn = T2VUnlearningLoss(
        alphas_cumprod=alphas_cumprod,
        unlearning_weight=args.unlearning_weight,
        preservation_weight=args.preservation_weight,
        localization_weight=args.localization_weight,
        guidance_scale=args.guidance_scale,
        loss_type=args.loss_type
    )
    
    # 7. 初始化优化器（只优化 eraser）
    eraser_params = []
    for eraser in erasers.values():
        eraser_params.extend(list(eraser.parameters()))
    
    optimizer = torch.optim.AdamW(
        eraser_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 8. 训练循环
    print(f"\n开始训练...")
    print(f"  目标概念: {args.target_concept}")
    print(f"  保留概念: {preserve_concepts}")
    print(f"  Prompt 数量: {len(prompts)}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    
    global_step = 0
    accumulated_loss = None
    
    print(f"训练配置:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"  有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    
    # 初始化优化器（清除之前的梯度）
    optimizer.zero_grad()
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # 每个 epoch 开始时清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for batch_idx, batch in enumerate(dataloader):
            prompts_batch = batch["prompt"]
            target_concept = batch["target_concept"][0]
            preserve_concepts_batch = batch["preserve_concepts"][0]
            
            # 训练步骤
            # 对于多 GPU，需要确定数据应该放在哪个设备上
            # text_encoder 所在的设备（用于 token 编码）
            if device_map:
                # 获取 text_encoder 所在的设备
                text_encoder_device = next(pipe_unlearned.text_encoder.parameters()).device
                # 获取 transformer 的第一个参数所在的设备（用于数据）
                data_device = next(pipe_unlearned.transformer.parameters()).device
            else:
                text_encoder_device = device
                data_device = device
            
            # 判断是否是梯度累积的最后一步
            is_accumulation_step = (batch_idx + 1) % args.gradient_accumulation_steps != 0
            is_last_accumulation_step = (batch_idx + 1) % args.gradient_accumulation_steps == 0
            
            loss_dict = train_step_prompt_only(
                transformer_original=pipe_original.transformer,
                transformer_unlearned=pipe_unlearned.transformer,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                tokenizer=pipe_unlearned.tokenizer,
                text_encoder=pipe_unlearned.text_encoder,
                prompts=prompts_batch,
                target_concept=target_concept,
                preserve_concepts=preserve_concepts_batch,
                erasers=erasers,
                num_frames=args.num_frames,
                latent_channels=args.latent_channels,
                latent_height=args.latent_height,
                latent_width=args.latent_width,
                num_timesteps=args.num_timesteps,
                device=data_device,
                text_encoder_device=text_encoder_device,
                gradient_accumulation=is_accumulation_step,
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
            
            # 累积损失用于打印
            if accumulated_loss is None:
                accumulated_loss = {k: 0.0 for k in loss_dict.keys()}
            for k, v in loss_dict.items():
                accumulated_loss[k] += v
            
            # 如果是梯度累积的最后一步，更新参数
            if is_last_accumulation_step or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                accumulated_loss = None
                
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            global_step += 1
            
            # 打印损失（只在梯度累积完成后打印）
            if not is_accumulation_step and global_step % args.log_interval == 0:
                print(f"  Step {global_step}: "
                      f"Total={loss_dict.get('total_loss', 0):.4f}, "
                      f"Unlearn={loss_dict.get('unlearning_loss_weighted', 0):.4f}, "
                      f"Preserve={loss_dict.get('preservation_loss_weighted', 0):.4f}, "
                      f"Local={loss_dict.get('localization_loss_weighted', 0):.4f}")
            
            # 保存 checkpoint
            if global_step % args.save_interval == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_cogvideo_eraser_from_transformer(
                    checkpoint_dir,
                    pipe_unlearned.transformer
                )
                print(f"  Saved checkpoint to {checkpoint_dir}")
                
                # 保存后清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # 9. 保存最终模型
    print(f"\n保存最终模型...")
    output_dir = os.path.join(
        args.output_dir,
        f"{args.model_name}_{args.target_concept}_eraser"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    save_cogvideo_eraser_from_transformer(output_dir, pipe_unlearned.transformer)
    
    print(f"训练完成！Eraser 已保存到: {output_dir}")
    print(f"  - eraser_config.json")
    print(f"  - eraser_weights.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2VUnlearning 训练（仅使用 Prompt）")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--model_name", type=str, default="cogvideox", help="模型名称")
    parser.add_argument("--eraser_rank", type=int, default=128, help="Eraser rank")
    parser.add_argument("--use_fp16", action="store_true", help="使用 FP16")
    
    # 训练参数
    parser.add_argument("--target_concept", type=str, required=True, help="目标概念（要消除的，如 'nudity'）")
    parser.add_argument("--preserved_concepts_file", type=str, default=None, help="保留概念文件路径")
    parser.add_argument("--auto_generate_preserved", action="store_true", help="自动生成保留概念（推荐）")
    parser.add_argument("--num_preserved_concepts", type=int, default=15, help="自动生成的保留概念数量")
    parser.add_argument("--use_llm_for_preserved", action="store_true", help="使用LLM生成保留概念（如果可用）")
    parser.add_argument("--save_generated_preserved", type=str, default=None, help="保存自动生成的保留概念到文件")
    parser.add_argument("--prompts_file", type=str, default=None, help="Prompt 文件路径（可选）")
    parser.add_argument("--num_prompts", type=int, default=100, help="如果未提供 prompts_file，生成多少个 prompt")
    parser.add_argument("--num_samples", type=int, default=None, help="数据集大小（如果提供，会重复）")
    
    # 数据参数
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小（内存不足时建议设为1）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数（实际batch_size = batch_size * gradient_accumulation_steps）")
    parser.add_argument("--num_frames", type=int, default=49, help="帧数")
    parser.add_argument("--latent_channels", type=int, default=16, help="Latent 通道数")
    parser.add_argument("--latent_height", type=int, default=10, help="Latent 高度")
    parser.add_argument("--latent_width", type=int, default=10, help="Latent 宽度")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="时间步总数")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # 损失参数
    parser.add_argument("--unlearning_weight", type=float, default=1.0, help="去学习损失权重")
    parser.add_argument("--preservation_weight", type=float, default=0.5, help="保留损失权重")
    parser.add_argument("--localization_weight", type=float, default=1.0, help="Localization 损失权重")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="引导强度")
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "l1"], help="损失类型")
    
    # 优化器参数
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="保存 checkpoint 间隔")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--multi_gpu", action="store_true", help="启用多 GPU 训练（尝试使用 device_map 进行模型并行）")
    parser.add_argument("--use_cpu_offload", action="store_true", help="强制使用 CPU offload 而不是多 GPU 并行（更稳定但可能较慢）")
    
    args = parser.parse_args()
    
    # 自动检测多 GPU
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        if not args.multi_gpu:
            print(f"检测到 {torch.cuda.device_count()} 个 GPU，但未启用 --multi_gpu")
            print(f"提示: 添加 --multi_gpu 参数可以使用多 GPU 训练")
        else:
            print(f"已启用多 GPU 训练，将使用 {torch.cuda.device_count()} 个 GPU")
    
    train_unlearning_prompt_only(args)

