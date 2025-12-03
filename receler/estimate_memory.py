"""
显存需求估算工具

用于提前计算模型训练/推理所需的显存，避免 OOM 错误。
"""

import torch
import os
from typing import Dict, Optional, Tuple
from diffusers import CogVideoXPipeline


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())


def estimate_model_memory(
    model_path: str,
    dtype: torch.dtype = torch.float32,
    include_vae: bool = False
) -> Dict[str, float]:
    """
    估算模型各组件占用的显存（仅参数，不包括激活值）
    
    Args:
        model_path: 模型路径
        dtype: 数据类型（float32=4字节, float16=2字节）
        include_vae: 是否包含 VAE（训练时通常不需要）
    
    Returns:
        包含各组件显存需求的字典（单位：GB）
    """
    bytes_per_param = 4 if dtype == torch.float32 else 2
    
    print(f"正在加载模型以估算显存需求...")
    print(f"  模型路径: {model_path}")
    print(f"  数据类型: {dtype} ({bytes_per_param} 字节/参数)")
    
    try:
        # 只加载配置，不加载完整模型到 GPU
        pipe = CogVideoXPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype
        )
        
        memory_dict = {}
        
        # 1. Transformer 参数
        if hasattr(pipe, 'transformer'):
            transformer_params = count_parameters(pipe.transformer)
            transformer_memory_gb = transformer_params * bytes_per_param / (1024**3)
            memory_dict['transformer'] = transformer_memory_gb
            print(f"  Transformer: {transformer_params/1e9:.2f}B 参数 = {transformer_memory_gb:.2f} GB")
        
        # 2. Text Encoder 参数
        if hasattr(pipe, 'text_encoder'):
            text_encoder_params = count_parameters(pipe.text_encoder)
            text_encoder_memory_gb = text_encoder_params * bytes_per_param / (1024**3)
            memory_dict['text_encoder'] = text_encoder_memory_gb
            print(f"  Text Encoder: {text_encoder_params/1e9:.2f}B 参数 = {text_encoder_memory_gb:.2f} GB")
        
        # 3. VAE 参数（训练时通常不需要）
        if include_vae and hasattr(pipe, 'vae'):
            vae_params = count_parameters(pipe.vae)
            vae_memory_gb = vae_params * bytes_per_param / (1024**3)
            memory_dict['vae'] = vae_memory_gb
            print(f"  VAE: {vae_params/1e9:.2f}B 参数 = {vae_memory_gb:.2f} GB")
        
        # 4. 总参数显存
        total_params = sum([
            memory_dict.get('transformer', 0),
            memory_dict.get('text_encoder', 0),
            memory_dict.get('vae', 0)
        ])
        memory_dict['total_model'] = total_params
        
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return memory_dict
        
    except Exception as e:
        print(f"⚠️  加载模型失败: {e}")
        print(f"   使用默认估算值...")
        # 返回默认估算（基于 CogVideoX-5b）
        return {
            'transformer': 10.0,  # ~5B 参数 * 2 bytes (FP16)
            'text_encoder': 0.5,  # T5-base
            'total_model': 10.5
        }


def estimate_training_memory(
    model_path: str,
    batch_size: int = 1,
    num_frames: int = 17,
    latent_height: int = 32,
    latent_width: int = 32,
    latent_channels: int = 16,
    dtype: torch.dtype = torch.float16,
    include_optimizer: bool = True,
    include_gradients: bool = True,
    include_activations: bool = True,
    include_vae: bool = False
) -> Dict[str, float]:
    """
    估算训练时的总显存需求
    
    Args:
        model_path: 模型路径
        batch_size: Batch 大小
        num_frames: 视频帧数
        latent_height: Latent 高度
        latent_width: Latent 宽度
        latent_channels: Latent 通道数
        dtype: 数据类型
        include_optimizer: 是否包含优化器状态（AdamW 需要 2倍参数）
        include_gradients: 是否包含梯度（与参数相同大小）
        include_activations: 是否包含激活值估算
        include_vae: 是否包含 VAE
    
    Returns:
        包含各组件显存需求的字典（单位：GB）
    """
    bytes_per_param = 4 if dtype == torch.float32 else 2
    
    print("=" * 60)
    print("显存需求估算")
    print("=" * 60)
    
    # 1. 模型参数显存
    model_memory = estimate_model_memory(model_path, dtype, include_vae)
    
    memory_breakdown = {
        'model_parameters': model_memory.get('total_model', 0)
    }
    
    # 2. 梯度显存（训练时需要）
    if include_gradients:
        gradient_memory = model_memory.get('total_model', 0)
        memory_breakdown['gradients'] = gradient_memory
        print(f"\n梯度显存: {gradient_memory:.2f} GB")
    
    # 3. 优化器状态（AdamW: momentum + variance = 2倍参数）
    if include_optimizer:
        optimizer_memory = model_memory.get('total_model', 0) * 2
        memory_breakdown['optimizer_states'] = optimizer_memory
        print(f"优化器状态 (AdamW): {optimizer_memory:.2f} GB")
    
    # 4. Batch 数据显存
    # Latent: (batch_size, latent_channels, latent_height, latent_width, num_frames)
    latent_size = batch_size * latent_channels * latent_height * latent_width * num_frames
    latent_memory_gb = latent_size * bytes_per_param / (1024**3)
    memory_breakdown['batch_latents'] = latent_memory_gb
    print(f"\nBatch Latents ({batch_size}x{num_frames}): {latent_memory_gb:.2f} GB")
    
    # 5. Text Encoder 输出（条件嵌入）
    # 假设 text_encoder 输出 shape: (batch_size, seq_len, hidden_dim)
    # T5-base: hidden_dim=768, 假设 seq_len=77
    text_encoder_output_size = batch_size * 77 * 768
    text_encoder_output_memory_gb = text_encoder_output_size * bytes_per_param / (1024**3)
    memory_breakdown['text_encoder_outputs'] = text_encoder_output_memory_gb
    print(f"Text Encoder 输出: {text_encoder_output_memory_gb:.4f} GB")
    
    # 6. 激活值估算（粗略估算）
    if include_activations:
        # Transformer 激活值：通常为参数大小的 0.5-2倍，取决于序列长度和 batch size
        # 这里使用保守估算：batch_size * num_frames * latent_size 的激活值
        activation_size = batch_size * num_frames * latent_height * latent_width * latent_channels
        # 假设每个位置需要存储中间激活值（粗略估算为 latent 大小的 2-5倍）
        activation_multiplier = 3.0  # 保守估算
        activation_memory_gb = activation_size * bytes_per_param * activation_multiplier / (1024**3)
        memory_breakdown['activations'] = activation_memory_gb
        print(f"激活值（估算）: {activation_memory_gb:.2f} GB")
    
    # 7. 总显存需求
    total_memory = sum(memory_breakdown.values())
    memory_breakdown['total'] = total_memory
    
    # 8. 如果加载两个模型（original + unlearned）
    if include_gradients:  # 训练时需要两个模型
        # original 模型（冻结，不需要梯度）
        original_model_memory = model_memory.get('total_model', 0)
        memory_breakdown['original_model'] = original_model_memory
        # unlearned 模型（需要梯度）
        unlearned_model_memory = model_memory.get('total_model', 0)
        memory_breakdown['unlearned_model'] = unlearned_model_memory
        
        # 重新计算总显存
        total_memory = sum(memory_breakdown.values())
        memory_breakdown['total'] = total_memory
    
    print("\n" + "=" * 60)
    print("显存需求汇总")
    print("=" * 60)
    for key, value in memory_breakdown.items():
        if key != 'total':
            print(f"  {key:25s}: {value:8.2f} GB")
    print("-" * 60)
    print(f"  总显存需求: {total_memory:.2f} GB")
    print("=" * 60)
    
    return memory_breakdown


def check_gpu_memory(device_id: int = 0) -> Tuple[float, float]:
    """
    检查 GPU 可用显存
    
    Returns:
        (总显存, 可用显存) 单位：GB
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    allocated_memory = torch.cuda.memory_allocated(device_id) / (1024**3)
    reserved_memory = torch.cuda.memory_reserved(device_id) / (1024**3)
    free_memory = total_memory - reserved_memory
    
    print(f"\nGPU {device_id} 显存状态:")
    print(f"  总显存: {total_memory:.2f} GB")
    print(f"  已分配: {allocated_memory:.2f} GB")
    print(f"  已保留: {reserved_memory:.2f} GB")
    print(f"  可用显存: {free_memory:.2f} GB")
    
    return (total_memory, free_memory)


def recommend_config(
    model_path: str,
    target_memory_gb: Optional[float] = None,
    device_id: int = 0
) -> Dict[str, any]:
    """
    根据可用显存推荐训练配置
    
    Args:
        model_path: 模型路径
        target_memory_gb: 目标显存限制（GB），如果为 None 则使用当前 GPU 可用显存
        device_id: GPU 设备 ID
    
    Returns:
        推荐的配置字典
    """
    if target_memory_gb is None:
        _, free_memory = check_gpu_memory(device_id)
        target_memory_gb = free_memory * 0.9  # 保留 10% 缓冲
    
    print(f"\n根据 {target_memory_gb:.2f} GB 显存限制推荐配置...")
    
    # 估算模型基础显存
    model_memory = estimate_model_memory(model_path, dtype=torch.float16)
    base_model_memory = model_memory.get('total_model', 0)
    
    # 两个模型（original + unlearned）
    two_models_memory = base_model_memory * 2
    
    # 梯度 + 优化器
    training_overhead = base_model_memory * 3  # 梯度 + 优化器状态
    
    # 剩余显存用于 batch 和激活值
    available_for_batch = target_memory_gb - two_models_memory - training_overhead
    
    if available_for_batch < 0:
        print(f"⚠️  警告：显存不足！")
        print(f"   模型基础显存需求: {two_models_memory + training_overhead:.2f} GB")
        print(f"   建议使用 CPU offload 或减少 batch_size")
        return {
            'use_cpu_offload': True,
            'batch_size': 1,
            'dtype': 'float16'
        }
    
    # 估算合适的 batch_size
    # 假设每个样本需要 ~0.5 GB（latent + 激活值）
    estimated_batch_memory_per_sample = 0.5
    recommended_batch_size = max(1, int(available_for_batch / estimated_batch_memory_per_sample))
    
    recommendations = {
        'use_cpu_offload': False,
        'batch_size': recommended_batch_size,
        'dtype': 'float16',
        'estimated_memory_usage': two_models_memory + training_overhead + recommended_batch_size * estimated_batch_memory_per_sample
    }
    
    print(f"\n推荐配置:")
    print(f"  Batch Size: {recommended_batch_size}")
    print(f"  数据类型: {recommendations['dtype']}")
    print(f"  使用 CPU Offload: {recommendations['use_cpu_offload']}")
    print(f"  估算显存使用: {recommendations['estimated_memory_usage']:.2f} GB")
    
    return recommendations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="估算模型训练显存需求")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch 大小")
    parser.add_argument("--num_frames", type=int, default=17, help="视频帧数")
    parser.add_argument("--latent_height", type=int, default=32, help="Latent 高度")
    parser.add_argument("--latent_width", type=int, default=32, help="Latent 宽度")
    parser.add_argument("--latent_channels", type=int, default=16, help="Latent 通道数")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="数据类型")
    parser.add_argument("--device_id", type=int, default=0, help="GPU 设备 ID")
    parser.add_argument("--recommend", action="store_true", help="根据可用显存推荐配置")
    
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # 检查 GPU 显存
    check_gpu_memory(args.device_id)
    
    if args.recommend:
        # 推荐配置
        recommend_config(args.model_path, device_id=args.device_id)
    else:
        # 详细估算
        estimate_training_memory(
            model_path=args.model_path,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            latent_height=args.latent_height,
            latent_width=args.latent_width,
            latent_channels=args.latent_channels,
            dtype=dtype
        )

