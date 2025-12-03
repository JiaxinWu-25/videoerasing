"""
T2VUnlearning 训练示例

展示如何使用负引导速度损失进行去学习训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List
import argparse
import os
import json
from diffusers import CogVideoXPipeline
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from .unlearning_losses import T2VUnlearningLoss, NegativelyGuidedVelocityLoss, ConceptPreservationLoss
from .concept_reg_cogvideo import AttnMapsCapture, get_mask, EraserOutputsCapture
from .erasers.cogvideo_erasers import CogVideoXWithEraser, setup_cogvideo_adapter_eraser, save_cogvideo_eraser_from_transformer
from .erasers.utils import AdapterEraser


def load_preserved_concepts(file_path: str) -> List[str]:
    """
    从文件中加载保留概念列表
    
    Args:
        file_path: 文件路径，每行一个概念
        
    Returns:
        保留概念列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preserved concepts file not found: {file_path}")
    
    preserved_concepts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            concept = line.strip()
            if concept and not concept.startswith('#'):  # 忽略空行和注释
                preserved_concepts.append(concept)
    
    if not preserved_concepts:
        raise ValueError(f"No preserved concepts found in file: {file_path}")
    
    return preserved_concepts


def train_unlearning_step(
    model_original: nn.Module,
    model_unlearned: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: T2VUnlearningLoss,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond_target: torch.Tensor,
    cond_preserve: Optional[torch.Tensor] = None,
    cond_neg: Optional[torch.Tensor] = None,
    uncond: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    eraser_outputs: Optional[dict] = None,
    device: str = "cuda"
):
    """
    执行一个去学习训练步骤
    
    Args:
        model_original: 原始模型（冻结）
        model_unlearned: 去学习后的模型（可训练）
        optimizer: 优化器
        loss_fn: 损失函数
        x_t: 噪声样本 (B, C, H, W, F)
        t: 时间步 (B,)
        cond_target: 目标概念的条件嵌入（要去除的概念，如 "nudity"）
        cond_preserve: 保留概念的条件嵌入（如 "person"）
        cond_neg: 负提示词的条件嵌入（可选）
        uncond: 无条件嵌入（可选）
        mask: 空间掩码（可选）
        eraser_outputs: eraser 输出字典（可选）
        device: 设备
    """
    # 前向传播计算损失
    total_loss, loss_dict = loss_fn(
        model_original=model_original,
        model_unlearned=model_unlearned,
        x_t=x_t,
        t=t,
        cond_target=cond_target,
        cond_preserve=cond_preserve,
        cond_neg=cond_neg,
        uncond=uncond,
        mask=mask,
        eraser_outputs=eraser_outputs
    )
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_dict


def train_with_mask_localization(
    pipe_original,
    pipe_unlearned,
    optimizer: torch.optim.Optimizer,
    loss_fn: T2VUnlearningLoss,
    dataloader: DataLoader,
    tokenizer,
    text_encoder,
    target_concept: str,
    preserved_concepts: List[str],
    erasers: dict,
    scheduler,
    num_timesteps: int = 1000,
    device: str = "cuda"
):
    """
    使用 mask-based localization 进行训练
    
    Args:
        pipe_original: 原始 pipeline（冻结）
        pipe_unlearned: 去学习后的 pipeline（可训练）
        optimizer: 优化器
        loss_fn: 损失函数
        dataloader: 数据加载器
        tokenizer: 文本分词器
        text_encoder: 文本编码器
        target_concept: 目标概念（要去除的，如 "nudity"）
        preserved_concepts: 保留概念列表（如 ["person", "face"]）
        erasers: eraser 字典
        scheduler: 调度器
        num_timesteps: 时间步总数
        device: 设备
    """
    transformer_original = pipe_original.transformer
    transformer_unlearned = pipe_unlearned.transformer
    
    transformer_unlearned.train()
    transformer_original.eval()
    
    # 获取 alphas_cumprod
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    for batch_idx, batch in enumerate(dataloader):
        # 1. 准备数据 - 假设数据加载器返回视频的 latent
        # 实际使用时需要根据数据格式调整
        if "latent" in batch:
            x_start = batch["latent"].to(device)  # (B, C, H, W, F)
        elif "video" in batch:
            # 如果输入是视频，需要通过 VAE 编码
            video = batch["video"].to(device)  # (B, F, C, H, W)
            with torch.no_grad():
                # 编码视频到 latent space
                # 这里需要根据实际的 VAE 接口调整
                x_start = pipe_unlearned.vae.encode(video).latent_dist.sample()
                x_start = x_start * pipe_unlearned.vae.config.scaling_factor
        else:
            raise ValueError("Batch must contain 'latent' or 'video' key")
        
        prompts = batch.get("prompt", [target_concept] * x_start.shape[0])
        
        # 2. 编码文本条件
        # 目标概念条件
        target_tokens = tokenizer(target_concept, return_tensors="pt", padding=True, truncation=True).to(device)
        cond_target = text_encoder(**target_tokens).last_hidden_state
        
        # 保留概念条件（支持多个概念）
        preserved_text = ", ".join(preserved_concepts)
        preserve_tokens = tokenizer(preserved_text, return_tensors="pt", padding=True, truncation=True).to(device)
        cond_preserve = text_encoder(**preserve_tokens).last_hidden_state
        
        # 负提示词条件（可选）
        neg_prompt = f"not {target_concept}"
        neg_tokens = tokenizer(neg_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        cond_neg = text_encoder(**neg_tokens).last_hidden_state
        
        # 3. 提取注意力掩码（mask-based localization）
        attn_maps = {}
        with AttnMapsCapture(transformer_unlearned, attn_maps):
            # 进行一次前向传播以捕获注意力映射
            t_zero = torch.zeros(x_start.shape[0], dtype=torch.long, device=device)
            # 调用 transformer 的前向传播
            hidden_states = x_start.view(x_start.shape[0], -1, x_start.shape[1])  # 调整形状
            _ = transformer_unlearned(hidden_states, timestep=t_zero, encoder_hidden_states=cond_target)
        
        # 获取目标概念的 token 索引
        target_token_ids = target_tokens["input_ids"][0]
        word_indices = torch.where(target_token_ids != tokenizer.pad_token_id)[0]
        
        # 提取掩码 - 需要根据实际的 latent 尺寸调整
        # 假设 latent 形状为 (B, C, H, W, F)
        latent_h = x_start.shape[2] if len(x_start.shape) > 2 else 10
        latent_w = x_start.shape[3] if len(x_start.shape) > 3 else 10
        text_len = cond_target.shape[1]
        
        masks = get_mask(
            attn_maps=attn_maps,
            word_indices=word_indices,
            thres=0.5,  # 阈值
            height=latent_h,
            width=latent_w,
            head_num=30,  # 根据模型配置调整
            text_len=text_len
        )
        mask = masks["average_mask"]  # 使用平均掩码
        
        # 4. 采样时间步
        t = torch.randint(0, num_timesteps, (x_start.shape[0],), device=device)
        
        # 5. 添加噪声
        noise = torch.randn_like(x_start)
        t_expanded = t.view(-1, 1, 1, 1, 1) if len(x_start.shape) == 5 else t.view(-1, 1, 1, 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(t_expanded.shape)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(t_expanded.shape)
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        # 6. 捕获 eraser 输出（用于 mask-based localization）
        eraser_outputs = {}
        with EraserOutputsCapture(transformer_unlearned, erasers, eraser_outputs):
            # 前向传播以捕获 eraser 输出
            hidden_states = x_t.view(x_t.shape[0], -1, x_t.shape[1]) if len(x_t.shape) == 5 else x_t
            _ = transformer_unlearned(hidden_states, timestep=t, encoder_hidden_states=cond_target)
        
        # 7. 训练步骤
        loss, loss_dict = train_unlearning_step(
            model_original=transformer_original,
            model_unlearned=transformer_unlearned,
            optimizer=optimizer,
            loss_fn=loss_fn,
            x_t=x_t,
            t=t,
            cond_target=cond_target,
            cond_preserve=cond_preserve,
            cond_neg=cond_neg,
            mask=mask,
            eraser_outputs=eraser_outputs,
            device=device
        )
        
        # 8. 打印损失
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss:.4f}")
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.item():.4f}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="T2VUnlearning 训练脚本")
    
    # 超参数
    parser.add_argument(
        "--unlearning_weight",
        type=float,
        default=1.0,
        help="去学习损失的权重 (默认: 1.0)"
    )
    parser.add_argument(
        "--preservation_weight",
        type=float,
        default=0.5,
        help="保留损失的权重 (默认: 0.5)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="引导强度，控制负引导的程度 (默认: 7.5)"
    )
    
    # 概念相关参数
    parser.add_argument(
        "--target_concept",
        type=str,
        required=True,
        help="目标概念（要去除的概念，如 'nudity'）"
    )
    parser.add_argument(
        "--preserved_concepts_file",
        type=str,
        required=True,
        help="保留概念文件路径，每行一个概念"
    )
    
    # 模型相关参数
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX-5b",
        help="预训练模型路径 (默认: THUDM/CogVideoX-5b)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cogvideox5b",
        help="模型名称，用于生成输出文件夹名 (默认: cogvideox5b)"
    )
    parser.add_argument(
        "--eraser_rank",
        type=int,
        default=128,
        help="Eraser rank (默认: 128)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="输出目录 (默认: ./)"
    )
    
    # 训练相关参数
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="训练轮数 (默认: 10)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学习率 (默认: 1e-4)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批次大小 (默认: 1)"
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="时间步总数 (默认: 1000)"
    )
    
    # 其他参数
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1", "l2"],
        help="损失类型 (默认: l2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (默认: cuda)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="数据类型 (默认: bfloat16)"
    )
    
    return parser.parse_args()


# 使用示例
if __name__ == "__main__":
    args = parse_args()
    
    # 1. 准备设备
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    # 2. 加载保留概念列表
    preserved_concepts = load_preserved_concepts(args.preserved_concepts_file)
    print(f"加载了 {len(preserved_concepts)} 个保留概念: {preserved_concepts}")
    
    # 3. 加载模型
    print(f"加载模型: {args.model_path}")
    pipe_original = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe_unlearned = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    
    pipe_original.to(device)
    pipe_unlearned.to(device)
    
    # 4. 设置 eraser
    print(f"设置 eraser (rank={args.eraser_rank})")
    erasers = setup_cogvideo_adapter_eraser(
        pipe_unlearned.transformer,
        eraser_rank=args.eraser_rank,
        device=device,
        dtype=dtype
    )
    print(f"创建了 {len(erasers)} 个 eraser layers")
    
    # 5. 冻结原始模型，只训练 eraser
    for param in pipe_original.transformer.parameters():
        param.requires_grad = False
    for param in pipe_unlearned.transformer.parameters():
        param.requires_grad = False
    # 只优化 eraser 参数
    for eraser in erasers.values():
        for param in eraser.parameters():
            param.requires_grad = True
    
    # 6. 获取调度器和 alphas_cumprod
    scheduler = pipe_unlearned.scheduler
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    # 7. 初始化损失函数
    loss_fn = T2VUnlearningLoss(
        alphas_cumprod=alphas_cumprod,
        unlearning_weight=args.unlearning_weight,
        preservation_weight=args.preservation_weight,
        guidance_scale=args.guidance_scale,
        loss_type=args.loss_type
    )
    
    # 8. 初始化优化器（只优化 eraser 参数）
    eraser_params = []
    for eraser in erasers.values():
        eraser_params.extend(list(eraser.parameters()))
    optimizer = torch.optim.AdamW(eraser_params, lr=args.learning_rate)
    
    # 9. 准备数据加载器
    # 注意：这里需要根据实际的数据格式实现数据加载器
    # 示例数据结构：{"latent": tensor, "prompt": str} 或 {"video": tensor, "prompt": str}
    print("警告: 请确保数据加载器已正确配置")
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 10. 训练循环
    print("开始训练...")
    print(f"超参数配置:")
    print(f"  unlearning_weight: {args.unlearning_weight}")
    print(f"  preservation_weight: {args.preservation_weight}")
    print(f"  guidance_scale: {args.guidance_scale}")
    print(f"  target_concept: {args.target_concept}")
    print(f"  preserved_concepts: {preserved_concepts}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  num_epochs: {args.num_epochs}")
    
    # 注意：实际训练需要取消注释以下代码并提供数据加载器
    # for epoch in range(args.num_epochs):
    #     print(f"\nEpoch {epoch+1}/{args.num_epochs}")
    #     train_with_mask_localization(
    #         pipe_original=pipe_original,
    #         pipe_unlearned=pipe_unlearned,
    #         optimizer=optimizer,
    #         loss_fn=loss_fn,
    #         dataloader=dataloader,
    #         tokenizer=pipe_unlearned.tokenizer,
    #         text_encoder=pipe_unlearned.text_encoder,
    #         target_concept=args.target_concept,
    #         preserved_concepts=preserved_concepts,
    #         erasers=erasers,
    #         scheduler=scheduler,
    #         num_timesteps=args.num_timesteps,
    #         device=device
    #     )
    
    # 11. 保存 eraser
    # 生成输出文件夹名：模型名_消去的概念_eraser
    output_folder_name = f"{args.model_name}_{args.target_concept}_eraser"
    output_path = os.path.join(args.output_dir, output_folder_name)
    
    print(f"\n保存 eraser 到: {output_path}")
    save_cogvideo_eraser_from_transformer(output_path, pipe_unlearned.transformer)
    
    print(f"训练完成！Eraser 已保存到: {output_path}")
    print(f"  - eraser_config.json")
    print(f"  - eraser_weights.pt")

    