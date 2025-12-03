"""
负引导速度预测损失函数 (Negatively-guided Velocity Prediction Loss)

基于 Bayes 定理和 score → velocity 的等价关系，实现去学习损失。
目标：在扩散时刻降低生成含目标概念 c 的概率。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


def extract_into_tensor(a, t, x_shape):
    """
    从张量 a 中提取对应时间步 t 的值
    
    Args:
        a: 时间步相关的系数张量 (num_timesteps,)
        t: 时间步索引 (batch_size,)
        x_shape: 目标张量的形状
    
    Returns:
        提取的值，形状为 (batch_size, 1, 1, ...) 以匹配 x_shape
    """
    b = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)


class NegativelyGuidedVelocityLoss(nn.Module):
    """
    负引导速度预测损失
    
    基于 Bayes 定理和 score → velocity 的等价关系：
    - 目标：降低 P(x_t | c)，即增加 P(x_t | ¬c)
    - 通过负引导 velocity 来实现去学习
    """
    
    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        guidance_scale: float = 7,
        loss_type: str = "l2",
        reduction: str = "mean"
    ):
        """
        Args:
            alphas_cumprod: 累积 alpha 值 (num_timesteps,)
            guidance_scale: 引导强度，控制负引导的程度
            loss_type: 损失类型，"l2" 或 "l1"
            reduction: 约简方式，"mean" 或 "none"
        """
        super().__init__()
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.guidance_scale = guidance_scale
        self.loss_type = loss_type
        self.reduction = reduction
        
    def compute_negative_guided_velocity(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_c: torch.Tensor,
        cond_neg: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算负引导 velocity
        
        基于 Bayes 定理：
        P(x_t | c) → 降低为 P(x_t | ¬c)
        
        使用 classifier-free guidance 的方式：
        v_negative = v_uncond + guidance_scale * (v_uncond - v_c)
        
        Args:
            model: 扩散模型
            x_t: 噪声样本 (B, C, H, W, ...)
            t: 时间步 (B,)
            cond_c: 目标概念的条件嵌入 (B, seq_len, dim)
            cond_neg: 负提示词的条件嵌入，如果提供则使用
            uncond: 无条件（空条件）嵌入，如果提供则使用
        
        Returns:
            负引导 velocity (B, C, H, W, ...)
        """
        # 1. 计算原模型在目标概念 c 上的 velocity
        v_c = model(x_t, t, cond_c)
        
        # 2. 计算无条件 velocity（降低概念 c 的概率）
        if uncond is not None:
            v_uncond = model(x_t, t, uncond)
        elif cond_neg is not None:
            # 使用负提示词
            v_uncond = model(x_t, t, cond_neg)
        else:
            # 使用空条件（None embedding）
            # 注意：这里假设模型支持 None 条件，实际使用时需要根据模型调整
            # 如果模型不支持 None，可以使用空字符串的条件嵌入
            try:
                v_uncond = model(x_t, t, None)
            except (TypeError, ValueError):
                # 如果模型不支持 None，使用零向量作为无条件嵌入
                # 需要根据实际模型的条件嵌入维度调整
                batch_size = x_t.shape[0]
                # 创建一个与 cond_c 形状相同的零向量
                uncond_shape = list(cond_c.shape)
                uncond_zeros = torch.zeros(uncond_shape, device=x_t.device, dtype=cond_c.dtype)
                v_uncond = model(x_t, t, uncond_zeros)
        
        # 3. 计算负引导 velocity
        # 通过引导使模型远离目标概念
        # v_negative = v_uncond + guidance_scale * (v_uncond - v_c)
        # 这相当于：v_negative = (1 + guidance_scale) * v_uncond - guidance_scale * v_c
        v_negative = v_uncond + self.guidance_scale * (v_uncond - v_c)
        
        return v_negative
    
    def forward(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_c: torch.Tensor,
        cond_neg: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算负引导速度损失
        
        Args:
            model: 扩散模型（需要去学习的模型）
            x_t: 噪声样本 (B, C, H, W, ...)
            t: 时间步 (B,)
            cond_c: 目标概念的条件嵌入 (B, seq_len, dim)
            cond_neg: 负提示词的条件嵌入（可选）
            uncond: 无条件嵌入（可选）
            mask: 空间掩码，用于 mask-based localization (B, H, W, ...)
        
        Returns:
            loss: 损失值
            loss_dict: 包含详细损失的字典
        """
        # 1. 模型在当前参数下预测的 velocity
        v_predicted = model(x_t, t, cond_c)
        
        # 2. 计算负引导 velocity（目标 velocity）
        v_negative = self.compute_negative_guided_velocity(
            model, x_t, t, cond_c, cond_neg, uncond
        )
        
        # 3. 计算损失：使预测的 velocity 接近负引导 velocity
        if self.loss_type == "l2":
            loss = F.mse_loss(v_predicted, v_negative, reduction='none')
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_predicted, v_negative, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 4. 应用空间掩码（mask-based localization）
        if mask is not None:
            # 确保 mask 的形状匹配
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(1)
            # 只在目标概念出现的区域计算损失
            # mask: 1 表示目标概念区域，0 表示背景区域
            # 我们可以在目标区域计算损失，或者在整个区域计算但加权
            loss = loss * mask
        
        # 5. 计算平均损失
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # "none" 时保持原样
        
        # 6. 构建损失字典
        loss_dict = {
            "unlearning_loss": loss.clone().detach(),
            "v_pred_norm": v_predicted.norm().detach(),
            "v_neg_norm": v_negative.norm().detach(),
        }
        
        return loss, loss_dict


class ConceptPreservationLoss(nn.Module):
    """
    概念保留损失（Concept Preservation Loss）
    
    防止灾难性遗忘：保持模型在相关非目标概念上的生成能力
    例如：去除 nudity 时，保留 "person" 概念
    """
    
    def __init__(
        self,
        loss_type: str = "l2",
        reduction: str = "mean"
    ):
        """
        Args:
            loss_type: 损失类型，"l2" 或 "l1"
            reduction: 约简方式，"mean" 或 "none"
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(
        self,
        model_original: nn.Module,
        model_unlearned: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_preserve: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算概念保留损失
        
        目标：最小化去学习后模型与原模型在保留概念上的 velocity 差异
        
        Args:
            model_original: 原始模型（冻结）
            model_unlearned: 去学习后的模型（可训练）
            x_t: 噪声样本 (B, C, H, W, ...)
            t: 时间步 (B,)
            cond_preserve: 保留概念的条件嵌入 (B, seq_len, dim)
            mask: 空间掩码（可选）
        
        Returns:
            loss: 损失值
            loss_dict: 包含详细损失的字典
        """
        # 1. 原模型在保留概念上的 velocity
        with torch.no_grad():
            v_original = model_original(x_t, t, cond_preserve)
        
        # 2. 去学习后模型在保留概念上的 velocity
        v_unlearned = model_unlearned(x_t, t, cond_preserve)
        
        # 3. 计算损失：保持原模型的能力
        if self.loss_type == "l2":
            loss = F.mse_loss(v_unlearned, v_original, reduction='none')
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_unlearned, v_original, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 4. 应用空间掩码（如果需要）
        if mask is not None:
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(1)
            loss = loss * mask
        
        # 5. 计算平均损失
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        # 6. 构建损失字典
        loss_dict = {
            "preservation_loss": loss.clone().detach(),
            "v_orig_norm": v_original.norm().detach(),
            "v_unlearn_norm": v_unlearned.norm().detach(),
        }
        
        return loss, loss_dict


class MaskLocalizationLoss(nn.Module):
    """
    Mask-based Localization Loss (基于注意力掩码的局部化损失)
    
    根据公式：L_loc = (1/L) * sum_{l=1}^{L} ||o^l ⊙ (1 - M)||_2^2
    
    目标：迫使 eraser 在非目标概念区域（背景区域，即 1-M）输出为 0，
    从而将擦除"局限"到目标概念出现的视觉区域，避免对上下文产生过度遗忘。
    """
    
    def __init__(
        self,
        loss_type: str = "l2",
        reduction: str = "mean"
    ):
        """
        Args:
            loss_type: 损失类型，"l2" 或 "l1"（虽然公式是 L2，但提供选项）
            reduction: 约简方式，"mean" 或 "none"
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(
        self,
        eraser_outputs: Dict[str, torch.Tensor],
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 mask-based localization 损失
        
        Args:
            eraser_outputs: 字典，键为 eraser 名称，值为 eraser 输出
                {eraser_name: output_tensor}，其中 output_tensor 形状为 (B, C, H, W, ...)
            mask: 注意力掩码 M，形状为 (B, H, W, ...)，1 表示目标概念区域，0 表示背景区域
        
        Returns:
            loss: 损失值
            loss_dict: 包含详细损失的字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 确保 mask 在正确的设备上
        if not eraser_outputs:
            return torch.tensor(0.0, device=mask.device), {"total_loss": torch.tensor(0.0, device=mask.device)}
        
        device = next(iter(eraser_outputs.values())).device
        mask = mask.to(device)
        
        # 计算每个 eraser 层的损失
        layer_losses = []
        L = len(eraser_outputs)  # 层数
        for eraser_name, output in eraser_outputs.items():
            # 1. 调整 mask 形状以匹配 output
            # mask 形状通常是 (B, H, W) 或 (B, F, H, W)，需要扩展到匹配 output 的形状
            mask_expanded = mask.clone()
            while mask_expanded.dim() < output.dim():
                mask_expanded = mask_expanded.unsqueeze(1)
            
            # 确保 mask 和 output 的空间维度匹配
            if mask_expanded.shape != output.shape:
                # 如果空间维度不匹配，需要插值
                # 获取 output 的空间维度（排除 batch 和 channel 维度）
                if output.dim() == 4:  # (B, C, H, W)
                    target_h, target_w = output.shape[2], output.shape[3]
                    if mask_expanded.dim() == 4:
                        mask_expanded = F.interpolate(
                            mask_expanded.float(),
                            size=(target_h, target_w),
                            mode='nearest'
                        )
                elif output.dim() == 5:  # (B, C, F, H, W)
                    target_f, target_h, target_w = output.shape[2], output.shape[3], output.shape[4]
                    if mask_expanded.dim() == 5:
                        mask_expanded = F.interpolate(
                            mask_expanded.float(),
                            size=(target_f, target_h, target_w),
                            mode='nearest'
                        )
                    elif mask_expanded.dim() == 4:  # (B, F, H, W)
                        # 需要添加 channel 维度
                        mask_expanded = mask_expanded.unsqueeze(1)  # (B, 1, F, H, W)
                        mask_expanded = F.interpolate(
                            mask_expanded.float(),
                            size=(target_f, target_h, target_w),
                            mode='nearest'
                        )
            
            # 2. 计算背景区域：1 - M
            background_mask = 1.0 - mask_expanded
            
            # 3. 计算 o^l ⊙ (1 - M)
            masked_output = output * background_mask
            
            # 4. 计算 L2 范数平方：||o^l ⊙ (1 - M)||_2^2
            if self.loss_type == "l2":
                layer_loss = (masked_output ** 2).sum(dim=tuple(range(1, output.dim())))
            else:  # l1
                layer_loss = masked_output.abs().sum(dim=tuple(range(1, output.dim())))
            
            # 平均所有样本
            layer_loss_mean = layer_loss.mean()
            layer_losses.append(layer_loss_mean)
            loss_dict[f"{eraser_name}/loss"] = layer_loss_mean.detach()
        
        # 5. 总损失：L_loc = (1/L) * sum_{l=1}^{L} ||o^l ⊙ (1 - M)||_2^2
        if layer_losses:
            # 对所有层求平均
            total_loss = torch.stack(layer_losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=device)
        
        loss_dict["total_loss"] = total_loss.detach()
        loss_dict["num_layers"] = L
        
        return total_loss, loss_dict


class T2VUnlearningLoss(nn.Module):
    """
    T2VUnlearning 总损失函数
    
    结合负引导速度损失、概念保留损失和 Mask-based localization 损失
    """
    
    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        unlearning_weight: float = 1.0,
        preservation_weight: float = 0.5,
        localization_weight: float = 1.0,
        guidance_scale: float = 7.5,
        loss_type: str = "l2"
    ):
        """
        Args:
            alphas_cumprod: 累积 alpha 值
            unlearning_weight: 去学习损失的权重
            preservation_weight: 保留损失的权重
            localization_weight: Mask-based localization 损失的权重
            guidance_scale: 负引导强度
            loss_type: 损失类型
        """
        super().__init__()
        
        self.unlearning_loss_fn = NegativelyGuidedVelocityLoss(
            alphas_cumprod=alphas_cumprod,
            guidance_scale=guidance_scale,
            loss_type=loss_type
        )
        
        self.preservation_loss_fn = ConceptPreservationLoss(
            loss_type=loss_type
        )
        
        self.localization_loss_fn = MaskLocalizationLoss(
            loss_type=loss_type
        )
        
        self.unlearning_weight = unlearning_weight
        self.preservation_weight = preservation_weight
        self.localization_weight = localization_weight
    
    def forward(
        self,
        model_original: nn.Module,
        model_unlearned: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_target: torch.Tensor,
        cond_preserve: Optional[torch.Tensor] = None,
        cond_neg: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        eraser_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失
        
        Args:
            model_original: 原始模型（冻结）
            model_unlearned: 去学习后的模型（可训练）
            x_t: 噪声样本
            t: 时间步
            cond_target: 目标概念的条件嵌入（要去除的概念）
            cond_preserve: 保留概念的条件嵌入（可选）
            cond_neg: 负提示词的条件嵌入（可选）
            uncond: 无条件嵌入（可选）
            mask: 空间掩码（可选），用于 mask-based localization
            eraser_outputs: eraser 输出字典（可选），用于 mask-based localization
                {eraser_name: output_tensor}
        
        Returns:
            total_loss: 总损失
            loss_dict: 包含所有损失的字典
        """
        loss_dict = {}
        
        # 1. 负引导速度损失（去学习损失）
        unlearning_loss, unlearning_dict = self.unlearning_loss_fn(
            model=model_unlearned,
            x_t=x_t,
            t=t,
            cond_c=cond_target,
            cond_neg=cond_neg,
            uncond=uncond,
            mask=mask
        )
        loss_dict.update({f"unlearning/{k}": v for k, v in unlearning_dict.items()})
        
        # 2. 概念保留损失（如果提供了保留概念）
        preservation_loss = torch.tensor(0.0, device=x_t.device)
        if cond_preserve is not None:
            preservation_loss, preservation_dict = self.preservation_loss_fn(
                model_original=model_original,
                model_unlearned=model_unlearned,
                x_t=x_t,
                t=t,
                cond_preserve=cond_preserve,
                mask=mask
            )
            loss_dict.update({f"preservation/{k}": v for k, v in preservation_dict.items()})
        
        # 3. Mask-based localization 损失（如果提供了 eraser 输出和 mask）
        localization_loss = torch.tensor(0.0, device=x_t.device)
        if eraser_outputs is not None and mask is not None:
            localization_loss, localization_dict = self.localization_loss_fn(
                eraser_outputs=eraser_outputs,
                mask=mask
            )
            loss_dict.update({f"localization/{k}": v for k, v in localization_dict.items()})
        
        # 4. 总损失
        total_loss = (
            self.unlearning_weight * unlearning_loss +
            self.preservation_weight * preservation_loss +
            self.localization_weight * localization_loss
        )
        
        loss_dict["total_loss"] = total_loss.clone().detach()
        loss_dict["unlearning_loss_weighted"] = (self.unlearning_weight * unlearning_loss).detach()
        loss_dict["preservation_loss_weighted"] = (self.preservation_weight * preservation_loss).detach()
        loss_dict["localization_loss_weighted"] = (self.localization_weight * localization_loss).detach()
        
        return total_loss, loss_dict

