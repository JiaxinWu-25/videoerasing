import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from einops import rearrange

from diffusers.models.attention import Attention
from .erasers.utils import ldm_module_prefix_name
import math

import os
from torchvision.utils import save_image

def concatenate_frames(frames):
    """
    Concatenate multiple frames horizontally into a single image
    
    Args:
        frames: List of frames (F, H, W)
        
    Returns:
        Concatenated image (H, F*W)
    """
    # Ensure all frames are of the same size
    H, W = frames[0].shape
    num_frames = len(frames)
    
    # Create a blank canvas
    concat_image = torch.zeros((H, W * num_frames), device=frames[0].device, dtype=frames[0].dtype)
    
    # Horizontally concatenate all frames
    for i, frame in enumerate(frames):
        concat_image[:, i*W:(i+1)*W] = frame
        
    return concat_image
    #把它们横向拼接成一张图像

def save_mask_frames(tensor: torch.Tensor, save_dir: str, prefix: str = "frame"):
    """
    Saves each frame of a binary tensor as images and concatenates all frames into a single image.

    Args:
        tensor (torch.Tensor): A tensor of shape [1, frames, height, width] containing only 0s and 1s.
        save_dir (str): Directory to save the images.
        prefix (str): Prefix for saved image filenames.
    """
    if tensor.ndimension() != 4 or tensor.shape[0] != 1:
        raise ValueError("Tensor must have shape [1, frames, height, width]")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Concatenate all frames horizontally and save as one image
    concatenated = concatenate_frames(tensor[0])
    save_path = os.path.join(save_dir, f"{prefix}_concatenated.png")
    save_image(concatenated, save_path)

@torch.no_grad()
def get_cross_attn_mask(query, key, token_idx, head_num = None,text_len = 226,height = 10, width = 10):
    #you'll have to input the height and width of the latent here
    inner_dim = key.shape[-1]#inner dim = 1920
    
    # 如果 head_num 未提供，尝试从 inner_dim 推断
    # CogVideoX 常见的配置：30 heads * 64 dim = 1920
    # 或者 40 heads * 48 dim = 1920
    if head_num is None:
        # 尝试常见的 head_num 值
        for possible_head_num in [30, 40, 32, 24, 20]:
            if inner_dim % possible_head_num == 0:
                head_num = possible_head_num
                break
        if head_num is None:
            # 如果都不行，尝试找到最大的能整除的因子
            for i in range(inner_dim, 0, -1):
                if inner_dim % i == 0 and i <= 64:  # head_num 通常不会太大
                    head_num = i
                    break
        if head_num is None:
            raise ValueError(f"Cannot infer head_num from inner_dim={inner_dim}")
    
    head_dim = inner_dim // head_num
    
    # 检查 query 和 key 的形状是否兼容
    query_total_elements = query.numel()
    key_total_elements = key.numel()
    batch_size = query.shape[0]
    
    # query 应该是 (batch_size, seq_len, inner_dim) 或 (batch_size, seq_len * inner_dim)
    # 需要 reshape 成 (batch_size, seq_len, head_num, head_dim)
    if len(query.shape) == 2:
        # 如果是 2D，需要推断 seq_len
        seq_len = query_total_elements // (batch_size * inner_dim)
        if seq_len * batch_size * inner_dim != query_total_elements:
            # 尝试直接 reshape
            query = query.view(batch_size, -1, inner_dim)
        else:
            query = query.view(batch_size, seq_len, inner_dim)
    elif len(query.shape) == 3:
        # 已经是 3D，确保最后一维是 inner_dim
        if query.shape[-1] != inner_dim:
            # 可能需要 flatten 中间维度
            query = query.view(batch_size, -1, inner_dim)
    else:
        query = query.view(batch_size, -1, inner_dim)
    
    # 对 key 做同样的处理
    if len(key.shape) == 2:
        seq_len = key_total_elements // (batch_size * inner_dim)
        if seq_len * batch_size * inner_dim != key_total_elements:
            key = key.view(batch_size, -1, inner_dim)
        else:
            key = key.view(batch_size, seq_len, inner_dim)
    elif len(key.shape) == 3:
        if key.shape[-1] != inner_dim:
            key = key.view(batch_size, -1, inner_dim)
    else:
        key = key.view(batch_size, -1, inner_dim)
    
    # 现在 reshape 成 (batch_size, seq_len, head_num, head_dim) 然后 transpose
    # 从 (batch_size, seq_len, head_num, head_dim) -> (batch_size, head_num, seq_len, head_dim)
    query = query.view(query.shape[0], query.shape[1], head_num, head_dim).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], head_num, head_dim).transpose(1, 2) 
   

    attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
    attn_probs = torch.softmax(attn_scores, dim=-1)

    attn_map_mean = attn_probs.sum(dim=1) / head_num #heads

    attn_map = attn_map_mean[:, :text_len, text_len:    ] #batch_size,text_len,visual_len

    B, HWF, T = attn_map.shape
    F = HWF // (height * width)
    attn_map = attn_map.reshape(B, F, height, width, T)

    
    attn_map = attn_map[..., token_idx]
    attn_map = attn_map.sum(dim=-1)  # Sum over selected tokens
   
    # Get min and max values using PyTorch
    attn_min = attn_map.amin(dim=(2, 3), keepdim=True)
    attn_max = attn_map.amax(dim=(2, 3), keepdim=True)
    
    normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)  # Add small value to avoid division by zero
    
    return normalized_attn



#apparently, we will get b * f * h * w latents
@torch.no_grad()
def get_mask(attn_maps, word_indices, thres, height, width, head_num=30, text_len=226):
    """
    attn_maps: {module: {name : b, seqlen, head_dim * heads}}}
    word_indices: (num_tokens,)
    thres: float, threshold of mask
    """
    ret_masks = {}
    attns_choosen = []
    average_mask = None
    count = 0.0
    for name, attns in attn_maps.items():
        
        #print(attns)
        query = attns['to_q']# (bs, )
        key = attns['to_k']
        mask = get_cross_attn_mask(query,key,word_indices,height=height,width=width,head_num=head_num,text_len=text_len)
        if average_mask is None:
            average_mask = mask
        else:
            average_mask += mask
        count += 1.0
        mask[mask >= thres] = 1
        mask[mask < thres] = 0
    
        ret_masks[name] = mask

    average_mask /= count

    ret_masks["average_mask"] = average_mask
    return ret_masks


class AttnMapsCapture:
    def __init__(self, model, attn_maps):
        self.model = model
        self.attn_maps = attn_maps
        self.handlers = []

    def __enter__(self):
        
        for name,module in self.model.named_modules():
            if isinstance(module, Attention):
                #print("add forward hook in:",name)
                h_q = module.to_q.register_forward_hook(self.get_attn_maps(name,"to_q"))
                h_k = module.to_k.register_forward_hook(self.get_attn_maps(name,"to_k"))

                self.handlers.append(h_q)
                self.handlers.append(h_k)

    def __exit__(self, exc_type, exc_value, traceback):
        for handler in self.handlers:
            handler.remove()

    def get_attn_maps(self, module,name):
            def hook(model, input, output):
                if module not in self.attn_maps.keys():
                    self.attn_maps[module] = {}
                self.attn_maps[module][name] = output.detach()
            return hook

class EraserOutputsCapture:
    def __init__(self, model, erasers, eraser_outs):
        self.model = model
        self.eraser_names = list(erasers.keys())
        self.eraser_outs = eraser_outs
        self.handlers = []

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if module_name in self.eraser_names:
                handler = module.register_forward_hook(self.get_eraser_outs(module_name))
                self.handlers.append(handler)

    def __exit__(self, exc_type, exc_value, traceback):
        for handler in self.handlers:
            handler.remove()

    def get_eraser_outs(self, module_name):
            def hook(model, input, output):
                self.eraser_outs[module_name] = output
            return hook

"""
提取 Mask：利用 get_mask 知道“自行车”在哪。
局限更新：通过 Loss 强迫 Eraser 在“非自行车区域”（即 1−M ）输出为 0。
避免过度遗忘：因为 Eraser 在背景区域不工作（输出为 0），所以模型原本关于草地、天空、背景的知识被保留了下来（Adapter 加上 0 等于没加），只有目标物体被修改了。
"""