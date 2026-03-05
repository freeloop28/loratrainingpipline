import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA 权重初始化
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 冻结原始权重
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x):
        # 原始路径 + LoRA 路径
        return self.original_linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

def inject_lora(model, rank=4, alpha=1.0, target_replace_modules=["CrossAttention", "Attention"]):
    """
    递归遍历模型，将指定的 Attention 层中的线性层替换为 LoRA 版本
    """
    for name, module in model.named_modules():
        if any(t in module.__class__.__name__ for t in target_replace_modules):
            # 针对 Attention 中的 Q, K, V, Out 线性层进行替换
            for subname, submodule in module.named_children():
                if isinstance(submodule, nn.Linear):
                    setattr(module, subname, LoRALinear(submodule, rank, alpha))
    return model

def freeze_non_lora(model):
    """
    冻结所有非 LoRA 参数
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    return model
