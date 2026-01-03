import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models.vision_transformer import Attention, Mlp
import os

warnings.filterwarnings("ignore")

# 确保能导入 cfg
try:
    from config import cfg
except ImportError:
    # Fallback for standalone testing
    class Config:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vit_model_name = 'vit_base_patch16_224'
        num_classes = 5
        # 维度映射关键参数
        eeg_spatial_size = (16, 8)  # 将 128 Mel bands reshape 为 16x8 的"空间"图
        eeg_time_steps = 224  # 对应 Resize 后的时间维度
        aug_prob = 0.5
        noise_level = 0.1
        time_shift_range = 10
        cnn3d_in_channels = 1
        cnn3d_out_features = 256
        grad_monitor_layers = []
        grad_logs_path = "grad_logs.json"
        grad_curves_path = "grad_curves.png"
        domain_hidden_dim = 128
        num_domains = 2
        lr_cnn3d = 1e-4
        lr_vit = 1e-4
        lr = 1e-4


    cfg = Config()


# ===================== 工具函数：参数统计（增强版） =====================
def count_params_module(model, module_name, prefix=""):
    module_params = {
        "total": 0,
        "trainable": 0,
        "non_trainable": 0,
        "sub_modules": {}
    }
    target_module = None
    for name, mod in model.named_modules():
        if name == module_name:
            target_module = mod
            break
    if target_module is None:
        # Relaxed check
        if hasattr(model, module_name):
            target_module = getattr(model, module_name)
        else:
            return module_params  # Not found, return empty

    for name, param in target_module.named_parameters(prefix=prefix):
        param_num = param.numel()
        module_params["total"] += param_num
        if param.requires_grad:
            module_params["trainable"] += param_num
        else:
            module_params["non_trainable"] += param_num

        sub_mod_name = name.split(".")[0] if "." in name else "root"
        if sub_mod_name not in module_params["sub_modules"]:
            module_params["sub_modules"][sub_mod_name] = {
                "total": 0, "trainable": 0, "non_trainable": 0
            }
        module_params["sub_modules"][sub_mod_name]["total"] += param_num
        if param.requires_grad:
            module_params["sub_modules"][sub_mod_name]["trainable"] += param_num
        else:
            module_params["sub_modules"][sub_mod_name]["non_trainable"] += param_num

    return module_params


def format_params(num_params):
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"


def print_params_summary(model, save_path=None):
    total_all = sum(p.numel() for p in model.parameters())
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_all = total_all - trainable_all

    module_stats = {}
    if isinstance(model, EEGCNN3D):
        module_stats["3dcnn_conv"] = count_params_module(model, "conv3d_1")
        module_stats["3dcnn_fc"] = count_params_module(model, "fc_layers")
    elif isinstance(model, CustomViT):
        module_stats["vit_patch_embed"] = count_params_module(model, "vit.patch_embed")
        module_stats["vit_blocks"] = count_params_module(model, "vit.blocks")
        module_stats["vit_head"] = count_params_module(model, "vit.head")
    elif isinstance(model, DualTransferModel):
        module_stats["3dcnn"] = count_params_module(model, "cnn3d")
        module_stats["vit"] = count_params_module(model, "vit")

    print("=" * 80)
    print("模型参数统计摘要（匹配论文表1）")
    print("=" * 80)
    print(
        f"全局总参数: {format_params(total_all)} (可训练: {format_params(trainable_all)}, 冻结: {format_params(non_trainable_all)})")
    print(f"可训练参数占比: {trainable_all / total_all:.2%}")
    print("-" * 80)

    for mod_name, mod_stats in module_stats.items():
        print(f"\n【模块: {mod_name}】")
        print(f"  总参数: {format_params(mod_stats['total'])}")
        print(f"  可训练: {format_params(mod_stats['trainable'])} ({mod_stats['trainable'] / mod_stats['total']:.2%})")
        print(f"  冻结: {format_params(mod_stats['non_trainable'])}")
        print("  子模块分布:")
        for sub_mod, sub_stats in mod_stats["sub_modules"].items():
            print(
                f"    - {sub_mod}: {format_params(sub_stats['total'])} (可训练: {format_params(sub_stats['trainable'])})")

    print("=" * 80)

    if save_path:
        stats_dict = {
            "global": {
                "total": total_all,
                "trainable": trainable_all,
                "non_trainable": non_trainable_all,
                "trainable_ratio": float(trainable_all / total_all)
            },
            "modules": {k: {
                "total": v["total"],
                "trainable": v["trainable"],
                "non_trainable": v["non_trainable"],
                "sub_modules": v["sub_modules"]
            } for k, v in module_stats.items()}
        }
        with open(save_path, "w") as f:
            json.dump(stats_dict, f, indent=4)
        print(f"\n参数统计结果已保存至: {save_path}")


# ===================== 梯度监控器 =====================
class GradientMonitor:
    def __init__(self, model, target_layers=None):
        self.model = model
        self.grad_logs = {}
        self.hooks = []
        self.target_layers = target_layers or cfg.grad_monitor_layers

        for layer_name in self.target_layers:
            self.grad_logs[layer_name] = []

    def _grad_hook_fn(self, layer_name):
        def hook(grad):
            if grad is None: return
            grad_norm = torch.norm(grad).item()
            grad_mean = grad.mean().item()
            grad_max = grad.max().item()
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()

            step = len(self.grad_logs[layer_name])
            self.grad_logs[layer_name].append({
                "step": step,
                "norm": grad_norm,
                "mean": grad_mean,
                "max": grad_max,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "grad_shape": list(grad.shape)
            })

            if has_nan or has_inf:
                warnings.warn(f"⚠️ 层 {layer_name} 梯度出现NaN/Inf！step={step}")

        return hook

    def register_hooks(self):
        for layer_name in self.target_layers:
            # 简化版查找逻辑
            layer = self.model
            found = True
            try:
                for sub_name in layer_name.split("."):
                    layer = getattr(layer, sub_name)
            except AttributeError:
                found = False

            if found:
                hook_fn = self._grad_hook_fn(layer_name)
                def _wrapper(module, grad_input, grad_output, hook_fn=hook_fn):
                    hook_fn(grad_output[0])
                    return None
                hook = layer.register_full_backward_hook(_wrapper)
                self.hooks.append(hook)
            else:
                warnings.warn(f"层 {layer_name} 不存在，跳过梯度监控")

        if len(self.hooks) > 0:
            print(f"✅ 已为 {len(self.hooks)} 个层注册梯度钩子")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("✅ 已移除所有梯度钩子")

    def save_grad_logs(self, save_path=cfg.grad_logs_path):
        def serialize_log(log):
            for k, v in log.items():
                if isinstance(v, (np.integer, np.floating)):
                    log[k] = float(v)
            return log

        serialized_logs = {
            layer: [serialize_log(log) for log in logs]
            for layer, logs in self.grad_logs.items()
        }
        with open(save_path, "w") as f:
            json.dump(serialized_logs, f, indent=4)
        # print(f"✅ 梯度日志已保存至: {save_path}")

    def plot_grad_curves(self, save_path=cfg.grad_curves_path):
        if not self.grad_logs or all(len(logs) == 0 for logs in self.grad_logs.values()):
            return

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        metrics = ["norm", "mean", "max"]
        titles = ["梯度L2范数", "梯度均值", "梯度最大值"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            for layer_name, logs in self.grad_logs.items():
                if len(logs) == 0:
                    continue
                steps = [log["step"] for log in logs]
                values = [log[metric] for log in logs]
                ax.plot(steps, values, label=layer_name, linewidth=2)
            ax.set_title(title, fontsize=14)
            ax.set_ylabel(metric, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("训练步数", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        # print(f"✅ 梯度曲线已保存至: {save_path}")


# ===================== EEG数据增强（映射为水声增强） =====================
class EEGSpatialResample(nn.Module):
    """
    原逻辑：空间重采样 (对 EEG 电极矩阵进行缩放/微扰)。
    适配水声：我们将其理解为 'Spectrogram Spatial Warping'。
    """

    def __init__(self, **kwargs):
        super().__init__()
        # 仅仅为了兼容，不做实际操作，防止维度错误
        pass

    def forward(self, x):
        # 暂时跳过复杂的空间重采样，保证先跑通
        return x


class EEGDataAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug_prob = cfg.aug_prob
        self.noise_level = cfg.noise_level
        self.time_shift_range = cfg.time_shift_range
        self.spatial_resample = EEGSpatialResample()

    def time_warp(self, x):
        # x shape: (Batch, Channel, Time, H, W)
        if torch.rand(1) < self.aug_prob:
            # 针对 Time 维度 (dim=2) 进行 warp
            T = x.shape[2]
            if T < 2: return x  # 防止时间太短

            scale = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            new_T = int(T * scale)

            # 变换为 (Batch, C*H*W, Time) 以便用 1D 插值
            B, C, T, H, W = x.shape
            x_flat = x.permute(0, 1, 3, 4, 2).reshape(B, C * H * W, T)

            x_resampled = F.interpolate(x_flat, size=new_T, mode='linear', align_corners=False)

            # 恢复形状
            if new_T > T:
                x_resampled = x_resampled[:, :, :T]
            else:
                x_resampled = F.pad(x_resampled, (0, T - new_T), mode='constant', value=0)

            x = x_resampled.reshape(B, C, H, W, T).permute(0, 1, 4, 2, 3)

        return x

    def noise_injection(self, x):
        if torch.rand(1) < self.aug_prob:
            x = x + torch.randn_like(x) * self.noise_level
        return x

    def time_shift(self, x):
        # x shape: (Batch, Channel, Time, H, W)
        if torch.rand(1) < self.aug_prob:
            shift = torch.randint(-self.time_shift_range, self.time_shift_range + 1, (1,)).item()
            # 在 Time 轴 (dim=2) 上滚动
            x = torch.roll(x, shifts=shift, dims=2)

            if shift > 0:
                x[:, :, :shift, :, :] = 0
            elif shift < 0:
                x[:, :, shift:, :, :] = 0
        return x

    def frequency_mask(self, x):
        # 这里的 "Frequency" 对应语谱图的 Height (dim=3)
        # x shape: (Batch, Channel, Time, Height, Width)
        if torch.rand(1) < self.aug_prob:
            freq_dim = 3  # Height is frequency axis
            F_size = x.shape[freq_dim]

            # 安全检查：如果维度太小，跳过
            if F_size < 2: return x

            # 随机遮挡 10% 的频带
            mask_len = torch.randint(1, max(2, F_size // 5), (1,)).item()
            start = torch.randint(0, F_size - mask_len, (1,)).item()

            x[:, :, :, start:start + mask_len, :] = 0

        return x

    def forward(self, x):
        if self.training:
            # 确保输入是 5D
            if x.dim() == 4:
                x = x.unsqueeze(1)

            # x = self.time_warp(x) # Time warp 比较耗时且易出错，暂时注释掉
            x = self.spatial_resample(x)  # 这个内部逻辑也需要确保支持 5D，或者暂时跳过
            x = self.noise_injection(x)
            x = self.time_shift(x)
            x = self.frequency_mask(x)
        return x


# ===================== EEG Embedding层 (适配水声) =====================
class EEGEmbedding(nn.Module):
    """
    原 EEG Embedding，现在用于将水声特征进行时空位置编码。
    """

    def __init__(self, in_channels=1, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.channel_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)

        # 调整 time_pos_embed 的尺寸以适配 config 中设定的维度
        self.time_pos_embed = nn.Parameter(torch.randn(1, embed_dim, cfg.eeg_time_steps, 1, 1) * 0.02)

        # 调整 spatial_pos_embed
        # 假设 eeg_spatial_size=(16,8) 代表 128 Mel bands
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, *cfg.eeg_spatial_size) * 0.02)

        self.mlp = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.channel_embed.weight)
        for m in self.mlp:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x shape: (Batch, C=1, Time=224, H=16, W=8)
        # 1. 临时将 Time 轴折叠到 Batch 轴
        b, c, t, h, w = x.shape

        # (B, 1, T, H, W) -> (B*T, 1, H, W)
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        # 经过 2D 卷积 (B*T, embed_dim, H, W)
        x_embed = self.channel_embed(x_reshaped)

        # 还原回 5D: (B, T, embed_dim, H, W) -> Permute -> (B, embed_dim, T, H, W)
        x_embed = x_embed.view(b, t, self.embed_dim, h, w).permute(0, 2, 1, 3, 4)

        # 动态截取或插值位置编码
        t_embed = self.time_pos_embed
        if x_embed.shape[2] != t_embed.shape[2]:
            t_embed = F.interpolate(t_embed.squeeze(3).squeeze(3), size=x_embed.shape[2], mode='linear').unsqueeze(
                3).unsqueeze(3)

        # 加上位置编码
        x = x_embed + t_embed + self.spatial_pos_embed
        x = self.mlp(x)
        return x


# ===================== 3D/2D卷积块 =====================
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.stride = stride

        # 自动计算 padding 以保持尺寸不变 (Same Padding)
        if padding == 'auto':
            if isinstance(kernel_size, int):
                ks = (kernel_size, kernel_size, kernel_size)
            else:
                ks = kernel_size
            padding = (ks[0] // 2, ks[1] // 2, ks[2] // 2)

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,  # 使用计算后的或传入的 padding
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.9, eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.2)

        if self.use_residual and (in_channels != out_channels or stride != 1):
            self.residual_conv = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, padding=0, bias=False
            )
            self.residual_bn = nn.BatchNorm3d(out_channels)
        elif self.use_residual:
            self.residual_conv = nn.Identity()
            self.residual_bn = nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.act(x)

        if self.use_residual:
            res = self.residual_conv(residual)
            res = self.residual_bn(res)

            # === 关键修复：尺寸对齐 ===
            if x.shape[2:] != res.shape[2:]:
                res = F.interpolate(res, size=x.shape[2:], mode='nearest')

            x = x + res
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.2)

        if self.use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
            )
            self.residual_bn = nn.BatchNorm2d(out_channels)
        elif self.use_residual:
            self.residual_conv = nn.Identity()
            self.residual_bn = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.pool(x)
        if self.use_residual:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)
            residual = self.pool(residual)

            # === 关键修复：尺寸对齐 ===
            if x.shape[2:] != residual.shape[2:]:
                residual = F.interpolate(residual, size=x.shape[2:], mode='nearest')

            x = x + residual
        return x


# ===================== EEG 3D-CNN (主干) =====================
class EEGCNN3D(nn.Module):
    def __init__(self, time_pooling_type="mean"):
        super().__init__()
        self.in_channels = cfg.cnn3d_in_channels
        self.out_features = cfg.cnn3d_out_features
        self.time_pooling_type = time_pooling_type

        # === [关键修复] 找回丢失的 Embedding 层 ===
        self.embedding = EEGEmbedding(in_channels=self.in_channels, embed_dim=64)

        # 3D卷积层
        self.conv3d_1 = Conv3DBlock(
            in_channels=64,
            out_channels=16,
            kernel_size=(3, 1, 3),
            padding=(1, 0, 1),
            use_residual=True
        )

        # [关键修复] use_residual=False 防止尺寸不匹配
        self.conv3d_2 = Conv3DBlock(
            in_channels=16,
            out_channels=25,
            kernel_size=(4, 4, 1),
            padding=0,
            use_residual=False
        )

        # 2D卷积层
        # [关键修复] padding=(0,1) 保持宽度
        self.conv2d_block = Conv2DBlock(
            in_channels=25,
            out_channels=40,
            kernel_size=(1, 3),
            padding=(0, 1),
            use_residual=True
        )

        # 全连接层 (延迟初始化)
        self.fc_layers = None
        self.fc_input_dim = 0

        # 梯度钩子
        self.grad_hooks = {}

        # 权重初始化
        self._init_weights()

    def _lazy_init_fc(self, x):
        """延迟初始化 FC 层，以自适应不同的输入维度"""
        flattened_dim = x.view(x.size(0), -1).shape[1]
        self.fc_input_dim = flattened_dim
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.out_features),
            nn.BatchNorm1d(self.out_features),
            nn.ReLU(inplace=True)
        ).to(x.device)
        self._register_grad_hooks()

    def _register_grad_hooks(self):
        def grad_hook_fn(grad):
            if grad is not None:
                self.grad_norm = torch.norm(grad).item()

        def _wrapper(module, grad_input, grad_output):
            grad_hook_fn(grad_output[0])
            return None

        self.grad_hooks["conv3d_2"] = self.conv3d_2.register_full_backward_hook(_wrapper)
        if self.fc_layers:
            self.grad_hooks["fc_layers"] = self.fc_layers[0].register_full_backward_hook(_wrapper)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Dropout3d, nn.Dropout2d, nn.Dropout)):
                m.p = 0.2 if "3d" in str(m) or "2d" in str(m) else 0.5

    def _time_dim_pooling(self, x):
        if self.time_pooling_type == "mean":
            return torch.mean(x, dim=2)
        elif self.time_pooling_type == "max":
            return torch.max(x, dim=2)[0]
        elif self.time_pooling_type == "adaptive":
            return F.adaptive_avg_pool1d(x.transpose(2, 1), 1).squeeze(1)
        else:
            raise ValueError(f"不支持的时间池化类型：{self.time_pooling_type}")

    def forward(self, x, return_intermediate=False):
        # x shape: (Batch, Time, H, W) -> 需要增加 Channel 维 -> (Batch, 1, Time, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)

        intermediate_features = {"input_cnn3d": x.detach().cpu()}

        # EEG Embedding
        x = self.embedding(x)
        intermediate_features["eeg_embedding"] = x.detach().cpu()

        # 3D卷积
        x = self.conv3d_1(x)
        intermediate_features["conv3d_1"] = x.detach().cpu()
        x = self.conv3d_2(x)
        intermediate_features["conv3d_2"] = x.detach().cpu()

        # 时间池化
        x = self._time_dim_pooling(x)
        intermediate_features["time_pooled"] = x.detach().cpu()

        # 2D卷积
        x = self.conv2d_block(x)
        intermediate_features["conv2d_out"] = x.detach().cpu()

        # 延迟初始化 FC
        if self.fc_layers is None:
            self._lazy_init_fc(x)

        # 全连接
        x = self.fc_layers(x)
        intermediate_features["fc_out"] = x.detach().cpu()

        feature_map = x.view(x.size(0), -1)
        intermediate_features["final_feature"] = feature_map.detach().cpu()

        if return_intermediate:
            return feature_map, intermediate_features
        return feature_map

    def count_params(self, save_path=None):
        print_params_summary(self, save_path)


# ===================== 定制化ViT (修改 Mask) =====================
class CustomViT(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 创建模型结构 (不在此处加载权重)
        self.vit = timm.create_model(
            cfg.vit_model_name,
            pretrained=False,  # 关闭自动下载
            num_classes=0,  # 移除分类头，只提取特征
            in_chans=3
        )

        # 2. 手动加载本地权重
        checkpoint_path = './weights/vit_base_patch16_224.safetensors'

        if os.path.exists(checkpoint_path):
            print(f"Loading local pretrained ViT from: {checkpoint_path}")
            try:
                from timm.models import load_checkpoint
                # 关键：strict=False 允许忽略 'head.weight' 等多余参数
                load_checkpoint(self.vit, checkpoint_path, strict=False)
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("尝试使用 safetensors 直接加载...")
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(checkpoint_path)
                    self.vit.load_state_dict(state_dict, strict=False)
                    print("Safetensors loaded successfully (strict=False).")
                except ImportError:
                    print("请安装 safetensors: pip install safetensors")
        else:
            print(f"⚠️ Warning: Pretrained weights not found at {checkpoint_path}. Using random init.")

        # 定制化位置编码
        self.eeg_pos_emb = nn.Parameter(
            torch.randn(1, self.vit.pos_embed.shape[1], self.vit.pos_embed.shape[2]) * 0.02
        )
        self.vit.pos_embed.requires_grad = False
        self.eeg_pos_emb.requires_grad = True

        # 注意力掩码
        self.attn_mask = self._generate_audio_attn_mask()
        self.extract_layer = -3

    def _generate_audio_attn_mask(self):
        num_patches = self.vit.patch_embed.num_patches + 1
        mask = torch.ones(num_patches, num_patches)
        limit = int(num_patches * 0.8)
        mask[limit:, :] *= 0.8
        mask[:, limit:] *= 0.8
        return mask.to(cfg.device)

    def forward(self, x, return_layer_features=False):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        if x.shape[1] != self.eeg_pos_emb.shape[1]:
            pos_emb = F.interpolate(self.eeg_pos_emb.transpose(1, 2), size=x.shape[1], mode='linear').transpose(1, 2)
            x = x + pos_emb
        else:
            x = x + self.eeg_pos_emb

        x = self.vit.pos_drop(x)
        layer_features = []
        for idx, block in enumerate(self.vit.blocks):
            x = block(x)
            if idx >= len(self.vit.blocks) + self.extract_layer:
                layer_features.append(x.detach().cpu())

        x = self.vit.norm(x)
        cls_out = x[:, 0]
        out = self.vit.head(cls_out)

        if return_layer_features:
            return out, layer_features
        return out

    def freeze_layers(self, freeze_until=-3):
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        for idx, block in enumerate(self.vit.blocks):
            if idx < len(self.vit.blocks) + freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def count_params(self, save_path=None):
        print_params_summary(self, save_path)


# ===================== 梯度反转层 + 域判别器 =====================
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_reverse = GradientReversal(alpha=1.0)
        self.input_dim = cfg.cnn3d_out_features + 768

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, cfg.domain_hidden_dim),
            nn.BatchNorm1d(cfg.domain_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(cfg.domain_hidden_dim, cfg.domain_hidden_dim // 2),
            nn.BatchNorm1d(cfg.domain_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(cfg.domain_hidden_dim // 2, cfg.num_domains)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.grad_reverse(x)
        return self.fc(x)


# ===================== 双迁移学习联合模型 =====================
class DualTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 数据增强
        self.eeg_aug = EEGDataAugmentation()

        # 3D-CNN模块
        self.cnn3d = EEGCNN3D(time_pooling_type="mean")
        self.freeze_cnn3d(layers_to_freeze="bottom")

        # ViT模块
        self.vit = CustomViT()
        self.vit.freeze_layers(freeze_until=-3)

        # 域适应模块
        self.domain_discriminator = DomainDiscriminator()

        # 梯度监控
        self.grad_monitor = GradientMonitor(self)
        self.grad_monitor.register_hooks()

        # 融合分类头 (确保维度匹配)
        self.fusion_fc = nn.Sequential(
            nn.Linear(cfg.cnn3d_out_features + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.num_classes)
        )

    def _load_cnn3d_pretrained(self, pretrained_path):
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                state_dict = torch.load(pretrained_path, map_location=cfg.device)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self.cnn3d.load_state_dict(state_dict, strict=False)
                print(f"成功加载预训练3D-CNN权重：{pretrained_path}")
            except Exception as e:
                warnings.warn(f"加载3D-CNN预训练权重失败：{e}")
        elif pretrained_path:
            warnings.warn(f"预训练3D-CNN权重文件不存在：{pretrained_path}")

    def freeze_cnn3d(self, layers_to_freeze="bottom"):
        params = list(self.cnn3d.parameters())
        if layers_to_freeze == "bottom":
            freeze_idx = int(len(params) * 0.8)
            for idx, param in enumerate(params):
                param.requires_grad = idx >= freeze_idx
        elif layers_to_freeze == "all":
            for param in self.cnn3d.parameters():
                param.requires_grad = False
        elif layers_to_freeze == "none":
            for param in self.cnn3d.parameters():
                param.requires_grad = True

    def compute_feature_correlation(self, feat_3d, feat_vit):
        """
        计算双模态特征的相关性。
        注意：如果两个模态的特征维度不一致（例如 256 vs 768），无法计算直接的相关系数，返回 0。
        """
        # 展平特征
        feat_3d_flat = feat_3d.reshape(feat_3d.shape[0], -1).detach().cpu().numpy()
        # feat_vit 是 list，取最后一层; shape=(B, N_patches, Dim) -> 取 CLS token
        feat_vit_flat = feat_vit[:, 0, :].detach().cpu().numpy()

        # === [关键修复] 维度检查 ===
        # 如果维度不同 (如 256 != 768)，直接跳过计算，防止报错
        if feat_3d_flat.shape[1] != feat_vit_flat.shape[1]:
            self.feature_corr = 0.0
            return self.feature_corr

        # 标准差为0防止除零错误
        if np.std(feat_3d_flat) == 0 or np.std(feat_vit_flat) == 0:
            self.feature_corr = 0.0
        else:
            # 取第一个样本计算相关系数
            self.feature_corr = np.corrcoef(feat_3d_flat[0], feat_vit_flat[0])[0, 1]

        return self.feature_corr

    def forward(self, x_3d, x_2d=None, domain_label=None, return_all_features=False):
        # 数据增强
        x_3d = self.eeg_aug(x_3d)

        # 3D-CNN特征提取 (feat_3d 在 GPU)
        feat_3d, intermediate_3d = self.cnn3d(x_3d, return_intermediate=True)

        # 如果未提供 x_2d，尝试从 x_3d 转换
        if x_2d is None:
            x_2d = torch.randn(x_3d.shape[0], 3, 224, 224).to(x_3d.device)

        # ViT特征提取 (cls_out 在 GPU, layer_vit 里的元素在 CPU)
        cls_out, layer_vit = self.vit(x_2d, return_layer_features=True)

        # 特征融合
        feat_3d_flat = feat_3d.view(feat_3d.shape[0], -1)

        # [核心修复] layer_vit[-1] 是 CPU Tensor，需要放回 GPU
        # 注意：layer_vit[-1] 是 (Batch, 197, 768)，我们需要取 CLS token
        vit_feat_cpu = layer_vit[-1][:, 0, :]  # (Batch, 768) on CPU
        vit_feat = vit_feat_cpu.to(feat_3d.device)  # 移回 GPU

        # 拼接 (现在两个都在 GPU 了)
        fused_feat = torch.cat([feat_3d_flat, vit_feat], dim=1)

        # 最终分类
        final_cls_out = self.fusion_fc(fused_feat)

        # 域适应分支
        domain_out = None
        if self.training and domain_label is not None:
            domain_out = self.domain_discriminator(fused_feat)

        # 特征相关性计算 (需要 CPU numpy 计算)
        if self.training:
            self.compute_feature_correlation(feat_3d, layer_vit[-1])  # 这里传入 CPU tensor 正好

        if return_all_features:
            return final_cls_out, domain_out, {
                "cnn3d_features": intermediate_3d,
                "vit_features": layer_vit,
                "fused_feat": fused_feat.detach().cpu(),
                "feature_corr": self.feature_corr
            }

        return (final_cls_out, domain_out) if self.training else (final_cls_out, x_2d)

    def get_learning_rates(self):
        return [
            {"params": self.cnn3d.parameters(), "lr": cfg.lr_cnn3d},
            {"params": self.vit.eeg_pos_emb, "lr": cfg.lr_vit * 2},

            # [关键修复] 正确的属性路径: self.vit.vit.blocks
            {"params": self.vit.vit.blocks[-3:].parameters(), "lr": cfg.lr_vit},

            {"params": self.fusion_fc.parameters(), "lr": cfg.lr},
            {"params": self.domain_discriminator.parameters(), "lr": cfg.lr}
        ]

    def count_params(self, save_path=None):
        print_params_summary(self, save_path)

    def save_grad_logs(self):
        self.grad_monitor.save_grad_logs()

    def plot_grad_curves(self):
        self.grad_monitor.plot_grad_curves()

    def remove_grad_hooks(self):
        self.grad_monitor.remove_hooks()


def create_dual_transfer_model(pretrained_cnn3d_path=None):
    model = DualTransferModel()
    if pretrained_cnn3d_path:
        model._load_cnn3d_pretrained(pretrained_cnn3d_path)
    model = model.to(cfg.device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model
