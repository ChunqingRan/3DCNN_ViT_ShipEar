import torch
import torch.nn as nn
import os
import warnings
from config import cfg


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=None, delta=None, monitor=None, mode=None, save_path=None):
        # 优先使用传入参数，否则尝试从cfg读取，最后使用默认值
        self.patience = patience or getattr(cfg, 'patience', 15)
        self.min_delta = delta or getattr(cfg, 'min_delta', 0.0)
        self.monitor = monitor or getattr(cfg, 'monitor', 'val_acc')  # 默认监控准确率
        self.mode = mode or getattr(cfg, 'early_stop_mode', 'max')  # acc是max, loss是min

        # 确定保存路径
        if save_path:
            self.save_path = save_path
        elif hasattr(cfg, 'save_dir'):
            self.save_path = os.path.join(cfg.save_dir, "best_model.pth")
        else:
            self.save_path = "best_model.pth"

        self.verbose = True
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, model, epoch):
        # 初始化
        if self.best_score is None:
            self.best_score = current_score
            self._save_best_model(model, epoch)
            return

        # 判断是否改进
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:  # max
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            self._save_best_model(model, epoch)
            if self.verbose:
                print(f"✅ 监控指标改进 ({self.monitor}): {current_score:.6f} → 保存最佳模型（Epoch {epoch}）")
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"⚠️ 早停计数器: {self.counter}/{self.patience} (当前{self.monitor}: {current_score:.6f}, 最佳: {self.best_score:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 早停触发！最佳Epoch: {self.best_epoch}, 最佳{self.monitor}: {self.best_score:.6f}")

    def _save_best_model(self, model, epoch):
        # 确保目录存在
        directory = os.path.dirname(self.save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 处理 DataParallel
        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        save_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "best_score": self.best_score,
            # 安全地保存 config (过滤掉不可序列化的对象)
            "config": {k: str(v) for k, v in cfg.__dict__.items() if not k.startswith("__")}
        }

        torch.save(save_dict, self.save_path)
        if self.verbose:
            print(f"📌 最佳模型已保存至: {self.save_path}")

    def load_best_model(self, model):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location=cfg.device)

            # 处理 state_dict 键名可能带 module. 前缀的问题
            state_dict = checkpoint["model_state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=False)
            if self.verbose:
                print(f"📥 加载最佳模型（Epoch {checkpoint['epoch']}, {self.monitor}: {checkpoint['best_score']:.6f}）")
            return model
        else:
            raise FileNotFoundError(f"最佳模型文件不存在: {self.save_path}")


def build_optimizer_and_scheduler(model):
    """构建优化器和学习率调度器"""
    # 尝试获取分层学习率参数，如果没有实现接口则获取默认参数
    if hasattr(model, "get_learning_rates"):
        params = model.get_learning_rates()
    elif hasattr(model, "module") and hasattr(model.module, "get_learning_rates"):
        params = model.module.get_learning_rates()
    else:
        warnings.warn("模型未实现 get_learning_rates 接口，使用统一学习率")
        params = model.parameters()

    # 优化器
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.lr if hasattr(cfg, 'lr') else 1e-4,
        weight_decay=cfg.weight_decay if hasattr(cfg, 'weight_decay') else 1e-4,
        betas=(0.9, 0.999)
    )

    # 调度器 (默认使用 CosineAnnealingLR)
    scheduler_type = getattr(cfg, 'scheduler_type', 'CosineAnnealingLR')
    max_epochs = getattr(cfg, 'epochs', 50)  # 注意之前是 max_epochs，新config是 epochs
    min_lr = getattr(cfg, 'min_lr', 1e-6)

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # 注意：如果是监控准确率，这里可能是 max
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=True
        )
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=min_lr,
            # verbose=True # Pytorch 旧版本可能不支持 verbose
        )
    elif scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.7,
            verbose=True
        )
    else:
        scheduler = None

    return optimizer, scheduler