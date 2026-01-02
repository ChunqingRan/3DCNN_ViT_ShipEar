import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm

# ===================== 导入自定义模块 =====================
# 确保所有模块都在同一目录下
from config import cfg
from models import create_dual_transfer_model, DualTransferModel
from losses import JointLoss
from metrics import MetricTracker
from data_loader import get_dataloaders  # 导入真实的 ShipEar 数据加载器


# ===================== 简单的早停类 (EarlyStopping) =====================
class EarlyStopping:
    """早停机制：当验证指标不再提升时停止训练"""

    def __init__(self, patience=cfg.patience if hasattr(cfg, 'patience') else 15, delta=0, verbose=True,
                 path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(cfg.save_dir, path)

    def __call__(self, val_score, model, epoch):
        # 假设 val_score 是 Accuracy (越大越好)
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存最佳模型'''
        if self.verbose:
            print(f'Validation score improved ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # 兼容 DataParallel
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.path))
        return model


# ===================== 训练主函数 =====================
def main():
    # 0. 参数解析 (可选)
    parser = argparse.ArgumentParser(description="ShipEar Training")
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    args = parser.parse_args()

    # 设置显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {args.gpu}")
    print(f"Config: {cfg.data_root}")

    # 1. 初始化真实数据集
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 2. 初始化模型
    print("Creating Model...")
    # 注意：这里不再需要 pretrained_cnn3d_path，因为我们从头训练音频 3D-CNN
    model = create_dual_transfer_model()

    # 统计参数
    if isinstance(model, nn.DataParallel):
        model.module.count_params(save_path=os.path.join(cfg.log_dir, "params_stats.json"))
    else:
        model.count_params(save_path=os.path.join(cfg.log_dir, "params_stats.json"))

    # 3. 初始化损失函数、优化器、调度器
    criterion = JointLoss(lambda_domain=0.1)  # 权重可调

    # 获取分组学习率 (如果模型提供了 get_learning_rates 接口)
    if hasattr(model, "module"):
        param_groups = model.module.get_learning_rates()
    elif hasattr(model, "get_learning_rates"):
        param_groups = model.get_learning_rates()
    else:
        param_groups = model.parameters()  # Fallback

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    # 学习率调度器 (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=15, verbose=True)
    metric_tracker = MetricTracker()

    # 4. 训练循环
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss_meter = []

        # 使用 tqdm 显示进度
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{cfg.epochs}] Train")

        for batch_idx, batch_data in enumerate(pbar):
            # 解包数据 (来自 ShipEarDataset)
            # data_3d: (B, 1, Depth, H, W) -> 模拟视频流输入
            # data_vit: (B, 3, 224, 224)   -> 模拟图像输入
            # label: (B,)
            x_3d = batch_data['data_3d'].to(cfg.device)
            x_2d = batch_data['data_vit'].to(cfg.device)
            y = batch_data['label'].to(cfg.device)

            # 构造虚拟 Domain Label (因为目前是单数据集训练)
            # 简单策略：一半设为0，一半设为1，强迫模型学习域无关特征 (Self-adversarial)
            B = x_3d.size(0)
            domain_label = torch.zeros(B, dtype=torch.long).to(cfg.device)
            domain_label[B // 2:] = 1

            optimizer.zero_grad()

            # 前向传播 (传入所有需要的输入)
            # V5 models.py return: (cls_out, domain_out)
            outputs = model(x_3d, x_2d=x_2d, domain_label=domain_label)

            # 解析输出
            if isinstance(outputs, tuple):
                cls_out, domain_out = outputs
            else:
                cls_out = outputs
                domain_out = None

            # 计算损失
            loss, loss_cls, loss_domain = criterion(
                cls_out=cls_out,
                target_cls=y,
                domain_out=domain_out,
                target_domain=domain_label
            )

            # 反向传播 + 梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg.clip_grad if hasattr(cfg, 'clip_grad') else 5.0)
            optimizer.step()

            train_loss_meter.append(loss.item())

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Cls': f"{loss_cls.item():.4f}",
                'Dom': f"{loss_domain.item():.4f}" if isinstance(loss_domain, torch.Tensor) else "0"
            })

        avg_train_loss = np.mean(train_loss_meter)

        # ==================== 验证阶段 ====================
        model.eval()
        metric_tracker.reset()
        val_loss_meter = []

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{cfg.epochs}] Val", leave=False):
                x_3d = batch_data['data_3d'].to(cfg.device)
                x_2d = batch_data['data_vit'].to(cfg.device)
                y = batch_data['label'].to(cfg.device)

                # 验证时不需要 domain_label
                outputs = model(x_3d, x_2d=x_2d, domain_label=None)

                # V5 models.py 推理时返回 (cls, rgb) 或 (cls, domain)
                if isinstance(outputs, tuple):
                    cls_out = outputs[0]
                else:
                    cls_out = outputs

                # 简单计算验证集的 cls loss
                loss_val = nn.CrossEntropyLoss()(cls_out, y)
                val_loss_meter.append(loss_val.item())

                # 更新评估指标
                metric_tracker.update(cls_out, y)

        # 计算验证指标
        avg_val_loss = np.mean(val_loss_meter)
        _, val_acc = metric_tracker.result()

        # 打印验证日志
        print(f"\nEpoch [{epoch + 1}/{cfg.epochs}] Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 打印详细分类报告 (每 5 个 epoch 或 最后一个 epoch)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.epochs:
            metric_tracker.print_report()

        # 更新学习率调度器
        scheduler.step()

        # 早停判断 (基于 Accuracy)
        early_stopping(val_acc, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # 5. 训练后处理
    print("Loading best model...")
    model = early_stopping.load_best_model(model)

    # 绘制最终的混淆矩阵
    print("Plotting confusion matrix...")
    # 重新跑一遍验证集以获取完整预测
    metric_tracker.reset()
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            x_3d = batch_data['data_3d'].to(cfg.device)
            x_2d = batch_data['data_vit'].to(cfg.device)
            y = batch_data['label'].to(cfg.device)
            outputs = model(x_3d, x_2d=x_2d)
            if isinstance(outputs, tuple):
                cls_out = outputs[0]
            else:
                cls_out = outputs
            metric_tracker.update(cls_out, y)

    metric_tracker.plot_confusion_matrix(save_path=os.path.join(cfg.log_dir, "confusion_matrix.png"))

    # 保存梯度监控 (如果启用了)
    if hasattr(model, "module"):
        try:
            model.module.save_grad_logs()
            model.module.plot_grad_curves()
            model.module.remove_grad_hooks()
        except:
            pass
    else:
        try:
            model.save_grad_logs()
            model.plot_grad_curves()
            model.remove_grad_hooks()
        except:
            pass


    print("\nAll Done! Results saved to:", cfg.log_dir)


if __name__ == "__main__":
    main()