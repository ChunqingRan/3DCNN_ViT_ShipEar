import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from config import cfg


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []
        self.losses = []

    def update(self, output, target, loss=None):
        """
        output: 可能是 Tensor (cls_only) 或者 Tuple (cls, domain)
        """
        # --- 适配 V5 models.py 的返回格式 ---
        if isinstance(output, tuple):
            # 训练时返回 (cls, domain)，我们只关心 cls 用于计算准确率
            # 推理时返回 (cls, rgb)，同样取第一个
            cls_output = output[0]
        else:
            cls_output = output

        # 获取预测类别
        _, predicted = torch.max(cls_output, 1)

        self.preds.extend(predicted.cpu().detach().numpy())
        self.targets.extend(target.cpu().detach().numpy())

        if loss is not None:
            # 如果是 Tensor (单GPU) 提取item，如果是 list (多GPU) 取均值
            if torch.is_tensor(loss):
                self.losses.append(loss.item())
            else:
                self.losses.append(loss)

    def result(self):
        acc = accuracy_score(self.targets, self.preds) * 100.0
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        return avg_loss, acc

    def print_report(self):
        print("\n" + "=" * 30)
        print(classification_report(
            self.targets,
            self.preds,
            target_names=cfg.class_names,
            digits=4,
            zero_division=0
        ))
        print("=" * 30)

    def plot_confusion_matrix(self, save_path=None):
        """绘制并保存混淆矩阵"""
        if not self.targets or not self.preds:
            print("Warning: No predictions to plot. Skipping confusion matrix generation.")
            return
        
        cm = confusion_matrix(self.targets, self.preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=cfg.class_names,
                    yticklabels=cfg.class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()