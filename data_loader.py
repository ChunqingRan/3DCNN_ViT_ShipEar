import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import cv2
from config import cfg


class ShipEarDataset(Dataset):
    def __init__(self, metadata_path, split='train', transform=None):
        """
        ShipEar 数据集加载器 (适配 V5 models.py)
        Args:
            metadata_path: metadata.csv 的路径
            split: 'train' 或 'test'
        """
        # 读取 metadata
        try:
            df = pd.read_csv(metadata_path)
        except Exception as e:
            raise FileNotFoundError(f"无法读取 metadata.csv: {e}")

        # 筛选 train 或 test
        self.meta = df[df['split'] == split].reset_index(drop=True)

        self.data_root = os.path.dirname(metadata_path)
        self.class_map = {name: i for i, name in enumerate(cfg.class_names)}
        self.transform = transform

        # 预计算一些参数
        self.target_len = int(cfg.sample_rate * cfg.duration)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        wav_path = os.path.join(self.data_root, row['filepath'])
        label_str = row['class']
        label = self.class_map[label_str]

        # 1. 读取音频
        # 注意：librosa加载较慢，如果是大规模训练建议预存为 .npy 或 .pt
        y, sr = librosa.load(wav_path, sr=cfg.sample_rate, duration=cfg.duration)

        # 填充或截断
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        else:
            y = y[:self.target_len]

        # 2. 提取 Mel 语谱图
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Shape: (n_mels, time)

        # 归一化 (0~1)
        min_val = log_mel_spec.min()
        max_val = log_mel_spec.max()
        if max_val - min_val > 1e-5:
            norm_spec = (log_mel_spec - min_val) / (max_val - min_val)
        else:
            norm_spec = np.zeros_like(log_mel_spec)

        # ==========================================
        # 3. 构造 3D-CNN 输入 (伪3D: Channel=1, Depth=Time, H=Mel)
        # ==========================================
        # 原 models.py 期望 (Batch, 1, Depth, H, W)
        # 这里我们将 语谱图 视为 (1, Time, H, W=1) 或者 (1, Time, Mel)
        # 为了适配 EEGCNN3D 的卷积核，我们需要将语谱图 reshape。
        # 简单方案：将 Mel(128) reshape 为 (16, 8) 模拟 EEG 空间分布
        # Time 轴 resize 到 cfg.eeg_time_steps (224)

        # Resize Time 轴
        # 输入 cv2.resize 需要 (W, H) -> (Time, Mel)
        spec_resized = cv2.resize(norm_spec, (cfg.eeg_time_steps, cfg.n_mels))  # -> (128, 224) -> (Mel, Time)

        # Reshape Mel 为 Spatial (16, 8)
        # 结果 Shape: (16, 8, 224)
        # 我们需要 (Time, H, W) -> (224, 16, 8)

        # 转置为 (Time, Mel) -> (224, 128)
        spec_t = spec_resized.T
        # Reshape
        spatial_h, spatial_w = cfg.eeg_spatial_size  # (16, 8)
        spec_3d = spec_t.reshape(cfg.eeg_time_steps, spatial_h, spatial_w)  # (224, 16, 8)

        # 增加 Channel 维 -> (1, 224, 16, 8)
        # models.py 会在内部 unsqueeze 0 维 (Batch)，所以这里只需要 (C, D, H, W)
        img_3d = torch.FloatTensor(spec_3d).unsqueeze(0)

        # ==========================================
        # 4. 构造 ViT 输入 (RGB Image)
        # ==========================================
        # 将语谱图 resize 成 (224, 224)
        img_2d_raw = cv2.resize(norm_spec, (224, 224))
        # 堆叠成 3 通道
        img_2d = np.stack([img_2d_raw] * 3, axis=0)  # (3, 224, 224)
        img_2d = torch.FloatTensor(img_2d)

        return {
            "data_3d": img_3d,  # 用于 3D-CNN
            "data_vit": img_2d,  # 用于 ViT
            "label": torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders():
    train_ds = ShipEarDataset(cfg.metadata_path, split='train')
    test_ds = ShipEarDataset(cfg.metadata_path, split='test')

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size if hasattr(cfg, 'test_batch_size') else cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader