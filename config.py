import torch
import os


class Config:
    def __init__(self):
        # ================= 基础配置 =================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.num_workers = 4

        # ================= 路径配置 =================
        self.data_root = r"./ShipsEar_16k_30s_hop15"
        self.metadata_path = os.path.join(self.data_root, "metadata.csv")
        self.save_dir = "./checkpoints_v5"
        self.log_dir = "./logs_v5"

        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

        # ================= 数据参数 (映射逻辑) =================
        self.num_classes = 5
        self.class_names = ["ClassA", "ClassB", "ClassC", "ClassD", "ClassE"]

        # 音频基础参数
        self.sample_rate = 16000
        self.duration = 30
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 128

        # [关键] 维度映射参数 (适配 models.py 中的 EEG 变量名)
        # eeg_spatial_size: 原本是电极矩阵 (9,9) 等
        # 映射为 -> Mel 频带的重塑尺寸。128 个 Mel 频带可以 reshape 为 (16, 8)
        self.eeg_spatial_size = (16, 8)

        # eeg_time_steps: 模型期望的时间维度
        # 30s音频 @ 16kSR / 512hop ≈ 937 帧
        # 我们将在 DataLoader 中 resize 到 224 以适配 ViT 和 3D-CNN
        self.eeg_time_steps = 224

        # ================= 模型结构参数 =================
        self.vit_model_name = 'vit_base_patch16_224'
        # [新增] 本地权重路径 (请修改为您实际存放的路径)
        self.pretrained_path = r"./weights/vit_base_patch16_224.safetensors"
        self.cnn3d_in_channels = 1  # 音频只有1个通道 (Log-Mel)
        self.cnn3d_out_features = 256  # 3D-CNN 输出维度

        # 域适应相关
        self.domain_hidden_dim = 128
        self.num_domains = 2  # 比如 Train set=0, Test set=1，或者不同船型作为域

        # ================= 增强参数 (用于 EEGDataAugmentation) =================
        self.aug_prob = 0.5  # 增强概率
        self.noise_level = 0.05  # 注入噪声水平
        self.time_shift_range = 10  # 时间平移范围

        # ================= 梯度监控参数 =================
        self.grad_monitor_layers = ["cnn3d.conv3d_2", "vit.vit.blocks.10"]
        self.grad_logs_path = os.path.join(self.log_dir, "grad_logs.json")
        self.grad_curves_path = os.path.join(self.log_dir, "grad_curves.png")

        # ================= 训练超参数 =================
        self.batch_size = 32
        self.epochs = 50

        # 学习率配置 (用于 get_learning_rates)
        self.lr = 1e-4  # 全局基础 LR
        self.lr_cnn3d = 1e-4  # 3D-CNN LR
        self.lr_vit = 5e-5  # ViT LR (预训练模型通常用小一点的LR)
        self.weight_decay = 1e-4


cfg = Config()