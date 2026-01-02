import os
import torch
import numpy as np
import random
import sys

# 导入配置和训练入口
from config import cfg
from train import main as start_training


def set_seed(seed=42):
    """设置全局随机种子以保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 保证 CuDNN 的确定性行为 (会稍微降低性能，但保证结果一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")


def check_environment():
    """检查必要的环境和路径"""
    print("=" * 30)
    print("Environment Check:")
    print(f"  - Python: {sys.version.split()[0]}")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - Device: {cfg.device}")

    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")

    print(f"  - Data Root: {cfg.data_root}")
    print(f"  - Metadata: {cfg.metadata_path}")
    print("=" * 30)

    # 检查数据目录是否存在
    if not os.path.exists(cfg.data_root):
        print(f"❌ Error: Data root not found at {cfg.data_root}")
        print("   Please check 'config.py' or run the preprocessing script first.")
        sys.exit(1)

    # 检查 metadata 是否存在
    if not os.path.exists(cfg.metadata_path):
        print(f"❌ Error: Metadata file not found at {cfg.metadata_path}")
        print("   Please ensure your preprocessing script generated 'metadata.csv'.")
        sys.exit(1)


if __name__ == "__main__":
    # 1. 环境初始化
    check_environment()
    set_seed(cfg.seed)

    # 2. 创建日志目录
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
        print(f"✅ Created log directory: {cfg.log_dir}")

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
        print(f"✅ Created checkpoint directory: {cfg.save_dir}")

    # 3. 启动训练
    print("\n🚀 Starting Training Pipeline...")
    try:
        start_training()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")
        raise e