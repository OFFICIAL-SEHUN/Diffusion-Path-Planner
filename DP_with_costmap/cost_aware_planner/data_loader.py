import torch
from torch.utils.data import Dataset
import os

class FixedDataset(Dataset):
    """
    Loads a pre-generated dataset from a .pt file.
    Faster training, but fixed diversity.
    """
    def __init__(self, config):
        self.config = config
        self.data_dir = "data" # 데이터가 저장된 폴더
        self.file_name = "dataset.pt"
        
        path = os.path.join(self.data_dir, self.file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}. Run generate_data.py first!")
            
        print(f"Loading dataset from {path}...")
        data = torch.load(path)
        
        self.costmaps = data["costmaps"] # [N, 64, 64]
        self.paths = data["paths"]       # [N, Horizon, 2]
        
        print(f"Loaded {len(self.costmaps)} samples.")

    def __len__(self):
        return len(self.costmaps)

    def __getitem__(self, idx):
        costmap = self.costmaps[idx]
        path = self.paths[idx]
        
        # 모델이 Flatten을 원한다면 Trainer나 Model에서 처리하므로
        # 여기서는 Raw Tensor 그대로 넘깁니다.
        return costmap, path