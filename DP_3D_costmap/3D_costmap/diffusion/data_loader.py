import torch
from torch.utils.data import Dataset
import os
from pathlib import Path

class GradientDataset(Dataset):
    """
    Slope + CoT 데이터셋 로더
    경사 기반 CoT 비용맵과 최적 경로를 로드합니다.
    """
    def __init__(self, config, load_auxiliary=False):
        """
        Args:
            config: 설정 객체
            load_auxiliary (bool): height_maps, slope_maps도 로드할지 여부
                                   학습에는 costmaps만 필요하지만, 
                                   시각화/분석 시 추가 맵이 필요할 수 있음
        """
        self.config = config
        self.load_auxiliary = load_auxiliary
        
        # 이 파일의 위치를 기준으로 절대 경로 생성
        current_dir = Path(__file__).parent
        self.data_dir = current_dir / "data"
        self.file_name = "dataset_gradient.pt"
        
        path = self.data_dir / self.file_name
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. "
                f"Run 'python generate_data.py' first to generate gradient terrain data!"
            )
            
        print(f"Loading gradient terrain dataset from {path}...")
        data = torch.load(str(path))
        
        # 데이터 타입 확인
        dataset_type = data.get("type", "unknown")
        if dataset_type not in ["gradient", "slope_cot", "slope_input_cot_gt", "slope_cot_2channel"]:
            print(f"Warning: Expected 'slope_cot_2channel' type, but got '{dataset_type}'")
        
        # 필수 데이터 (학습에 사용)
        self.costmaps = data["costmaps"]  # [N, 2, H, W] 2-channel: [Slope, CoT]
        self.paths = data["paths"]        # [N, Horizon, 2] CoT 기반 최적 경로 - GT
        
        # 추가 데이터 (선택적 - 시각화/분석용)
        if self.load_auxiliary:
            self.height_maps = data.get("height_maps", None)      # [N, H, W] 높이 (m)
            self.slope_maps = data.get("slope_maps", None)        # [N, H, W] 경사각 (도)
            
            print(f"Loaded {len(self.costmaps)} samples with auxiliary data.")
        else:
            self.height_maps = None
            self.slope_maps = None
            print(f"Loaded {len(self.costmaps)} samples (costmaps and paths only).")
        
        # 데이터 통계 출력
        print(f"  - Costmaps shape: {self.costmaps.shape} (2-channel: [Slope, CoT])")
        print(f"  - Paths shape: {self.paths.shape} (CoT-based GT)")
        if self.load_auxiliary and self.height_maps is not None:
            print(f"  - Height maps shape: {self.height_maps.shape}")
            print(f"  - Slope maps shape: {self.slope_maps.shape}")

    def __len__(self):
        return len(self.costmaps)

    def __getitem__(self, idx):
        """
        Returns:
            학습 모드 (load_auxiliary=False):
                costmap, path
            
            분석 모드 (load_auxiliary=True):
                {
                    'costmap': costmap,
                    'path': path,
                    'height_map': height_map,
                    'slope_map': slope_map
                }
        """
        costmap = self.costmaps[idx]
        path = self.paths[idx]
        
        if not self.load_auxiliary:
            # 학습용: 간단한 tuple 반환
            return costmap, path
        else:
            # 분석용: dictionary 반환
            result = {
                'costmap': costmap,
                'path': path
            }
            
            if self.height_maps is not None:
                result['height_map'] = self.height_maps[idx]
            if self.slope_maps is not None:
                result['slope_map'] = self.slope_maps[idx]
            
            return result


# 이전 버전과의 호환성을 위한 Alias
class FixedDataset(GradientDataset):
    """
    Deprecated: GradientDataset을 사용하세요.
    이전 코드와의 호환성을 위해 남겨둡니다.
    """
    def __init__(self, config):
        print("Warning: FixedDataset is deprecated. Use GradientDataset instead.")
        super().__init__(config, load_auxiliary=False)