import torch
from torch.utils.data import Dataset
import os
from pathlib import Path

class GradientDataset(Dataset):
    """
    Slope + Height 데이터셋 로더 (Text-conditioned)
    경사 맵과 높이 맵을 2채널로 로드하고, CoT 기반 최적 경로와 텍스트 라벨을 로드합니다.
    """
    def __init__(self, config, load_auxiliary=False):
        """
        Args:
            config: 설정 객체
            load_auxiliary (bool): 원본 height_maps, slope_maps도 로드할지 여부
                                   - False (기본): costmaps[Slope, Height]만 로드 → 학습용
                                   - True: 원본 height/slope maps도 로드 → 시각화/분석용
        """
        self.config = config
        self.load_auxiliary = load_auxiliary
        
        # 이 파일의 위치를 기준으로 절대 경로 생성
        current_dir = Path(__file__).parent
        self.data_dir = current_dir / "data"
        self.file_name = "test_dataset.pt"
        
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
        if dataset_type not in ["slope_height_2channel", "slope_height_2channel_text"]:
            print(f"Warning: Expected 'slope_height_2channel_text' type, but got '{dataset_type}'")
            print("Falling back to dataset without text labels")
        
        # 필수 데이터 (학습에 사용)
        self.costmaps = data["costmaps"]  # [N, 2, H, W] 2-channel: [정규화된 Slope, 정규화된 Height]
        self.paths = data["paths"]        # [N, Horizon, 2] CoT 기반 최적 경로 - GT
        
        # 🔥 텍스트 토큰 (text-conditioned)
        if "text_tokens" in data:
            self.text_tokens = data["text_tokens"]  # [N, max_seq_len] 텍스트 토큰
            self.vocab = data.get("vocab", {})      # Vocabulary 딕셔너리
            self.vocab_size = data.get("vocab_size", 1000)
            self.has_text = True
        else:
            # Fallback: 텍스트 없이도 작동하도록
            print("Warning: No text tokens found in dataset. Creating dummy tokens.")
            self.text_tokens = torch.zeros(len(self.costmaps), 32, dtype=torch.long)
            self.vocab = {"<PAD>": 0, "<UNK>": 1}
            self.vocab_size = 2
            self.has_text = False
        
        # 추가 데이터 (선택적 - 시각화/분석용)
        if self.load_auxiliary:
            self.height_maps = data.get("height_maps", None)      # [N, H, W] 원본 높이 (m 단위)
            self.slope_maps = data.get("slope_maps", None)        # [N, H, W] 원본 경사각 (도 단위)
            self.text_labels = data.get("text_labels", None)      # 원본 텍스트 라벨 리스트
            
            print(f"Loaded {len(self.costmaps)} samples with auxiliary data.")
        else:
            self.height_maps = None
            self.slope_maps = None
            self.text_labels = None
            print(f"Loaded {len(self.costmaps)} samples (costmaps, paths, and text tokens).")
        
        # 데이터 통계 출력
        print(f"  - Costmaps shape: {self.costmaps.shape} (2-channel: [Slope, Height])")
        print(f"  - Paths shape: {self.paths.shape} (CoT-based GT)")
        print(f"  - Text tokens shape: {self.text_tokens.shape}")
        print(f"  - Vocabulary size: {self.vocab_size}")
        if self.load_auxiliary and self.text_labels is not None:
            print(f"  - Text labels: {len(set(self.text_labels))} unique labels")
        if self.load_auxiliary and self.height_maps is not None:
            print(f"  - Height maps shape: {self.height_maps.shape}")
            print(f"  - Slope maps shape: {self.slope_maps.shape}")

    def __len__(self):
        return len(self.costmaps)

    def __getitem__(self, idx):
        """
        Returns:
            학습 모드 (load_auxiliary=False):
                (costmap, path, text_tokens)
                - costmap: [2, H, W] 정규화된 [Slope, Height]
                - path: [Horizon, 2] CoT 기반 최적 경로
                - text_tokens: [max_seq_len] 텍스트 토큰
            
            분석 모드 (load_auxiliary=True):
                {
                    'costmap': [2, H, W] 정규화된 [Slope, Height],
                    'path': [Horizon, 2] 경로,
                    'text_tokens': [max_seq_len] 텍스트 토큰,
                    'text_label': str 원본 텍스트 라벨,
                    'height_map': [H, W] 원본 높이(m),
                    'slope_map': [H, W] 원본 경사각(도)
                }
        """
        costmap = self.costmaps[idx]
        path = self.paths[idx]
        text_tokens = self.text_tokens[idx]  # [max_seq_len]
        
        if not self.load_auxiliary:
            # 학습용: tuple 반환 (costmap, path, text_tokens)
            return costmap, path, text_tokens
        else:
            # 분석용: dictionary 반환
            result = {
                'costmap': costmap,
                'path': path,
                'text_tokens': text_tokens
            }
            
            if self.text_labels is not None:
                result['text_label'] = self.text_labels[idx]
            if self.height_maps is not None:
                result['height_map'] = self.height_maps[idx]
            if self.slope_maps is not None:
                result['slope_map'] = self.slope_maps[idx]
            
            return result
