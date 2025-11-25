import os
import torch
import numpy as np
from tqdm import tqdm
from maze import MazeGenerator

"""
This code call "maze.py" to generate maze and path.
And save the data to a .pt file.
"""

# --- 설정 ---
NUM_SAMPLES = 30000      # 생성할 데이터 개수 (원하는 만큼 조절)
IMG_SIZE = 64            # 맵 크기
HORIZON = 128            # 경로 길이 (모델 입력 크기)
SCALE = 4                # Maze Scale
COST_WEIGHT = 15.0       # Cost-aware A* 가중치 (main.py와 유사한 값 사용)
SAVE_DIR = "data"        # 저장할 폴더
SAVE_NAME = "dataset.pt" # 저장할 파일명

def generate_and_save():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    costmaps_list = []
    paths_list = []

    print(f"Generating {NUM_SAMPLES} samples...")

    for _ in tqdm(range(NUM_SAMPLES)):
        # 1. 미로 생성
        maze_gen = MazeGenerator(IMG_SIZE, SCALE)
        costmap, path, start, end = maze_gen.generate(cost_weight=COST_WEIGHT)

        # 2. Costmap 전처리 (Inf 처리 및 정규화)
        # generate()에서 path가 None일 경우 재시도하므로, 여기서는 costmap만 처리
        costmap_float = costmap.astype(np.float32)
        inf_mask = np.isinf(costmap_float)
        
        if (~inf_mask).any():
            max_val = np.max(costmap_float[~inf_mask])
        else:
            max_val = 1.0
            
        costmap_float[inf_mask] = max_val
        # 안전한 나눗셈
        costmap_norm = costmap_float / (np.max(costmap_float) + 1e-8)
        
        # [1, 64, 64] 형태로 저장 (나중에 모델이 알아서 Flatten 하거나 처리)
        costmaps_list.append(costmap_norm)

        # 3. Path 리샘플링 (길이 맞추기: Variable -> Fixed Horizon)
        original_len = path.shape[0]
        if original_len == 0:
            path_fixed = np.zeros((HORIZON, 2), dtype=np.float32)
        else:
            t_current = np.linspace(0, 1, original_len)
            t_target = np.linspace(0, 1, HORIZON)
            
            x_interp = np.interp(t_target, t_current, path[:, 0])
            y_interp = np.interp(t_target, t_current, path[:, 1])
            path_fixed = np.stack([x_interp, y_interp], axis=1)

        paths_list.append(path_fixed)

    # 4. 텐서로 변환 및 저장
    # costmaps: [N, 64, 64] -> [N, 1, 64, 64] (채널 차원 추가가 필요할 수 있음, 여기선 그대로 둠)
    costmaps_tensor = torch.from_numpy(np.array(costmaps_list)).float()
    paths_tensor = torch.from_numpy(np.array(paths_list)).float()

    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    torch.save({
        "costmaps": costmaps_tensor,
        "paths": paths_tensor
    }, save_path)

    print(f"Dataset saved to {save_path}")
    print(f"Costmaps shape: {costmaps_tensor.shape}")
    print(f"Paths shape: {paths_tensor.shape}")
    print(f"File size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    generate_and_save()