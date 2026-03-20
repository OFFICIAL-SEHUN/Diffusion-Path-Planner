"""
패치 기반 Diffusion 추론 모듈
"""

import torch
import numpy as np
from scipy.interpolate import interp1d


def patch_based_inference(model, diffusion_scheduler, costmap_tensor, 
                         start_pos, goal_pos, img_size, horizon, 
                         num_patches=4, batch_size=8, device='cuda', show_progress=True):
    """
    패치 기반 Diffusion 추론 (병렬 배치 처리)
    
    Args:
        model: Diffusion 모델
        diffusion_scheduler: Diffusion 스케줄러
        costmap_tensor: [1, 2, H, W] costmap
        start_pos: 시작 위치 (row, col)
        goal_pos: 목표 위치 (row, col)
        img_size: 맵 크기
        horizon: 경로 waypoint 수
        num_patches: 한 축당 패치 개수 (4 = 4x4 = 16 patches)
        batch_size: 동시에 처리할 패치 개수 (병렬 처리)
        device: 디바이스
        show_progress: 진행률 표시
        
    Returns:
        generated_path: [horizon, 2] 생성된 경로 (픽셀 좌표)
    """
    print(f"\n{'='*60}")
    print(f"Patch-based Inference ({num_patches}x{num_patches} = {num_patches**2} patches)")
    print(f"Batch size: {batch_size} patches per inference")
    print(f"{'='*60}")
    
    patch_size = img_size // num_patches
    
    # waypoints_per_patch를 모델이 지원하는 크기로 조정
    raw_waypoints = horizon // num_patches**2
    waypoints_per_patch = ((raw_waypoints + 7) // 8) * 8
    waypoints_per_patch = max(waypoints_per_patch, 32)  # 최소 32
    
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Waypoints per patch: {waypoints_per_patch} (adjusted from {raw_waypoints})")
    
    # 전역 경로 계획: 시작->목표를 직선으로 연결하여 패치 순서 결정
    path_y = np.linspace(start_pos[0], goal_pos[0], num_patches**2 + 1)
    path_x = np.linspace(start_pos[1], goal_pos[1], num_patches**2 + 1)
    
    # 모든 패치 정보 수집
    patch_info_list = []
    
    for i in range(num_patches**2):
        current_start = (int(path_y[i]), int(path_x[i]))
        current_goal = (int(path_y[i+1]), int(path_x[i+1]))
        
        patch_row = current_start[0] // patch_size
        patch_col = current_start[1] // patch_size
        
        row_start = max(0, patch_row * patch_size)
        row_end = min(img_size, (patch_row + 1) * patch_size)
        col_start = max(0, patch_col * patch_size)
        col_end = min(img_size, (patch_col + 1) * patch_size)
        
        local_start = (current_start[0] - row_start, current_start[1] - col_start)
        local_goal = (current_goal[0] - row_start, current_goal[1] - col_start)
        
        patch_info_list.append({
            'index': i,
            'row_start': row_start,
            'row_end': row_end,
            'col_start': col_start,
            'col_end': col_end,
            'local_start': local_start,
            'local_goal': local_goal
        })
    
    all_waypoints = [None] * num_patches**2
    
    # 🔥 최적화: 모든 패치를 한 번에 처리 (메모리 허용 시)
    # 배치 크기를 전체 패치 수로 설정하여 한 번의 diffusion 호출로 처리
    total_patches = num_patches**2
    use_single_batch = (batch_size >= total_patches) or (total_patches <= 16)  # 16개 이하면 한 번에 처리
    
    if use_single_batch:
        # 모든 패치를 한 번에 처리 (가장 빠름)
        if show_progress:
            print(f"\nProcessing all {total_patches} patches in a single batch...")
        
        # 모든 패치용 텐서 준비
        all_costmaps = []
        all_starts = []
        all_goals = []
        
        for patch_info in patch_info_list:
            # Costmap 추출
            patch_costmap = costmap_tensor[:, :, 
                                          patch_info['row_start']:patch_info['row_end'],
                                          patch_info['col_start']:patch_info['col_end']]
            all_costmaps.append(patch_costmap)
            
            # 정규화된 좌표
            patch_actual_size = patch_costmap.shape[2]
            norm_start = np.array([patch_info['local_start'][1], patch_info['local_start'][0]], dtype=np.float32)
            norm_goal = np.array([patch_info['local_goal'][1], patch_info['local_goal'][0]], dtype=np.float32)
            norm_start = (norm_start / patch_actual_size) * 2 - 1
            norm_goal = (norm_goal / patch_actual_size) * 2 - 1
            
            all_starts.append(torch.from_numpy(norm_start).float())
            all_goals.append(torch.from_numpy(norm_goal).float())
        
        # 배치 텐서로 변환
        all_costmaps_tensor = torch.cat(all_costmaps, dim=0).to(device)  # [total_patches, 2, H, W]
        all_starts_tensor = torch.stack(all_starts).to(device)  # [total_patches, 2]
        all_goals_tensor = torch.stack(all_goals).to(device)  # [total_patches, 2]
        
        # 단일 배치 추론 (한 번의 diffusion 호출)
        with torch.no_grad():
            all_paths = diffusion_scheduler.sample(
                model=model,
                condition=all_costmaps_tensor,
                shape=(total_patches, waypoints_per_patch, 2),
                start_pos=all_starts_tensor,
                end_pos=all_goals_tensor,
                show_progress=show_progress
            )
        
        # 결과를 개별 패치로 분리하여 전역 좌표로 변환
        for idx, patch_info in enumerate(patch_info_list):
            patch_path_np = all_paths[idx].cpu().numpy()  # [waypoints_per_patch, 2] in [-1, 1]
            patch_actual_size = patch_info['row_end'] - patch_info['row_start']
            
            # [-1, 1] → [0, patch_size] → global coordinates
            patch_path_scaled = (patch_path_np + 1) / 2 * patch_actual_size
            patch_path_global = patch_path_scaled.copy()
            patch_path_global[:, 0] += patch_info['col_start']  # x (col)
            patch_path_global[:, 1] += patch_info['row_start']  # y (row)
            
            all_waypoints[patch_info['index']] = patch_path_global
        
        if show_progress:
            print(f"✓ All patches processed in single batch")
    else:
        # 배치 단위로 처리 (메모리 제한 시)
        num_batches = (total_patches + batch_size - 1) // batch_size
        
        if show_progress:
            print(f"\nProcessing {num_batches} batches (batch_size={batch_size})...")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_patches)
            batch_patches = patch_info_list[start_idx:end_idx]
            current_batch_size = len(batch_patches)
            
            # 배치용 텐서 준비
            batch_costmaps = []
            batch_starts = []
            batch_goals = []
            
            for patch_info in batch_patches:
                # Costmap 추출
                patch_costmap = costmap_tensor[:, :, 
                                              patch_info['row_start']:patch_info['row_end'],
                                              patch_info['col_start']:patch_info['col_end']]
                batch_costmaps.append(patch_costmap)
                
                # 정규화된 좌표
                patch_actual_size = patch_costmap.shape[2]
                norm_start = np.array([patch_info['local_start'][1], patch_info['local_start'][0]], dtype=np.float32)
                norm_goal = np.array([patch_info['local_goal'][1], patch_info['local_goal'][0]], dtype=np.float32)
                norm_start = (norm_start / patch_actual_size) * 2 - 1
                norm_goal = (norm_goal / patch_actual_size) * 2 - 1
                
                batch_starts.append(torch.from_numpy(norm_start).float())
                batch_goals.append(torch.from_numpy(norm_goal).float())
            
            # 배치 텐서로 변환
            batch_costmaps_tensor = torch.cat(batch_costmaps, dim=0).to(device)  # [B, 2, H, W]
            batch_starts_tensor = torch.stack(batch_starts).to(device)  # [B, 2]
            batch_goals_tensor = torch.stack(batch_goals).to(device)  # [B, 2]
            
            if show_progress:
                print(f"  Batch {batch_idx+1}/{num_batches}: Processing patches {start_idx+1}-{end_idx}...", end=' ')
            
            # 배치 추론
            with torch.no_grad():
                batch_paths = diffusion_scheduler.sample(
                    model=model,
                    condition=batch_costmaps_tensor,
                    shape=(current_batch_size, waypoints_per_patch, 2),
                    start_pos=batch_starts_tensor,
                    end_pos=batch_goals_tensor,
                    show_progress=False
                )
            
            # 결과를 개별 패치로 분리하여 전역 좌표로 변환
            for local_idx, patch_info in enumerate(batch_patches):
                patch_path_np = batch_paths[local_idx].cpu().numpy()  # [waypoints_per_patch, 2] in [-1, 1]
                patch_actual_size = patch_info['row_end'] - patch_info['row_start']
                
                # [-1, 1] → [0, patch_size] → global coordinates
                patch_path_scaled = (patch_path_np + 1) / 2 * patch_actual_size
                patch_path_global = patch_path_scaled.copy()
                patch_path_global[:, 0] += patch_info['col_start']  # x (col)
                patch_path_global[:, 1] += patch_info['row_start']  # y (row)
                
                all_waypoints[patch_info['index']] = patch_path_global
            
            if show_progress:
                print(f"✓")
    
    # 모든 패치 경로 연결
    full_path = np.vstack(all_waypoints)  # [total_waypoints, 2]
    
    # Resample to target horizon
    if full_path.shape[0] != horizon:
        t_current = np.linspace(0, 1, full_path.shape[0])
        t_target = np.linspace(0, 1, horizon)
        
        f_x = interp1d(t_current, full_path[:, 0], kind='linear')
        f_y = interp1d(t_current, full_path[:, 1], kind='linear')
        
        full_path = np.stack([f_x(t_target), f_y(t_target)], axis=1)
    
    # (x, y) → (row, col) for pixel coordinates
    path_pixels = [(int(p[1]), int(p[0])) for p in full_path]
    
    print(f"✓ Patch-based inference complete: {len(path_pixels)} waypoints")
    
    return path_pixels
