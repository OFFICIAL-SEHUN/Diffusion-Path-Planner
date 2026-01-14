import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import torch
from typing import Dict, Any, Optional

"""
Utility functions for path planning visualization and post-processing
"""

class SimpleKalmanFilter:
    """
    1D Kalman Filter for path smoothing
    
    경로의 노이즈를 제거하고 부드럽게 만들기 위한 간단한 칼만 필터
    각 좌표 차원(x, y)에 독립적으로 적용 가능
    """
    
    def __init__(self, 
                 process_noise: float, 
                 measurement_noise: float, 
                 initial_value: float = 0.0, 
                 initial_estimate_error: float = 1.0):
        """
        Args:
            process_noise (float): 프로세스 노이즈 (Q)
            measurement_noise (float): 측정 노이즈 (R)
            initial_value (float): 초기 상태 추정값
            initial_estimate_error (float): 초기 추정 오차
        """
        self.q = process_noise              # Process noise covariance
        self.r = measurement_noise          # Measurement noise covariance
        self.x = initial_value              # State estimate
        self.p = initial_estimate_error     # Estimate error covariance

    def update(self, measurement: float) -> float:
        """
        칼만 필터 업데이트
        
        Args:
            measurement (float): 새로운 측정값
            
        Returns:
            float: 필터링된 추정값
        """
        # Kalman gain
        k = self.p / (self.p + self.r)
        
        # Update estimate
        self.x = self.x + k * (measurement - self.x)
        
        # Update error covariance
        self.p = (1 - k) * self.p
        
        return self.x

def plot_results(costmap: torch.Tensor, 
                 generated_path: torch.Tensor, 
                 true_path: Optional[torch.Tensor], 
                 config: Dict[str, Any],
                 slope_map: Optional[np.ndarray] = None):
    """
    Slope + CoT map과 경로를 시각화하고 저장
    
    Features:
    - Slope map 표시 (Diffusion 입력은 2채널: Slope + CoT)
    - 생성된 경로와 CoT 기반 GT 경로 비교
    - Kalman filter로 경로 smoothing (선택적)
    - 결과를 results/ 폴더에 저장
    
    Args:
        costmap (torch.Tensor): 2채널 map [1, 2, H, W] - [Slope, CoT]
        generated_path (torch.Tensor): Diffusion 모델이 생성한 경로 [1, Horizon, 2] 또는 [Horizon, 2]
        true_path (torch.Tensor, optional): CoT 기반 GT 경로 [1, Horizon, 2] 또는 [Horizon, 2]
        config (dict): 설정 딕셔너리 (img_size 등)
        slope_map (np.ndarray, optional): 라디안 단위 slope map (시각화용)
    """
    img_size = config['data']['img_size']
    
    # === 1. Tensor to NumPy ===
    costmap_np = costmap.squeeze().cpu().numpy()
    gen_path_norm = generated_path.squeeze().cpu().numpy()
    
    if true_path is not None:
        true_path_norm = true_path.squeeze().cpu().numpy()
    else:
        true_path_norm = np.array([])  # Empty array

    # === 2. Denormalize paths: [-1, 1] → [0, img_size] ===
    gen_path_scaled = (gen_path_norm + 1) / 2 * img_size
    
    if true_path_norm.size > 0:
        true_path_scaled = (true_path_norm + 1) / 2 * img_size
    else:
        true_path_scaled = np.array([])
    
    # Handle NaN values in path range output
    if np.isnan(gen_path_scaled).any():
        print("Warning: Generated path contains NaN values")
    else:
        print(f"Generated path range: x=[{gen_path_scaled[:, 0].min():.1f}, {gen_path_scaled[:, 0].max():.1f}], "
              f"y=[{gen_path_scaled[:, 1].min():.1f}, {gen_path_scaled[:, 1].max():.1f}]")
    
    # === 3. Apply Kalman Filter Smoothing (Optional) ===
    kf_x = SimpleKalmanFilter(
        process_noise=0.01, 
        measurement_noise=0.7, 
        initial_value=gen_path_scaled[0, 0]
    )
    kf_y = SimpleKalmanFilter(
        process_noise=0.01, 
        measurement_noise=0.7, 
        initial_value=gen_path_scaled[0, 1]
    )
    
    smoothed_path = np.zeros_like(gen_path_scaled)
    for i in range(len(gen_path_scaled)):
        smoothed_path[i, 0] = kf_x.update(gen_path_scaled[i, 0])
        smoothed_path[i, 1] = kf_y.update(gen_path_scaled[i, 1])
    
    # === 4. Plotting ===
    plt.figure(figsize=(10, 10))
    
    # Choose visualization: slope_map (preferred) or costmap
    if slope_map is not None:
        # Visualize slope map in degrees (like gradient_map_generator.py)
        cmap = cm.get_cmap('jet').copy()
        cmap.set_bad(color='black')
        slope_degrees = np.degrees(slope_map)
        masked_map = np.ma.masked_invalid(slope_degrees)
        plt.imshow(masked_map, cmap=cmap, origin='lower')
        colorbar_label = "Slope Angle (Degrees)"
        vmin, vmax = None, None  # Auto scale
    else:
        # Fallback to normalized slope map visualization
        # ✅ costmap_np는 [2, H, W] - Channel 0: Slope, Channel 1: CoT
        cmap = cm.get_cmap('jet').copy()
        cmap.set_bad(color='black')
        slope_channel = costmap_np[0]  # Channel 0: Slope map
        masked_map = np.ma.masked_invalid(slope_channel)
        # slope_channel는 정규화된 slope (0~1), 0~90도로 변환하여 표시
        slope_degrees_vis = slope_channel * 90.0
        plt.imshow(slope_degrees_vis, cmap=cmap, origin='lower', vmin=0, vmax=30)
        colorbar_label = "Slope Angle (Degrees)"
        vmin, vmax = 0, 30
    
    # Plot ground truth path (if available)
    # ✅ 시각화만 반전: x,y 좌표 swap
    if true_path_scaled.size > 0:
        plt.plot(true_path_scaled[:, 0], true_path_scaled[:, 1], 
                'r--', linewidth=3, alpha=0.8, label='Ground Truth (A*)')
    
    # Plot generated path
    plt.plot(gen_path_scaled[:, 0], gen_path_scaled[:, 1], 
            'c-', linewidth=3, alpha=0.9, label='Generated Path (Diffusion)')
    
    # Optional: Plot smoothed path (commented out by default)
    # plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 
    #         'g-', linewidth=2, alpha=0.7, label='Smoothed Path (Kalman)')
    
    # Mark start and end points
    if true_path_scaled.size > 0:
        # Use ground truth start/end
        plt.scatter(true_path_scaled[0, 0], true_path_scaled[0, 1], 
                   c='cyan', marker='o', s=150, edgecolors='black', 
                   linewidths=2, label='Start', zorder=10)
        plt.scatter(true_path_scaled[-1, 0], true_path_scaled[-1, 1], 
                   c='yellow', marker='*', s=200, edgecolors='black', 
                   linewidths=2, label='Goal', zorder=10)
    else:
        # Fallback to generated path start/end
        plt.scatter(gen_path_scaled[0, 0], gen_path_scaled[0, 1], 
                   c='cyan', marker='o', s=150, edgecolors='black', 
                   linewidths=2, label='Start', zorder=10)
        plt.scatter(gen_path_scaled[-1, 0], gen_path_scaled[-1, 1], 
                   c='yellow', marker='*', s=200, edgecolors='black', 
                   linewidths=2, label='Goal', zorder=10)
    
    # Configure plot
    plt.legend(loc='upper right', fontsize=10)
    plt.title("Diffusion Path Planning (Model Input: Slope + CoT | Visualization: Slope Map)", 
             fontsize=12, fontweight='bold')
    plt.xlim(0, img_size)
    plt.ylim(0, img_size)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Colorbar - specify the axes to use
    ax = plt.gca()  # Get current axes
    if slope_map is not None:
        # Auto-scale for slope map
        im = ax.images[0]
        cbar = plt.colorbar(im, ax=ax, label=colorbar_label)
    else:
        # Fixed scale for costmap
        norm = plt.Normalize(vmin=0, vmax=1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=colorbar_label)
    
    # === 5. Save Figure ===
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(results_dir, 'diffusion_path_plan.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    plt.close()  # Free memory
