import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import torch
from typing import Dict, Any, Optional, List
from matplotlib.gridspec import GridSpec

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
                 slope_map: Optional[np.ndarray] = None,
                 height_map: Optional[np.ndarray] = None):
    """
    Slope + Height map과 경로를 시각화하고 저장
    
    Features:
    - Slope map과 Height map을 함께 표시
    - 생성된 경로와 CoT 기반 GT 경로 비교
    - Kalman filter로 경로 smoothing (선택적)
    - 결과를 results/ 폴더에 저장
    
    Args:
        costmap (torch.Tensor): 2채널 map [1, 2, H, W] - [Slope, Height]
        generated_path (torch.Tensor): Diffusion 모델이 생성한 경로 [1, Horizon, 2] 또는 [Horizon, 2]
        true_path (torch.Tensor, optional): CoT 기반 GT 경로 [1, Horizon, 2] 또는 [Horizon, 2]
        config (dict): 설정 딕셔너리 (img_size 등)
        slope_map (np.ndarray, optional): 라디안 단위 slope map (시각화용)
        height_map (np.ndarray, optional): 높이 맵 (시각화용)
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
    # Create 1x2 subplot for Slope and Height maps
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # --- Left: Slope Map with Paths ---
    ax_slope = axes[0]
    
    # Choose visualization: slope_map (preferred) or costmap
    if slope_map is not None:
        # Visualize slope map in degrees
        cmap_slope = cm.get_cmap('jet').copy()
        cmap_slope.set_bad(color='black')
        slope_degrees = np.degrees(slope_map)
        masked_slope = np.ma.masked_invalid(slope_degrees)
        im_slope = ax_slope.imshow(masked_slope, cmap=cmap_slope, origin='lower', vmin=0, vmax=35)
    else:
        # Fallback to normalized slope map visualization
        cmap_slope = cm.get_cmap('jet').copy()
        cmap_slope.set_bad(color='black')
        slope_channel = costmap_np[0]  # Channel 0: Slope map
        slope_degrees_vis = slope_channel * 90.0
        masked_slope = np.ma.masked_invalid(slope_degrees_vis)
        im_slope = ax_slope.imshow(masked_slope, cmap=cmap_slope, origin='lower', vmin=0, vmax=35)
    
    # --- Right: Height Map with Paths ---
    ax_height = axes[1]
    
    if height_map is not None:
        # Visualize height map
        cmap_height = cm.get_cmap('terrain').copy()
        cmap_height.set_bad(color='black')
        masked_height = np.ma.masked_invalid(height_map)
        im_height = ax_height.imshow(masked_height, cmap=cmap_height, origin='lower')
    else:
        # Fallback to normalized height from costmap
        height_channel = costmap_np[1]  # Channel 1: Height map
        cmap_height = cm.get_cmap('terrain').copy()
        masked_height = np.ma.masked_invalid(height_channel)
        im_height = ax_height.imshow(masked_height, cmap=cmap_height, origin='lower')
    
    # Plot paths on both subplots
    for ax in [ax_slope, ax_height]:
        # Plot ground truth path (if available)
        if true_path_scaled.size > 0:
            ax.plot(true_path_scaled[:, 0], true_path_scaled[:, 1], 
                    'r--', linewidth=3, alpha=0.8, label='Ground Truth (A*)')
        
        # Plot generated path
        ax.plot(gen_path_scaled[:, 0], gen_path_scaled[:, 1], 
                'c-', linewidth=3, alpha=0.9, label='Generated Path (Diffusion)')
        
        # Mark start and end points
        if true_path_scaled.size > 0:
            # Use ground truth start/end
            ax.scatter(true_path_scaled[0, 0], true_path_scaled[0, 1], 
                       c='cyan', marker='o', s=150, edgecolors='black', 
                       linewidths=2, label='Start', zorder=10)
            ax.scatter(true_path_scaled[-1, 0], true_path_scaled[-1, 1], 
                       c='yellow', marker='*', s=200, edgecolors='black', 
                       linewidths=2, label='Goal', zorder=10)
        else:
            # Fallback to generated path start/end
            ax.scatter(gen_path_scaled[0, 0], gen_path_scaled[0, 1], 
                       c='cyan', marker='o', s=150, edgecolors='black', 
                       linewidths=2, label='Start', zorder=10)
            ax.scatter(gen_path_scaled[-1, 0], gen_path_scaled[-1, 1], 
                       c='yellow', marker='*', s=200, edgecolors='black', 
                       linewidths=2, label='Goal', zorder=10)
        
        # Configure each subplot
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
    
    # Titles
    ax_slope.set_title("Slope Map (Input Channel 0)", fontsize=14, fontweight='bold')
    ax_height.set_title("Height Map (Input Channel 1)", fontsize=14, fontweight='bold')
    
    # Colorbars
    cbar_slope = plt.colorbar(im_slope, ax=ax_slope, label="Slope Angle (Degrees)")
    cbar_height = plt.colorbar(im_height, ax=ax_height, label="Height (m)")
    
    # Overall title
    fig.suptitle("Diffusion Path Planning: Slope + Height 2-Channel Input", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # === 5. Save Figure ===
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(results_dir, 'diffusion_path_plan.png')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    plt.close()  # Free memory


def visualize_diffusion_steps(intermediates: List[Dict],
                               costmap: torch.Tensor,
                               config: Dict[str, Any],
                               slope_map: Optional[np.ndarray] = None,
                               max_steps_to_show: int = 12):
    """
    Diffusion 생성 과정의 중간 step들을 시각화
    
    Args:
        intermediates (list): diffusion sample()에서 반환한 중간 결과 리스트
        costmap (torch.Tensor): 2채널 map [1, 2, H, W] - [Slope, Height]
        config (dict): 설정 딕셔너리
        slope_map (np.ndarray, optional): 라디안 단위 slope map (시각화용)
        max_steps_to_show (int): 표시할 최대 step 수 (기본 12개)
    """
    img_size = config['data']['img_size']
    
    # 표시할 step 선택 (너무 많으면 균등하게 샘플링)
    if len(intermediates) > max_steps_to_show:
        indices = np.linspace(0, len(intermediates) - 1, max_steps_to_show, dtype=int)
        steps_to_show = [intermediates[i] for i in indices]
    else:
        steps_to_show = intermediates
    
    # Costmap 준비
    costmap_np = costmap.squeeze().cpu().numpy()
    
    # Slope map 시각화 준비
    if slope_map is not None:
        slope_degrees = np.degrees(slope_map)
        cmap = cm.get_cmap('jet').copy()
        cmap.set_bad(color='black')
        masked_map = np.ma.masked_invalid(slope_degrees)
    else:
        slope_channel = costmap_np[0]
        slope_degrees = slope_channel * 90.0
        cmap = cm.get_cmap('jet').copy()
        masked_map = slope_degrees
    
    # Grid layout: 3 or 4 columns
    n_steps = len(steps_to_show)
    n_cols = 4
    n_rows = int(np.ceil(n_steps / n_cols))
    
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, step_data in enumerate(steps_to_show):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Plot terrain
        ax.imshow(masked_map, cmap=cmap, origin='lower', vmin=0, vmax=30)
        
        # Get path and denormalize
        path_norm = step_data['path'].squeeze().numpy()  # [Horizon, 2]
        path_scaled = (path_norm + 1) / 2 * img_size
        
        # Plot path
        ax.plot(path_scaled[:, 0], path_scaled[:, 1], 
               'c-', linewidth=2, alpha=0.8)
        
        # Mark start and end
        ax.scatter(path_scaled[0, 0], path_scaled[0, 1],
                  c='cyan', marker='o', s=80, edgecolors='black',
                  linewidths=1.5, zorder=10)
        ax.scatter(path_scaled[-1, 0], path_scaled[-1, 1],
                  c='yellow', marker='*', s=120, edgecolors='black',
                  linewidths=1.5, zorder=10)
        
        # Title with timestep info
        timestep = step_data['timestep']
        noise_level = step_data['noise_level']
        ax.set_title(f"t={timestep} (noise={noise_level:.3f})", 
                    fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.axis('off')
    
    # Overall title
    fig.suptitle("Diffusion Path Generation Process (Step-by-Step)", 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'diffusion_steps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Step-by-step visualization saved to: {save_path}")
    
    plt.close()


def create_diffusion_animation(intermediates: List[Dict],
                                costmap: torch.Tensor,
                                config: Dict[str, Any],
                                slope_map: Optional[np.ndarray] = None,
                                save_individual_frames: bool = True):
    """
    Diffusion 과정을 개별 프레임으로 저장 (GIF 생성 가능)
    
    Args:
        intermediates (list): diffusion sample()에서 반환한 중간 결과 리스트
        costmap (torch.Tensor): 2채널 map [1, 2, H, W]
        config (dict): 설정 딕셔너리
        slope_map (np.ndarray, optional): slope map
        save_individual_frames (bool): 개별 프레임 저장 여부
    """
    img_size = config['data']['img_size']
    results_dir = 'results/diffusion_frames'
    os.makedirs(results_dir, exist_ok=True)
    
    # Costmap 준비
    costmap_np = costmap.squeeze().cpu().numpy()
    
    if slope_map is not None:
        slope_degrees = np.degrees(slope_map)
        cmap = cm.get_cmap('jet').copy()
        cmap.set_bad(color='black')
        masked_map = np.ma.masked_invalid(slope_degrees)
    else:
        slope_channel = costmap_np[0]
        slope_degrees = slope_channel * 90.0
        cmap = cm.get_cmap('jet').copy()
        masked_map = slope_degrees
    
    # 각 step을 개별 이미지로 저장
    for idx, step_data in enumerate(intermediates):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot terrain
        ax.imshow(masked_map, cmap=cmap, origin='lower', vmin=0, vmax=30)
        
        # Get path
        path_norm = step_data['path'].squeeze().numpy()
        path_scaled = (path_norm + 1) / 2 * img_size
        
        # Plot path
        ax.plot(path_scaled[:, 0], path_scaled[:, 1],
               'c-', linewidth=3, alpha=0.9)
        
        # Mark start and end
        ax.scatter(path_scaled[0, 0], path_scaled[0, 1],
                  c='cyan', marker='o', s=150, edgecolors='black',
                  linewidths=2, zorder=10)
        ax.scatter(path_scaled[-1, 0], path_scaled[-1, 1],
                  c='yellow', marker='*', s=200, edgecolors='black',
                  linewidths=2, zorder=10)
        
        # Title
        timestep = step_data['timestep']
        noise_level = step_data['noise_level']
        ax.set_title(f"Diffusion Step: t={timestep} (noise level={noise_level:.4f})",
                    fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Save frame
        frame_path = os.path.join(results_dir, f'frame_{idx:04d}_t{timestep:04d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(intermediates)} frames to: {results_dir}")
    print(f"To create GIF: convert -delay 10 -loop 0 {results_dir}/frame_*.png diffusion_animation.gif")
