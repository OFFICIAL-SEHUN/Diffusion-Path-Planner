import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
from pathlib import Path

"""
Slope + CoT 데이터셋 시각화 도구
높이, 경사, CoT 비용 맵과 경로를 시각화합니다.
"""

class SlopeCotDataVisualizer:
    """Slope + CoT 데이터를 시각화하는 클래스"""
    
    def __init__(self, data_path=None):
        """
        Args:
            data_path (str): 데이터셋 파일 경로 (None이면 상위 디렉토리의 data/dataset_gradient.pt 사용)
        """
        # data_path가 None이면 상위 디렉토리의 data 폴더 사용
        if data_path is None:
            data_path = str(Path(__file__).parent.parent / 'data' / 'dataset_gradient.pt')
        
        self.data_path = data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        print(f"Loading dataset from {data_path}...")
        self.data = torch.load(data_path)
        
        self.costmaps = self.data["costmaps"].numpy()  # [N, 2, H, W]
        self.paths = self.data["paths"].numpy()
        self.height_maps = self.data["height_maps"].numpy()
        self.slope_maps = self.data["slope_maps"].numpy()
        
        # 2채널에서 각 채널 분리
        if self.costmaps.ndim == 4 and self.costmaps.shape[1] == 2:
            self.slope_channel = self.costmaps[:, 0, :, :]  # Channel 0: Slope (정규화)
            self.cot_channel = self.costmaps[:, 1, :, :]     # Channel 1: CoT (정규화)
            print(f"✓ Detected 2-channel format")
        else:
            # 이전 형식 호환성
            self.slope_channel = None
            self.cot_channel = self.costmaps
            print(f"⚠ Old format detected")
        
        self.num_samples = len(self.costmaps)
        print(f"Loaded {self.num_samples} samples")
        print(f"  - Costmaps: {self.costmaps.shape} (2-channel: [Slope, CoT])")
        print(f"  - Paths: {self.paths.shape}")
        print(f"  - Height maps: {self.height_maps.shape}")
        print(f"  - Slope maps: {self.slope_maps.shape}")
    
    def visualize_sample(self, idx=0, save_path=None):
        """
        단일 샘플의 모든 맵과 경로를 시각화합니다.
        
        Args:
            idx (int): 샘플 인덱스
            save_path (str): 저장 경로 (None이면 화면에만 표시)
        """
        if idx >= self.num_samples:
            raise ValueError(f"Index {idx} out of range (max: {self.num_samples-1})")
        
        # 데이터 추출
        height_map = self.height_maps[idx]
        slope_map = self.slope_maps[idx]  # 도 단위 (저장된 값)
        
        # 2채널에서 각 맵 추출
        if self.slope_channel is not None:
            slope_norm = self.slope_channel[idx]  # 정규화된 slope (0-1)
            cot_norm = self.cot_channel[idx]      # 정규화된 CoT (0-1)
            
            # 시각화용으로 역정규화
            slope_vis = slope_norm * 90.0  # [0, 1] → [0, 90°]
            cot_vis = cot_norm
        else:
            # 이전 형식 호환성
            slope_vis = slope_map
            cot_vis = self.costmaps[idx]
        
        path = self.paths[idx]  # [HORIZON, 2] 범위 [-1, 1]
        
        # 경로를 픽셀 좌표로 변환
        img_size = height_map.shape[0]
        path_pixels = ((path + 1) / 2) * img_size
        
        # 2x2 서브플롯 생성
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Height Map
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(height_map, cmap='terrain', origin='lower')
        ax1.plot(path_pixels[:, 1], path_pixels[:, 0], 'r-', linewidth=2, label='Path')
        ax1.scatter(path_pixels[0, 1], path_pixels[0, 0], c='cyan', s=200, 
                   edgecolors='black', linewidths=2, marker='o', label='Start', zorder=10)
        ax1.scatter(path_pixels[-1, 1], path_pixels[-1, 0], c='yellow', s=200,
                   edgecolors='black', linewidths=2, marker='*', label='Goal', zorder=10)
        plt.colorbar(im1, ax=ax1, label='Height (m)')
        ax1.set_title(f'Height Map (Sample {idx})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # 2. Slope Map (Channel 0)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(slope_vis, cmap='jet', origin='lower', vmin=0, vmax=30)
        ax2.plot(path_pixels[:, 1], path_pixels[:, 0], 'w-', linewidth=2, label='Path')
        ax2.scatter(path_pixels[0, 1], path_pixels[0, 0], c='cyan', s=200,
                   edgecolors='black', linewidths=2, marker='o', zorder=10)
        ax2.scatter(path_pixels[-1, 1], path_pixels[-1, 0], c='yellow', s=200,
                   edgecolors='black', linewidths=2, marker='*', zorder=10)
        plt.colorbar(im2, ax=ax2, label='Slope Angle (°)')
        ax2.set_title('Slope Map (Ch 0)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        # 3. CoT Cost Map (Channel 1)
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(cot_vis, cmap='hot', origin='lower', vmin=0, vmax=1)
        ax3.plot(path_pixels[:, 1], path_pixels[:, 0], 'c-', linewidth=2, label='Path')
        ax3.scatter(path_pixels[0, 1], path_pixels[0, 0], c='cyan', s=200,
                   edgecolors='black', linewidths=2, marker='o', zorder=10)
        ax3.scatter(path_pixels[-1, 1], path_pixels[-1, 0], c='yellow', s=200,
                   edgecolors='black', linewidths=2, marker='*', zorder=10)
        plt.colorbar(im3, ax=ax3, label='Normalized CoT Cost')
        ax3.set_title('CoT Cost Map (Ch 1)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        
        # 4. 3D Height Map with Path
        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        x = np.arange(img_size)
        y = np.arange(img_size)
        X, Y = np.meshgrid(x, y)
        surf = ax4.plot_surface(X, Y, height_map, cmap='terrain', alpha=0.7, 
                               edgecolor='none', antialiased=True)
        # 경로를 3D로 표시 (높이는 height_map에서 가져옴)
        path_heights = []
        for i in range(len(path_pixels)):
            px, py = int(np.clip(path_pixels[i, 0], 0, img_size-1)), \
                     int(np.clip(path_pixels[i, 1], 0, img_size-1))
            path_heights.append(height_map[px, py])
        ax4.plot(path_pixels[:, 1], path_pixels[:, 0], path_heights, 
                'r-', linewidth=3, label='Path')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Height (m)')
        ax4.set_title('3D Terrain with Path', fontsize=14, fontweight='bold')
        ax4.view_init(elev=30, azim=45)
        
        plt.suptitle(f'Slope + CoT Terrain Visualization - Sample {idx}', 
                    fontsize=16, y=0.98, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def visualize_multiple(self, num_samples=4, save_path=None):
        """
        여러 샘플을 한 번에 비교하여 표시합니다.
        
        Args:
            num_samples (int): 표시할 샘플 개수
            save_path (str): 저장 경로
        """
        num_samples = min(num_samples, self.num_samples)
        indices = np.random.choice(self.num_samples, num_samples, replace=False)
        
        fig, axes = plt.subplots(3, num_samples, figsize=(5*num_samples, 12))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for col, idx in enumerate(indices):
            height_map = self.height_maps[idx]
            slope_map = self.slope_maps[idx]
            
            # 2채널 데이터 처리
            if self.slope_channel is not None:
                slope_vis = self.slope_channel[idx] * 90.0  # 정규화 → 도
                cot_vis = self.cot_channel[idx]
            else:
                slope_vis = slope_map
                cot_vis = self.costmaps[idx]
            
            path = self.paths[idx]
            img_size = height_map.shape[0]
            path_pixels = ((path + 1) / 2) * img_size
            
            # Height Map
            axes[0, col].imshow(height_map, cmap='terrain', origin='lower')
            axes[0, col].plot(path_pixels[:, 1], path_pixels[:, 0], 'r-', linewidth=1.5)
            axes[0, col].scatter(path_pixels[0, 1], path_pixels[0, 0], c='cyan', s=50, zorder=10)
            axes[0, col].scatter(path_pixels[-1, 1], path_pixels[-1, 0], c='yellow', s=50, zorder=10)
            axes[0, col].set_title(f'Height - Sample {idx}')
            axes[0, col].axis('off')
            
            # Slope Map (Ch 0)
            axes[1, col].imshow(slope_vis, cmap='jet', origin='lower', vmin=0, vmax=30)
            axes[1, col].plot(path_pixels[:, 1], path_pixels[:, 0], 'w-', linewidth=1.5)
            axes[1, col].scatter(path_pixels[0, 1], path_pixels[0, 0], c='cyan', s=50, zorder=10)
            axes[1, col].scatter(path_pixels[-1, 1], path_pixels[-1, 0], c='yellow', s=50, zorder=10)
            axes[1, col].set_title(f'Slope (Ch 0) - {idx}')
            axes[1, col].axis('off')
            
            # CoT Cost Map (Ch 1)
            axes[2, col].imshow(cot_vis, cmap='hot', origin='lower', vmin=0, vmax=1)
            axes[2, col].plot(path_pixels[:, 1], path_pixels[:, 0], 'c-', linewidth=1.5)
            axes[2, col].scatter(path_pixels[0, 1], path_pixels[0, 0], c='cyan', s=50, zorder=10)
            axes[2, col].scatter(path_pixels[-1, 1], path_pixels[-1, 0], c='yellow', s=50, zorder=10)
            axes[2, col].set_title(f'CoT (Ch 1) - {idx}')
            axes[2, col].axis('off')
            
        plt.suptitle('Multiple Slope + CoT Terrain Samples Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def visualize_statistics(self, save_path=None):
        """
        데이터셋 전체의 통계를 시각화합니다.
        
        Args:
            save_path (str): 저장 경로
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Height 분포
        axes[0, 0].hist(self.height_maps.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Height Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Height (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Slope 분포
        axes[0, 1].hist(self.slope_maps.flatten(), bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Slope Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Slope Angle (°)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # CoT Cost 분포 (Channel 1)
        if self.cot_channel is not None:
            cot_data = self.cot_channel.flatten()
        else:
            cot_data = self.costmaps.flatten()
        
        axes[0, 2].hist(cot_data, bins=50, color='orange', alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('CoT Cost Distribution (Ch 1)', fontweight='bold')
        axes[0, 2].set_xlabel('Normalized CoT Cost')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Path Length 분포
        path_lengths = []
        img_size = self.height_maps.shape[1]  # H dimension
        for path in self.paths:
            path_pixels = ((path + 1) / 2) * img_size
            length = np.sum(np.sqrt(np.sum(np.diff(path_pixels, axis=0)**2, axis=1)))
            path_lengths.append(length)
        axes[1, 0].hist(path_lengths, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Path Length Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Path Length (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Path Statistics
        axes[1, 1].axis('off')
        path_stats_text = f"""
Path Statistics
━━━━━━━━━━━━━━━━━━━━━━━
Total Paths: {len(path_lengths)}

Length (pixels):
  Min: {min(path_lengths):.2f}
  Max: {max(path_lengths):.2f}
  Mean: {np.mean(path_lengths):.2f}
  Std: {np.std(path_lengths):.2f}
━━━━━━━━━━━━━━━━━━━━━━━
        """
        axes[1, 1].text(0.1, 0.5, path_stats_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round',
                       facecolor='lightgreen', alpha=0.5))
        
        # Summary Statistics
        axes[1, 2].axis('off')
        
        if self.cot_channel is not None:
            cot_stats = f"""CoT Cost (Ch 1):
  Min: {self.cot_channel.min():.3f}
  Max: {self.cot_channel.max():.3f}
  Mean: {self.cot_channel.mean():.3f}
  Std: {self.cot_channel.std():.3f}"""
        else:
            cot_stats = f"""CoT Cost:
  Min: {self.costmaps.min():.3f}
  Max: {self.costmaps.max():.3f}
  Mean: {self.costmaps.mean():.3f}
  Std: {self.costmaps.std():.3f}"""
        
        summary_text = f"""
Dataset Statistics Summary
━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples: {self.num_samples}
Format: 2-channel [Slope, CoT]

Height (m):
  Min: {self.height_maps.min():.2f}
  Max: {self.height_maps.max():.2f}
  Mean: {self.height_maps.mean():.2f}
  Std: {self.height_maps.std():.2f}

Slope (°):
  Min: {self.slope_maps.min():.2f}
  Max: {self.slope_maps.max():.2f}
  Mean: {self.slope_maps.mean():.2f}
  Std: {self.slope_maps.std():.2f}

{cot_stats}
━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round',
                       facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('Dataset Statistics Overview (Slope + CoT)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Slope + CoT Terrain Dataset')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset file (default: ../data/dataset_gradient.pt)')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'multiple', 'stats', 'all'],
                       help='Visualization mode')
    parser.add_argument('--index', type=int, default=0,
                       help='Sample index for single mode')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples for multiple mode')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save visualizations (default: ../results)')
    
    args = parser.parse_args()
    
    # 저장 디렉토리 설정 (기본값: 상위 디렉토리의 results 폴더)
    if args.save_dir is None:
        args.save_dir = str(Path(__file__).parent.parent / 'results')
    
    # 저장 디렉토리 생성
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Visualizer 초기화
    visualizer = SlopeCotDataVisualizer(args.data_path)
    
    # 모드에 따라 시각화
    if args.mode == 'single':
        save_path = os.path.join(args.save_dir, f'sample_{args.index}.png')
        visualizer.visualize_sample(args.index, save_path)
        
    elif args.mode == 'multiple':
        save_path = os.path.join(args.save_dir, 'multiple_samples.png')
        visualizer.visualize_multiple(args.num_samples, save_path)
        
    elif args.mode == 'stats':
        save_path = os.path.join(args.save_dir, 'statistics.png')
        visualizer.visualize_statistics(save_path)
        
    elif args.mode == 'all':
        print("\n=== Visualizing Single Sample ===")
        visualizer.visualize_sample(0, os.path.join(args.save_dir, 'sample_0.png'))
        
        print("\n=== Visualizing Multiple Samples ===")
        visualizer.visualize_multiple(4, os.path.join(args.save_dir, 'multiple_samples.png'))
        
        print("\n=== Visualizing Statistics ===")
        visualizer.visualize_statistics(os.path.join(args.save_dir, 'statistics.png'))


if __name__ == "__main__":
    main()
