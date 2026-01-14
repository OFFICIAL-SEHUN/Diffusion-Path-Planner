import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

def generate_height_map(size=100, scales=[(40, 50), (15, 20), (5, 5)], height_range=(0, 20)):
    """
    울퉁불퉁한 산악 지형의 높이 맵을 생성합니다.
    
    Parameters:
    -----------
    size : int
        맵의 크기 (size x size)
    scales : list of tuples
        각 노이즈 레이어의 (sigma, weight) 쌍
    height_range : tuple
        높이 범위 (min_height, max_height) in meters
    
    Returns:
    --------
    height_map : ndarray
        생성된 높이 맵 (size x size)
    """
    height_map = np.zeros((size, size))
    
    # 여러 스케일의 노이즈를 합성하여 자연스러운 지형 생성
    for scale, weight in scales:
        noise = np.random.rand(size, size)
        height_map += gaussian_filter(noise, sigma=scale) * weight
    
    # 지정된 높이 범위로 정규화
    height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
    height_map = height_map * (height_range[1] - height_range[0]) + height_range[0]
    
    return height_map


def visualize_height_map(height_map, save_path=None, interactive=True):
    """
    높이 맵을 3D로 시각화합니다.
    
    Parameters:
    -----------
    height_map : ndarray
        높이 맵
    save_path : str, optional
        저장 경로 (None이면 저장하지 않음)
    interactive : bool
        인터랙티브 모드 (True면 plt.show() 호출)
    """
    size = height_map.shape[0]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 그리드 생성
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # 3D 표면 그리기
    surf = ax.plot_surface(X, Y, height_map, cmap='terrain', alpha=0.9, 
                          linewidth=0, antialiased=True, edgecolor='none')
    
    # Colorbar 추가
    cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Height (m)", rotation=270, labelpad=20)
    
    # 축 레이블 및 제목
    ax.set_title("3D Height Map Terrain", fontsize=16, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Height (m)', fontsize=12)
    
    # 그리드 설정
    ax.grid(True, alpha=0.3)
    
    # 저장
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Height map saved to: {save_path}")
    
    # 인터랙티브 표시
    if interactive:
        plt.show()
    else:
        plt.close()


def main():
    """
    메인 실행 함수 - 높이 맵 생성 및 시각화
    """
    print("Generating height map...")
    
    # 높이 맵 생성
    size = 100
    height_map = generate_height_map(
        size=size,
        scales=[(40, 50), (15, 20), (5, 5)],
        height_range=(0, 20)
    )
    
    print(f"Height map generated: {size}x{size}")
    print(f"Height range: {height_map.min():.2f}m to {height_map.max():.2f}m")
    
    # 시각화
    print("\nVisualizing...")
    visualize_height_map(
        height_map=height_map,
        save_path='results/height_map_visualization.png',
        interactive=True
    )
    
    # 높이 맵 데이터 저장
    np.save('results/height_map.npy', height_map)
    print("\nHeight map data saved to results/height_map.npy")

if __name__ == "__main__":
    main()

