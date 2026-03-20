"""
경로 관련 유틸리티 함수들
"""

import numpy as np


def denormalize_path(path_norm, img_size):
    """정규화된 경로를 픽셀 좌표로 변환"""
    # [-1, 1] → [0, img_size]
    path_scaled = (path_norm + 1) / 2 * img_size
    # (x, y) → (row, col)
    path_pixels = [(int(p[1]), int(p[0])) for p in path_scaled]
    return path_pixels


def check_path_linearity(path_pixels, threshold=0.85):
    """
    PCA를 사용하여 경로의 선형성(직선 경향) 검사
    
    Args:
        path_pixels: [(row, col), ...] 경로 좌표 리스트
        threshold: 첫 번째 주성분의 분산 비율 임계값 (기본 0.85 = 85%)
                  값이 클수록 더 직선적
                  
    Returns:
        float: 첫 번째 주성분이 설명하는 분산 비율 (0~1)
        bool: threshold 이상이면 True (너무 직선적)
    """
    if len(path_pixels) < 3:
        return 1.0, True  # 경로가 너무 짧으면 직선으로 간주
    
    # 경로 좌표를 numpy array로 변환
    path_array = np.array(path_pixels, dtype=np.float64)  # [N, 2]
    
    # 중앙화 (centering)
    path_mean = np.mean(path_array, axis=0)
    path_centered = path_array - path_mean
    
    # 공분산 행렬 계산
    cov_matrix = np.cov(path_centered.T)  # [2, 2]
    
    # 고유값 분해 (eigendecomposition)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 고유값 정렬 (내림차순)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    
    # 첫 번째 주성분이 설명하는 분산 비율
    explained_variance_ratio = eigenvalues[0] / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 1.0
    
    # 임계값 이상이면 너무 직선적
    is_linear = explained_variance_ratio >= threshold
    
    return explained_variance_ratio, is_linear
