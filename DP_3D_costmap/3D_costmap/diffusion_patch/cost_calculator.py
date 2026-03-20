"""
경로의 CoT (Cost of Transport) 비용 계산 모듈
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from generate_data import calculate_directional_cot


def calculate_path_cot(path_pixels, height_map, pixel_resolution, limit_angle_deg=35.0):
    """
    경로의 총 CoT 비용 계산 (방향성 CoT 사용)
    
    Args:
        path_pixels (list): 경로 [(row, col), ...]
        height_map (np.ndarray): 높이 맵
        pixel_resolution (float): 픽셀당 실제 거리 (m/pixel)
        limit_angle_deg (float): 최대 등반 각도
        
    Returns:
        dict: {
            'total_cot': float,
            'avg_cot': float,
            'segments': list of segment info,
            'stats': dict of statistics
        }
    """
    if len(path_pixels) < 2:
        return None
    
    total_cot = 0.0
    segments = []
    uphill_count = 0
    downhill_count = 0
    flat_count = 0
    uphill_cot = 0.0
    downhill_cot = 0.0
    
    for i in range(len(path_pixels) - 1):
        curr_pos = path_pixels[i]
        next_pos = path_pixels[i + 1]
        
        # 높이 및 거리 계산
        height_curr = height_map[curr_pos]
        height_next = height_map[next_pos]
        
        # 이동 거리
        dr = next_pos[0] - curr_pos[0]
        dc = next_pos[1] - curr_pos[1]
        pixel_distance = np.sqrt(dr**2 + dc**2)
        real_distance = pixel_distance * pixel_resolution
        
        # 방향성 CoT 계산
        cot = calculate_directional_cot(height_curr, height_next, real_distance, limit_angle_deg)
        
        # 높이 차이 및 경사각
        height_diff = height_next - height_curr
        slope_rad = np.arctan2(height_diff, real_distance)
        slope_deg = np.degrees(slope_rad)
        
        # 통계 수집
        if np.isinf(cot):
            cot = 1000.0  # Inf를 큰 값으로 대체 (시각화용)
        
        total_cot += cot * real_distance
        
        # 오르막/내리막 분류
        if slope_deg > 1.0:  # 1도 이상 오르막
            uphill_count += 1
            uphill_cot += cot * real_distance
        elif slope_deg < -1.0:  # 1도 이상 내리막
            downhill_count += 1
            downhill_cot += cot * real_distance
        else:
            flat_count += 1
        
        segments.append({
            'index': i,
            'slope_deg': slope_deg,
            'height_diff': height_diff,
            'distance': real_distance,
            'cot': cot,
            'cot_cost': cot * real_distance
        })
    
    total_distance = sum(seg['distance'] for seg in segments)
    avg_cot = total_cot / total_distance if total_distance > 0 else 0
    
    stats = {
        'total_distance': total_distance,
        'uphill_ratio': uphill_count / len(segments) if segments else 0,
        'downhill_ratio': downhill_count / len(segments) if segments else 0,
        'flat_ratio': flat_count / len(segments) if segments else 0,
        'avg_slope': np.mean([seg['slope_deg'] for seg in segments]),
        'max_slope': max([seg['slope_deg'] for seg in segments]),
        'min_slope': min([seg['slope_deg'] for seg in segments]),
        'uphill_cot': uphill_cot,
        'downhill_cot': downhill_cot,
    }
    
    return {
        'total_cot': total_cot,
        'avg_cot': avg_cot,
        'segments': segments,
        'stats': stats
    }
