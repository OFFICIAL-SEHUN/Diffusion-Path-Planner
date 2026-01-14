import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def visualize_projected_costmap():
    # --- 1. 가상 데이터 생성 ---
    
    # 이미지 해상도 및 카메라 설정
    W, H = 64, 48 # 계산 속도를 위해 저해상도 격자 사용
    fx, fy = 40.0, 40.0 # 스케일에 맞게 조정
    cx, cy = W/2, H/2
    
    # 가상의 "주행 가능성 예측 이미지" 만들기
    # 이미지 중앙(길)은 안전(0:초록), 양옆은 위험(1:빨간색)하다고 가정
    # 0.0 (Safe) ~ 1.0 (Danger)
    prediction_img = np.zeros((H, W))
    
    for v in range(H):
        for u in range(W):
            # 이미지 중앙에서 멀어질수록 위험도가 높아지는 간단한 로직
            dist_from_center = abs(u - cx) / (W/2) 
            score = dist_from_center ** 2 # 제곱해서 대비를 줌
            prediction_img[v, u] = np.clip(score, 0, 1)
            
            # 하늘(이미지 위쪽 절반)은 땅이 아니므로 무시 (Masking)
            if v < H/2:
                prediction_img[v, u] = np.nan # 투영 안 함

    # --- 2. 2D 이미지 -> 3D 월드로 역투영 (Back-Projection) ---
    
    # 카메라 위치 (로봇: 0,0,0 / 카메라: 높이 1m, 약간 아래를 봄)
    cam_h = 1.0
    tilt_angle = np.radians(20) # 20도 아래로 숙임
    
    world_points = []
    colors = []

    # 회전 행렬 (Camera to World)
    # 카메라 좌표계(Z:앞, Y:아래) -> 월드(X:앞, Z:위)
    # Pitch 회전(아래로 숙임) 적용
    cos_t = np.cos(tilt_angle)
    sin_t = np.sin(tilt_angle)
    
    # 간단한 레이캐스팅(Ray-casting) 방식 사용
    # 평면 가정: Z_world = 0 인 지점 찾기
    
    for v in range(H):
        for u in range(W):
            score = prediction_img[v, u]
            if np.isnan(score): continue
            
            # 1. 정규화된 이미지 좌표 (Normalized Image Plane)
            # z=1 일 때의 x, y
            x_norm = (u - cx) / fx
            y_norm = (v - cy) / fy
            
            # 2. 카메라 좌표계 상의 방향 벡터 (Ray Vector)
            # Cam Frame: X(우), Y(하), Z(전)
            ray_cam = np.array([x_norm, y_norm, 1.0])
            
            # 3. 월드 좌표계로 회전 (카메라가 숙여져 있음)
            # World Frame: X(전), Y(좌), Z(상)이라고 가정하면 변환이 복잡하므로
            # 직관적으로: 카메라가 보고 있는 방향을 기준으로 계산
            
            # 카메라 좌표 -> 로봇 몸체 좌표 변환 (Pitch 회전)
            # y_rob = y_cam * cos - z_cam * sin
            # z_rob = y_cam * sin + z_cam * cos
            # (카메라 Y축이 아래쪽이므로, 아래로 20도 숙이면 Z축 성분이 생김)
            
            # 간단히 벡터 회전: 
            # Ray의 Y성분(아래)과 Z성분(앞)을 회전
            y_rot = ray_cam[1] * cos_t + ray_cam[2] * sin_t # 실제 아래쪽 성분
            z_rot = -ray_cam[1] * sin_t + ray_cam[2] * cos_t # 실제 앞쪽 성분
            
            # 이제 Ray는 (x_norm, y_rot, z_rot) 방향으로 나아감
            # 시작점은 (0, 0, cam_h)
            # 바닥(Z=0)에 닿으려면?
            # cam_h - t * y_rot = 0  => t = cam_h / y_rot
            # (여기서 y_rot은 아래쪽 방향 성분이므로 양수라고 가정)
            
            if y_rot <= 0.1: continue # 수평선 위거나 너무 멀면 패스
            
            t = cam_h / y_rot
            
            # 바닥 충돌 지점 계산
            # X(좌우) = t * x_norm
            # Y(전후) = t * z_rot (원래 Z가 앞이었음)
            
            # 월드 좌표계에 맞춰 재배치
            wx = t * z_rot # 앞쪽 거리
            wy = -t * x_norm # 왼쪽 거리 (이미지 x가 오른쪽이므로 -붙임)
            wz = 0 # 바닥
            
            if wx > 0 and wx < 10: # 10m 이내만 표시
                world_points.append([wx, wy, wz])
                colors.append(score)

    world_points = np.array(world_points)
    colors = np.array(colors)

    # --- 3. 시각화 ---
    fig = plt.figure(figsize=(14, 6))

    # [왼쪽] 2D 예측 이미지 (로봇이 본 것)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("1. Traversability Prediction (Camera View)", fontsize=14)
    # 빨강(위험) ~ 초록(안전) Colormap (Reverse Jet or RdYlGn)
    im = ax1.imshow(prediction_img, cmap='RdYlGn_r', interpolation='nearest') 
    ax1.set_xlabel('u')
    ax1.set_ylabel('v')
    plt.colorbar(im, ax=ax1, label='Traversability Cost')

    # [오른쪽] 3D 월드 투영 (바닥에 칠해진 것)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("2. Projected Costmap (3D World)", fontsize=14)
    
    # 로봇 위치
    ax2.scatter([0], [0], [cam_h], c='blue', s=100, label='Robot Camera')
    ax2.plot([0, 0], [0, 0], [0, cam_h], 'k-', lw=3) # 몸체
    
    # 바닥에 점 뿌리기
    # c=colors: 점수(0~1)에 따라 색상 매핑
    p = ax2.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], 
                    c=colors, cmap='RdYlGn_r', s=10, alpha=0.8)
    
    ax2.set_xlabel('X (Forward)')
    ax2.set_ylabel('Y (Left)')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-1, 8)
    ax2.set_ylim(-4, 4)
    ax2.set_zlim(0, 2)
    ax2.view_init(elev=40, azim=-100)
    
    # 설명 추가
    ax2.text(0, 0, 2.2, "Safe Path Projected on Ground", color='green', fontweight='bold')

    plt.tight_layout()
    plt.show()

visualize_projected_costmap()