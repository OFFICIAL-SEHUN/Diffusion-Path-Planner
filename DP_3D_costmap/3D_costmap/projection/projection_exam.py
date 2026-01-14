import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_robot_and_camera_projection():
    # --- 1. 설정 (이전과 동일) ---
    img_width, img_height = 640, 480
    fx, fy = 400.0, 400.0
    cx, cy = img_width / 2, img_height / 2
    
    # S자 경로 생성
    num_points = 100
    x_world = np.linspace(0.5, 4.0, num_points)
    y_world = 0.6 * np.sin(x_world * 2.5) 
    z_world = np.zeros(num_points) # 바닥
    path_points_world = np.stack((x_world, y_world, z_world), axis=1)

    # 카메라 위치 (로봇 머리)
    cam_h = 0.5  # 높이
    cam_pos = np.array([0, 0, cam_h])

    # --- 2. 투영 계산 (이전과 동일한 로직) ---
    projected_uv = []
    for pt in path_points_world:
        wx, wy, wz = pt
        # 월드 -> 카메라 좌표 변환
        cz = wx
        cx_cam = -wy
        cy_cam = cam_h - wz
        
        if cz > 0:
            u = fx * (cx_cam / cz) + cx
            v = fy * (cy_cam / cz) + cy
            if 0 <= u < img_width and 0 <= v < img_height:
                projected_uv.append([u, v])
            else:
                projected_uv.append([np.nan, np.nan])
        else:
            projected_uv.append([np.nan, np.nan])
    projected_uv = np.array(projected_uv)

    # --- 3. 시각화 ---
    fig = plt.figure(figsize=(14, 6))

    # [왼쪽] 3D 월드 뷰 (로봇 모델 추가)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("1. 3D World: Robot & Path", fontsize=14, pad=20)

    # A. 로봇 몸체 그리기 (직육면체)
    # 로봇 크기 설정 (길이, 너비, 높이)
    r_len, r_wid, r_h = 0.6, 0.4, 0.2
    r_center = [0, 0, 0.3] # 몸통 중심 높이
    
    # 박스 꼭지점 정의
    corners = np.array([
        [r_center[0]-r_len/2, r_center[1]-r_wid/2, r_center[2]-r_h/2],
        [r_center[0]+r_len/2, r_center[1]-r_wid/2, r_center[2]-r_h/2],
        [r_center[0]+r_len/2, r_center[1]+r_wid/2, r_center[2]-r_h/2],
        [r_center[0]-r_len/2, r_center[1]+r_wid/2, r_center[2]-r_h/2],
        [r_center[0]-r_len/2, r_center[1]-r_wid/2, r_center[2]+r_h/2],
        [r_center[0]+r_len/2, r_center[1]-r_wid/2, r_center[2]+r_h/2],
        [r_center[0]+r_len/2, r_center[1]+r_wid/2, r_center[2]+r_h/2],
        [r_center[0]-r_len/2, r_center[1]+r_wid/2, r_center[2]+r_h/2]
    ])
    
    # 면 구성
    faces = [
        [corners[0], corners[1], corners[5], corners[4]], # 옆면
        [corners[7], corners[6], corners[2], corners[3]], # 옆면
        [corners[0], corners[4], corners[7], corners[3]], # 뒷면
        [corners[1], corners[5], corners[6], corners[2]], # 앞면 (카메라쪽)
        [corners[4], corners[5], corners[6], corners[7]], # 윗면
        [corners[0], corners[1], corners[2], corners[3]]  # 아랫면
    ]
    
    # 로봇 몸체 그리기 (회색 박스)
    ax1.add_collection3d(Poly3DCollection(faces, facecolors='gray', linewidths=1, edgecolors='k', alpha=0.3))

    # B. 다리 4개 그리기 (간단한 선)
    leg_z = r_center[2] - r_h/2
    ax1.plot([corners[0][0], corners[0][0]], [corners[0][1], corners[0][1]], [leg_z, 0], 'k-', lw=3)
    ax1.plot([corners[1][0], corners[1][0]], [corners[1][1], corners[1][1]], [leg_z, 0], 'k-', lw=3)
    ax1.plot([corners[2][0], corners[2][0]], [corners[2][1], corners[2][1]], [leg_z, 0], 'k-', lw=3)
    ax1.plot([corners[3][0], corners[3][0]], [corners[3][1], corners[3][1]], [leg_z, 0], 'k-', lw=3)

    # C. 카메라 시야각(Frustum) 그리기 (핵심!)
    # 카메라 위치에서 앞쪽으로 뻗어나가는 피라미드
    scale = 0.5
    cam_face_center = [0.3, 0, 0.5] # 로봇 앞쪽 얼굴
    
    # 시야각의 4개 모서리 (X축 방향으로 뻗음)
    frustum_end_x = cam_face_center[0] + scale
    frustum_w = 0.3 # 시야 폭
    frustum_h = 0.2 # 시야 높이
    
    f_tl = [frustum_end_x,  frustum_w,  cam_face_center[2] + frustum_h]
    f_tr = [frustum_end_x, -frustum_w,  cam_face_center[2] + frustum_h]
    f_bl = [frustum_end_x,  frustum_w,  cam_face_center[2] - frustum_h]
    f_br = [frustum_end_x, -frustum_w,  cam_face_center[2] - frustum_h]
    
    # 카메라 렌즈 중심에서 모서리로 선 긋기
    cam_origin = [cam_face_center[0], 0, cam_face_center[2]]
    ax1.plot([cam_origin[0], f_tl[0]], [cam_origin[1], f_tl[1]], [cam_origin[2], f_tl[2]], 'c-', lw=1)
    ax1.plot([cam_origin[0], f_tr[0]], [cam_origin[1], f_tr[1]], [cam_origin[2], f_tr[2]], 'c-', lw=1)
    ax1.plot([cam_origin[0], f_bl[0]], [cam_origin[1], f_bl[1]], [cam_origin[2], f_bl[2]], 'c-', lw=1)
    ax1.plot([cam_origin[0], f_br[0]], [cam_origin[1], f_br[1]], [cam_origin[2], f_br[2]], 'c-', lw=1)
    
    # 앞쪽 사각형 연결
    ax1.plot([f_tl[0], f_tr[0]], [f_tl[1], f_tr[1]], [f_tl[2], f_tr[2]], 'c-', lw=1)
    ax1.plot([f_tr[0], f_br[0]], [f_tr[1], f_br[1]], [f_tr[2], f_br[2]], 'c-', lw=1)
    ax1.plot([f_br[0], f_bl[0]], [f_br[1], f_bl[1]], [f_br[2], f_bl[2]], 'c-', lw=1)
    ax1.plot([f_bl[0], f_tl[0]], [f_bl[1], f_tl[1]], [f_bl[2], f_tl[2]], 'c-', lw=1)

    # D. 경로 그리기
    ax1.plot(x_world, y_world, z_world, 'r.-', markersize=2, label='Path (Ground Truth)')
    ax1.plot([0, 4], [0, 0], [0, 0], 'k--', alpha=0.2) # 중심선

    # 축 설정
    ax1.set_xlabel('X (Forward)')
    ax1.set_ylabel('Y (Left/Right)')
    ax1.set_zlabel('Z (Height)')
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(0, 2)
    ax1.view_init(elev=25, azim=-130) # 로봇 등 뒤에서 보는 뷰

    # [오른쪽] 카메라 뷰
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("2. Camera View (Projected)", fontsize=14, pad=20)
    ax2.set_facecolor('#f0f0f0') 
    
    # 투영된 경로
    valid = ~np.isnan(projected_uv[:, 0])
    ax2.plot(projected_uv[valid, 0], projected_uv[valid, 1], 'r.-', lw=3, label='Projected Path')
    
    # 꾸미기
    ax2.set_xlim(0, img_width)
    ax2.set_ylim(img_height, 0)
    ax2.set_xlabel('u (pixel)')
    ax2.set_ylabel('v (pixel)')
    ax2.grid(True, alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

draw_robot_and_camera_projection()