# import os
# import glob
# import numpy as np
# import matplotlib.pyplot as plt

# def read_pfm(path):
#     with open(path, "rb") as f:
#         header = f.readline().decode("ascii").rstrip()
#         if header not in ("PF", "Pf"):
#             raise ValueError(f"Not a PFM file: {path}")

#         dims = f.readline().decode("ascii").strip()
#         while dims.startswith("#"):
#             dims = f.readline().decode("ascii").strip()
#         w, h = map(int, dims.split())

#         scale = float(f.readline().decode("ascii").strip())
#         endian = "<" if scale < 0 else ">"
#         channels = 3 if header == "PF" else 1

#         data = np.fromfile(f, endian + "f", count=w * h * channels)
#         data = data.reshape((h, w, channels)) if channels == 3 else data.reshape((h, w))

#         # PFM은 보통 bottom-up 저장
#         data = np.flipud(data)
#         return data

# def to_rgb_cost(cost, vmin=0.0, vmax=0.75, cmap_name="jet",
#                 unknown_rgba=(0, 0, 0, 0),      # NaN: 투명
#                 obstacle_rgba=(1, 0, 0, 1)):     # Inf: 빨강
#     if cost.ndim != 2:
#         raise ValueError("to_rgb_cost expects a 2D single-channel array (H,W).")

#     unknown = np.isnan(cost)
#     obstacle = np.isinf(cost)
#     finite = np.isfinite(cost)

#     x = np.zeros_like(cost, dtype=np.float32)
#     x[finite] = np.clip(cost[finite], vmin, vmax)
#     x[finite] = (x[finite] - vmin) / (vmax - vmin + 1e-12)

#     cmap = plt.get_cmap(cmap_name)
#     rgba = cmap(x)  # (H,W,4)

#     rgba[unknown]  = unknown_rgba
#     rgba[obstacle] = obstacle_rgba
#     return rgba

# def batch_visualize_pfms(input_dir,
#                          output_dir=None,
#                          pattern="*.pfm",
#                          vmin=0.1,
#                          vmax=0.7,
#                          cmap_name="jet"):
#     if output_dir is None:
#         output_dir = os.path.join(input_dir, "png_vis")
#     os.makedirs(output_dir, exist_ok=True)

#     pfm_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
#     if not pfm_paths:
#         raise FileNotFoundError(f"No PFM files found: {os.path.join(input_dir, pattern)}")

#     print(f"Found {len(pfm_paths)} PFM files.")
#     print(f"Saving PNGs to: {output_dir}")

#     for i, pfm_path in enumerate(pfm_paths, 1):
#         try:
#             pfm = read_pfm(pfm_path)
#             if pfm.ndim == 3:
#                 # PF(3채널)인 경우 첫 채널만 사용 (필요시 수정)
#                 pfm_2d = pfm[..., 0]
#             else:
#                 pfm_2d = pfm

#             rgba = to_rgb_cost(pfm_2d, vmin=vmin, vmax=vmax, cmap_name=cmap_name)

#             base = os.path.splitext(os.path.basename(pfm_path))[0]
#             out_path = os.path.join(output_dir, f"{base}.png")
#             plt.imsave(out_path, rgba)

#             if i % 20 == 0 or i == len(pfm_paths):
#                 print(f"[{i}/{len(pfm_paths)}] saved: {out_path}")

#         except Exception as e:
#             print(f"[{i}/{len(pfm_paths)}] FAILED: {pfm_path} -> {e}")

#     print("Done.")

# # 사용 예:
# batch_visualize_pfms(
#     input_dir="./dataset2",            # PFM들이 있는 폴더
#     output_dir="./png_vis2",   # 저장 폴더 (없으면 자동 생성)
#     pattern="map_cost_*.pfm", # 원하는 파일 패턴
#     vmin=0.1,
#     vmax=0.7,
#     cmap_name="jet"
# )

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")

        dims = f.readline().decode("ascii").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("ascii").strip()
        w, h = map(int, dims.split())

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if header == "PF" else 1

        data = np.fromfile(f, endian + "f", count=w * h * channels)
        data = data.reshape((h, w, channels)) if channels == 3 else data.reshape((h, w))

        data = np.flipud(data)
        return data

# --- 수정된 부분: c_cst 인자 추가 및 연산 적용 ---
def to_rgb_paper_style(cost, c_cst=0.1, vmin=0.1, vmax=0.5): 
    cost_proc = cost.copy()
    
    # 1. 마스크 생성
    unknown = np.isnan(cost_proc)
    # 데이터셋 특성에 따라 0.7 이상을 장애물로 간주 (논문의 c_obs와 유사)
    obstacle = np.isinf(cost_proc) | (cost_proc >= 0.7) 
    finite = np.isfinite(cost_proc) & ~obstacle

    # 2. 논문 수식 적용: c_aug = c_ele + c_cst
    # 상수 c_cst를 더해줌으로써 전체적인 비용 베이스를 높이고 그라데이션을 풍부하게 함
    cost_proc[finite] += c_cst

    # 3. 정규화 (PiYG 컬러맵: 0=Pink, 1=Green)
    x = np.zeros_like(cost_proc, dtype=np.float32)
    # 수정한 vmin, vmax 범위 내로 클리핑
    x[finite] = np.clip((cost_proc[finite] - vmin) / (vmax - vmin + 1e-12), 0, 1)

    # 4. 컬러맵 적용 (PiYG: Pink -> White -> Green)
    cmap = plt.get_cmap("PiYG") 
    rgba = cmap(x)

    # 5. 특수 색상 강제 지정
    rgba[unknown] = [0.7, 0.7, 0.7, 1.0]  # 배경: 회색
    rgba[obstacle] = [1.0, 0.0, 0.0, 1.0] # 장애물: 빨간색
    
    return rgba

def batch_visualize_for_paper(input_dir,
                              output_dir="./piyg_vis2",
                              c_cst=0.1,    
                              vmin=0.01,
                              vmax=0.5): # vmax를 조금 낮게 잡아야 색 대비가 잘 보입니다.
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pfm_paths = sorted(glob.glob(os.path.join(input_dir, "map_cost_*.pfm")))
    
    print(f"총 {len(pfm_paths)}개 파일을 논문 스타일로 변환합니다. (c_cst={c_cst})")

    for i, path in enumerate(pfm_paths, 1):
        try:
            cost_data = read_pfm(path)
            if cost_data.ndim == 3:
                cost_data = cost_data[..., 0]

            # 이제 인자가 일치하므로 에러가 발생하지 않습니다.
            rgba = to_rgb_paper_style(cost_data, c_cst=c_cst, vmin=vmin, vmax=vmax)

            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{base}_paper.png")
            plt.imsave(out_path, rgba)

            if i % 10 == 0 or i == len(pfm_paths):
                print(f"진행 중: {i}/{len(pfm_paths)}")

        except Exception as e:
            print(f"에러 발생 ({path}): {e}")

    print("완료되었습니다.")

# --- 실행 부분 ---
batch_visualize_for_paper(
    input_dir="./dataset2", 
    c_cst=0.15,  # 0.1 ~ 0.2 사이를 추천합니다.
    vmin=0.1, 
    vmax=0.45    # 이 값을 낮출수록 초록색이 더 진해집니다.
)