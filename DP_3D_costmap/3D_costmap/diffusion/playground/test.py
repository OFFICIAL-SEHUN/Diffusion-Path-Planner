import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import heapq

# -----------------------------------------
# 1. Terrain-like Custom Colormap
# -----------------------------------------
def create_terrain_cmap():
    colors = [
        (0.00, "#2ecc71"), (0.10, "#6ecc61"), (0.20, "#a8d44e"),
        (0.30, "#d6e14c"), (0.40, "#f1c40f"), (0.50, "#e67e22"),
        (0.60, "#d35400"), (0.70, "#c0392b"), (0.80, "#7d2b47"),
        (0.90, "#4b1c4f"), (1.00, "#2c0e4a")
    ]
    return LinearSegmentedColormap.from_list("terrain_map", colors)

# -----------------------------------------
# 2. Map Generation (Noise + Blobs)
# -----------------------------------------
def gaussian_noise(size, sigma):
    noise = np.random.rand(size, size)
    return gaussian_filter(noise, sigma=sigma)

def generate_blobs(size, num_blobs):
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    X, Y = np.meshgrid(x, y)
    blob_map = np.zeros((size, size))

    for _ in range(num_blobs):
        cx = np.random.uniform(0, size)
        cy = np.random.uniform(0, size)
        sigma = np.random.uniform(5, 15)      # 덩어리를 조금 더 키움
        intensity = np.random.uniform(50, 200) 
        gauss = intensity * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        blob_map += gauss
    return blob_map

def generate_random_cost_map(size=100, num_obstacles=30):
    cost_map = np.zeros((size, size))
    # 다양한 주파수의 노이즈 합성
    cost_map += gaussian_noise(size, sigma=40) * 30
    cost_map += gaussian_noise(size, sigma=15) * 15
    cost_map += generate_blobs(size, num_obstacles) # 장애물 추가

    # 0.0 ~ 1.0 정규화 (시각화를 위해)
    cost_map -= cost_map.min()
    cost_map /= cost_map.max()
    return cost_map

# -----------------------------------------
# 3. A* 알고리즘 (핵심 수정 부분)
# -----------------------------------------
def find_path(cost_map, start, goal):
    rows, cols = cost_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in np.ndindex(rows, cols)}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        cx, cy = current
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                # ---------------------------------------------------------
                # [수정됨] 비용 계산 로직
                # 시각적 맵(0~1)을 논리적 비용(1~수천)으로 변환
                # ---------------------------------------------------------
                visual_cost = cost_map[nx, ny]
                
                # 비용 함수: (기본값) + (시각적 비용)^지수 * 가중치
                # 예: 초록색(0.1) -> 0.1^5 * 500 = 0 (거의 공짜)
                # 예: 빨간색(0.9) -> 0.9^5 * 500 = 295 (엄청난 페널티)
                # 빨간색 한 칸 밟는 것보다 초록색 295칸 돌아가는 게 낫다고 판단하게 됨
                terrain_penalty = (visual_cost ** 5) * 500
                
                dist = 1.414 if dx != 0 and dy != 0 else 1.0
                step_cost = (1 + terrain_penalty) * dist

                new_g = g_score[current] + step_cost

                if new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    
                    # 휴리스틱 가중치 조절
                    # 휴리스틱(거리) 비중을 조금 낮춰서(0.8), '안전한 길' 탐색을 유도
                    h = (abs(nx - goal[0]) + abs(ny - goal[1])) * 0.8
                    heapq.heappush(open_set, (new_g + h, (nx, ny)))
                    came_from[(nx, ny)] = current
    return []

# -----------------------------------------
# 실행 및 시각화
# -----------------------------------------
map_size = 120
num_blobs = 35
# 랜덤 시드를 고정하지 않아 매번 다른 지도가 나옵니다
cost_map = generate_random_cost_map(map_size, num_blobs)

start_node = (10, 10)
goal_node = (100, 100)

path = find_path(cost_map, start_node, goal_node)

plt.figure(figsize=(10, 10))

# 1. 배경: 시각적으로 부드러운 0~1 range의 Cost Map
plt.imshow(cost_map, cmap=create_terrain_cmap(), origin='lower', interpolation='bicubic')
cbar = plt.colorbar(shrink=0.8)
cbar.set_label("Visual Cost (Green=Safe, Purple=Danger)")

# 2. 경로 그리기
if path:
    py, px = zip(*path)
    # 흰색 실선
    plt.plot(px, py, color='white', linewidth=2, label='Autonomous Path')
    # 검은색 외곽선 (잘 보이게)
    plt.plot(px, py, color='black', linewidth=4, alpha=0.4, zorder=1)
else:
    print("경로를 찾을 수 없습니다.")

plt.scatter(start_node[1], start_node[0], c='cyan', s=200, edgecolors='black', linewidth=2, label='Start', zorder=10)
plt.scatter(goal_node[1], goal_node[0], c='blue', s=250, edgecolors='white', marker='*', linewidth=2, label='Goal', zorder=10)

plt.title("High-Cost Avoidance Path Planning\n(Exponential Cost Penalty Applied)")
plt.legend(loc='upper right')
plt.axis('off')
plt.tight_layout()
plt.show()