import numpy as np
import random
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion
import heapq

"""
Maze generation code.
"""

class MazeGenerator:
    """ Generates a maze with a cost gradient and ensures a solvable path. """
    def __init__(self, img_size, scale):
        self.img_size = img_size
        self.scale = scale
        self.height = img_size // scale
        self.width = img_size // scale
        if self.height % 2 == 0: self.height -= 1
        if self.width % 2 == 0: self.width -= 1
        self.grid = None
        self.costmap = None

    def generate(self, cost_weight=50.0):
        # --- Maze Grid Generation ---
        start_node = (0, 0)
        self.grid = np.zeros((self.height, self.width))
        self._recursive_backtracking(start_node)

        # 1) 구조 만들기
        costmap_structure_small = np.kron(self.grid, np.ones((self.scale, self.scale)))
        h, w = costmap_structure_small.shape
        costmap_structure = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        costmap_structure[:h, :w] = costmap_structure_small

        wall_thin_px = max(1, self.scale // 3)
        free = (costmap_structure == 1)
        walls = ~free
        walls = binary_erosion(walls, iterations=wall_thin_px)
        free = ~walls

        # 2) EDT 및 Cost 계산
        d = distance_transform_edt(free)
        H_eff = (self.scale / 2.0) + float(wall_thin_px)
        
        r1 = max(1.0, 0.7 * H_eff)
        r2 = max(1.0, 1.3 * H_eff)
        gamma1, gamma2 = 2.0, 1.2

        edge = np.clip((r1 - d) / r1, 0.0, 1.0) ** gamma1
        body = np.clip((r2 - d) / r2, 0.0, 1.0) ** gamma2
        cost = (0.6 * edge + 0.4 * body).astype(np.float32)

        # --- [최종 수정] 벽 및 테두리 처리 ---
        margin = max(1, self.scale // 4)
        
        # 1. 구조적 마진 (벽 근처)
        mask_margin = d < margin

        # 2. 비용 기반 차단 (0.8 이상은 벽)
        mask_high_cost = cost > 0.8
        
        # 3. [추가됨] 이미지 테두리(Boundary) 강제 차단
        # 이미지의 상하좌우 끝부분을 물리적인 벽으로 만듭니다.
        mask_border = np.zeros_like(cost, dtype=bool)
        border_width = margin  # margin만큼 테두리를 칩니다 (start 위치는 scale/2라 안전함)
        mask_border[:border_width, :] = True
        mask_border[-border_width:, :] = True
        mask_border[:, :border_width] = True
        mask_border[:, -border_width:] = True

        # 4. 최종 차단 마스크 (원래 벽 | 마진 | 높은 비용 | 테두리)
        blocked = (~free) | mask_margin | mask_high_cost | mask_border

        # 5. Costmap 적용
        self.costmap = np.where(blocked, np.inf, cost).astype(np.float32)

        # --- Start / End 설정 ---
        start_pos_low_res = (0, 0)
        end_pos_low_res = (self.height - 1, self.width - 1)

        start_pos = (np.array(start_pos_low_res) * self.scale + self.scale // 2).astype(int)
        end_pos = (np.array(end_pos_low_res) * self.scale + self.scale // 2).astype(int)

        # Start/End 지점만 구멍 뚫기 (테두리나 벽에 막히지 않게)
        self.costmap[start_pos[0], start_pos[1]] = 0.0
        self.costmap[end_pos[0], end_pos[1]] = 0.0

        # --- Path Finding ---
        path_pixels = a_star_search(
            self.costmap,
            tuple(start_pos),
            tuple(end_pos),
            cost_weight=cost_weight
        )

        if path_pixels is None:
            return self.generate(cost_weight)

        path_pixels_np = np.array(path_pixels, dtype=np.float32)
        path_normalized = (path_pixels_np / self.img_size) * 2 - 1

        return self.costmap, path_normalized, start_pos, end_pos

    def _recursive_backtracking(self, pos):
        r, c = pos
        self.grid[r, c] = 1  # Mark as path

        neighbors = [(r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)]
        random.shuffle(neighbors)

        for next_r, next_c in neighbors:
            if 0 <= next_r < self.height and 0 <= next_c < self.width and self.grid[next_r, next_c] == 0:
                wall_r, wall_c = (r + next_r) // 2, (c + next_c) // 2
                self.grid[wall_r, wall_c] = 1  # Knock down wall
                self._recursive_backtracking((next_r, next_c))

    def _find_path_in_grid(self, start, end):
        q = [(start, [start])]
        visited = {start}
        while q:
            (r, c), path = q.pop(0)
            if (r, c) == end:
                return path
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                # Check grid boundaries and if it's a path (value=1)
                if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    new_path = list(path)
                    new_path.append((nr, nc))
                    q.append(((nr, nc), new_path))
        return None

def a_star_search(costmap, start, end, cost_weight=20.0):
    """
    A* on a costmap where walls/near-walls are np.inf.
    - Cost is applied to g only: step * (1 + cost_weight * c_mid)
    - Heuristic is Euclidean distance (no cost mixed in)
    - Strict anti-corner-cut for diagonals
    """
    rows, cols = costmap.shape
    start = tuple(start); end = tuple(end)

    # 시작/목표가 통과가능한 셀인지 확인
    if not (np.isfinite(costmap[start]) and np.isfinite(costmap[end])):
        return None

    def h(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    # open set: (f, tie, node)
    counter = 0
    open_heap = [(h(start, end), counter, start)]
    came_from = {}

    g = np.full((rows, cols), np.inf, dtype=np.float32)
    g[start] = 0.0

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur == end:
            # 경로 복원
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]

        cr, cc = cur
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not np.isfinite(costmap[nr, nc]):   # 벽/근벽은 스킵
                continue

            # 대각선 코너컷 금지
            if abs(dr) + abs(dc) == 2:
                if (not np.isfinite(costmap[cr + dr, cc]) or
                    not np.isfinite(costmap[cr, cc + dc])):
                    continue

            step = 1.0 if (dr == 0 or dc == 0) else np.sqrt(2.0)
            c_mid = 0.5 * (costmap[cr, cc] + costmap[nr, nc])
            tentative = g[cr, cc] + step * (1.0 + cost_weight * c_mid)

            if tentative < g[nr, nc]:
                came_from[(nr, nc)] = cur
                g[nr, nc] = tentative
                # tie-break 살짝(더 직선 선호)
                f = tentative + step * h((nr, nc), end) * (1.0 + 1e-3)
                counter += 1
                heapq.heappush(open_heap, (f, counter, (nr, nc)))

    return None