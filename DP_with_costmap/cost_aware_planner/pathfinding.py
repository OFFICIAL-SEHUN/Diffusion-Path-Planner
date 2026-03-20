"""
Pathfinding algorithms for costmap: Dijkstra and RRT*.
Same interface as maze.a_star_search(costmap, start, end, cost_weight=...).
Returns list of (r, c) tuples or None if no path.
"""
import heapq
import numpy as np


def _wall(costmap, r, c):
    return costmap[r, c] >= 0.99


def _neighbors_8(rows, cols, r, c):
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc), (1.0 if (dr == 0 or dc == 0) else np.sqrt(2.0))


def _edge_cost(costmap, cur, nxt, step, cost_weight):
    c_mid = 0.5 * (costmap[cur[0], cur[1]] + costmap[nxt[0], nxt[1]])
    return step * (1.0 + cost_weight * c_mid)


def _no_corner_cut(costmap, cr, cc, nr, nc):
    dr, dc = nr - cr, nc - cc
    if abs(dr) + abs(dc) != 2:
        return True
    if costmap[cr + dr, cc] >= 0.99 or costmap[cr, cc + dc] >= 0.99:
        return False
    return True


def dijkstra_search(costmap, start, end, cost_weight=20.0):
    """Dijkstra on costmap (same cost and collision rules as A*). Returns path or None."""
    rows, cols = costmap.shape
    start = tuple(start)
    end = tuple(end)
    if _wall(costmap, start[0], start[1]) or _wall(costmap, end[0], end[1]):
        return None

    g = np.full((rows, cols), np.inf, dtype=np.float32)
    g[start[0], start[1]] = 0.0
    came_from = {}
    open_heap = [(0.0, start)]
    visited = set()

    while open_heap:
        g_cur, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == end:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]

        cr, cc = cur
        for (nr, nc), step in _neighbors_8(rows, cols, cr, cc):
            if _wall(costmap, nr, nc):
                continue
            if not _no_corner_cut(costmap, cr, cc, nr, nc):
                continue
            cost = _edge_cost(costmap, cur, (nr, nc), step, cost_weight)
            tentative = g_cur + cost
            if tentative < g[nr, nc]:
                g[nr, nc] = tentative
                came_from[(nr, nc)] = cur
                heapq.heappush(open_heap, (tentative, (nr, nc)))

    return None


def _line_collision(costmap, a, b, n_checks=20):
    """True if any point on segment a->b is in wall."""
    for i in range(n_checks + 1):
        t = i / n_checks
        r = int(a[0] * (1 - t) + b[0] * t + 0.5)
        c = int(a[1] * (1 - t) + b[1] * t + 0.5)
        if 0 <= r < costmap.shape[0] and 0 <= c < costmap.shape[1]:
            if _wall(costmap, r, c):
                return True
    return False


def _path_length(costmap, waypoints, cost_weight):
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        step = np.hypot(b[0] - a[0], b[1] - a[1])
        c_mid = 0.5 * (costmap[a[0], a[1]] + costmap[b[0], b[1]])
        total += step * (1.0 + cost_weight * c_mid)
    return total


def rrt_star_search(costmap, start, end, cost_weight=0.0, max_iter=2000, step_size=8.0):
    """RRT* on costmap. Returns path as list of (r,c) or None."""
    rows, cols = costmap.shape
    start = tuple(int(x) for x in start)
    end = tuple(int(x) for x in end)
    if _wall(costmap, start[0], start[1]) or _wall(costmap, end[0], end[1]):
        return None

    nodes = [start]
    parent = {}
    cost_to = {start: 0.0}
    radius = step_size * 1.5

    def dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def steer(from_pt, to_pt, step):
        d = dist(from_pt, to_pt)
        if d <= step:
            return to_pt
        t = step / d
        return (int(from_pt[0] + t * (to_pt[0] - from_pt[0])),
                int(from_pt[1] + t * (to_pt[1] - from_pt[1])))

    for _ in range(max_iter):
        if np.random.random() < 0.1:
            rand = end
        else:
            rand = (np.random.randint(0, rows), np.random.randint(0, cols))
        if _wall(costmap, rand[0], rand[1]):
            continue

        # nearest
        best = min(nodes, key=lambda n: dist(n, rand))
        new_pt = steer(best, rand, step_size)
        new_pt = (max(0, min(rows - 1, new_pt[0])), max(0, min(cols - 1, new_pt[1])))
        if _wall(costmap, new_pt[0], new_pt[1]):
            continue
        if _line_collision(costmap, best, new_pt):
            continue

        edge_cost_val = _path_length(costmap, [best, new_pt], cost_weight)
        new_cost = cost_to[best] + edge_cost_val

        # near neighbors
        near = [n for n in nodes if dist(n, new_pt) <= radius]
        best_parent = best
        best_cost = new_cost
        for n in near:
            if _line_collision(costmap, n, new_pt):
                continue
            c = cost_to[n] + _path_length(costmap, [n, new_pt], cost_weight)
            if c < best_cost:
                best_cost = c
                best_parent = n
        new_cost = best_cost
        best = best_parent
        edge_cost_val = _path_length(costmap, [best, new_pt], cost_weight)
        new_cost = cost_to[best] + edge_cost_val

        nodes.append(new_pt)
        parent[new_pt] = best
        cost_to[new_pt] = new_cost

        # rewire
        for n in near:
            if n == best:
                continue
            if _line_collision(costmap, new_pt, n):
                continue
            via_new = cost_to[new_pt] + _path_length(costmap, [new_pt, n], cost_weight)
            if via_new < cost_to[n]:
                parent[n] = new_pt
                cost_to[n] = via_new

    # path from start to end (nearest node to end)
    end_nearest = min(nodes, key=lambda n: dist(n, end))
    if dist(end_nearest, end) > step_size * 2:
        return None
    if _line_collision(costmap, end_nearest, end):
        path_nodes = [end_nearest]
    else:
        path_nodes = [end]

    cur = end_nearest
    while cur in parent:
        path_nodes.append(cur)
        cur = parent[cur]
    path_nodes.append(start)
    return path_nodes[::-1]
