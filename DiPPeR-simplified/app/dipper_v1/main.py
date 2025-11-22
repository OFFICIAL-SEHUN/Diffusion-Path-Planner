import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import heapq

class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_value=0, initial_estimate_error=1):
        self.q = process_noise
        self.r = measurement_noise
        self.x = initial_value
        self.p = initial_estimate_error
    def update(self, measurement):
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x

# --- 1. Hyperparameters and Settings ---
IMG_SIZE = 32
HORIZON = 64
TIMESTEPS = 200
BATCH_SIZE = 32
EPOCHS = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 2. Data Generation (Complex Maze) & Caching ---
CACHE_SIZE = 500
data_cache = []

def generate_maze_and_path(img_size, horizon):
    scale = 4
    maze_size = img_size // scale
    grid = np.zeros((maze_size, maze_size), dtype=np.uint8)
    def is_valid(x, y):
        return 0 <= x < maze_size and 0 <= y < maze_size
    start_x, start_y = random.randint(0, maze_size//2-1)*2, random.randint(0, maze_size-1)
    stack = [(start_x, start_y)]
    grid[start_y, start_x] = 1
    while stack:
        cx, cy = stack[-1]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        found_neighbor = False
        for dx, dy in directions:
            nx, ny = cx + dx*2, cy + dy*2
            if is_valid(nx, ny) and grid[ny, nx] == 0:
                grid[cy+dy, cx+dx] = 1
                grid[ny, nx] = 1
                stack.append((nx, ny))
                found_neighbor = True
                break
        if not found_neighbor:
            stack.pop()
    start_node = (start_x, start_y)
    end_candidates = []
    for y in range(maze_size):
        if grid[y, maze_size-1] == 1:
            end_candidates.append((maze_size-1, y))
    if not end_candidates:
        max_x = -1
        exit_y = -1
        for y_ in range(maze_size):
            for x_ in range(maze_size-1, -1, -1):
                if grid[y_, x_] == 1:
                    if x_ > max_x:
                        max_x = x_
                        exit_y = y_
                    break
        if exit_y != -1:
            for x_ in range(max_x+1, maze_size):
                grid[exit_y, x_] = 1
            end_candidates.append((maze_size-1, exit_y))
    if not end_candidates: return None, None
    end_node = random.choice(end_candidates)
    def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
    open_set = []
    heapq.heappush(open_set, (heuristic(start_node, end_node), start_node))
    closed_set, came_from = set(), {}
    g_score = { (x,y): float('inf') for x in range(maze_size) for y in range(maze_size) }
    g_score[start_node] = 0
    path_found = False
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed_set: continue
        closed_set.add(current)
        if current == end_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path.reverse()
            path_found = True
            break
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze_size and 0 <= neighbor[1] < maze_size and grid[neighbor[1], neighbor[0]] == 1:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end_node)
                    heapq.heappush(open_set, (f_score, neighbor))
    if not path_found: return None, None
    path_np = np.array(path, dtype=np.float32)
    path_distances = np.cumsum(np.sqrt(np.sum(np.diff(path_np, axis=0)**2, axis=1)))
    path_distances = np.insert(path_distances, 0, 0)
    interp_distances = np.linspace(0, path_distances[-1], horizon)
    # Use np.interp for smoother path interpolation
    interp_x = np.interp(interp_distances, path_distances, path_np[:, 0])
    interp_y = np.interp(interp_distances, path_distances, path_np[:, 1])
    final_path = np.vstack([interp_x, interp_y]).T
    map_img = np.kron(1 - grid, np.ones((scale, scale))) # Obstacles are 0 (black)
    final_path_scaled = final_path * scale + scale / 2.0
    return map_img, final_path_scaled

def populate_cache():
    print(f"Populating data cache with {CACHE_SIZE} samples...")
    for _ in tqdm(range(CACHE_SIZE)):
        map_img, trajectory = generate_maze_and_path(IMG_SIZE, HORIZON)
        if map_img is None: continue
        maps = torch.from_numpy(map_img).float().unsqueeze(0).unsqueeze(0)
        trajectories = (torch.from_numpy(trajectory).float() / IMG_SIZE) * 2 - 1
        data_cache.append((maps, trajectories.unsqueeze(0)))
    print("Cache populated.")

def get_data_batch(batch_size):
    if not data_cache: populate_cache()
    maps_list, traj_list = [], []
    for _ in range(batch_size):
        maps, trajectories = random.choice(data_cache)
        maps_list.append(maps)
        traj_list.append(trajectories)
    return torch.cat(maps_list).to(device), torch.cat(traj_list).to(device)


# --- 3. Diffusion Scheduler ---
betas = torch.linspace(0.0001, 0.02, TIMESTEPS, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

def forward_diffusion_sample(x0, t, device=device):
    noise = torch.randn_like(x0)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None]
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])[:, None, None]
    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# --- 4. Model Architecture ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = IMG_SIZE * 4
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_dim), nn.Linear(time_dim, time_dim), nn.ReLU())
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128 * (IMG_SIZE//8)**2, 512), nn.ReLU())
        self.path_processor = nn.Sequential(
            nn.Linear(HORIZON * 2 + 512 + time_dim, 2048), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(2048, 2048), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, HORIZON * 2))
    def forward(self, noisy_path, map_condition, t):
        map_features = self.map_encoder(map_condition)
        time_features = self.time_mlp(t)
        path_flat = noisy_path.view(noisy_path.shape[0], -1)
        combined = torch.cat([path_flat, map_features, time_features], dim=1)
        predicted_noise_flat = self.path_processor(combined)
        return predicted_noise_flat.view(noisy_path.shape)

model = ConditionalUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- 5. Training Loop ---
print("Starting training...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    maps, trajectories = get_data_batch(BATCH_SIZE)
    t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=device).long()
    noisy_trajectories, noise = forward_diffusion_sample(trajectories, t)
    predicted_noise = model(noisy_trajectories, maps, t)
    loss = F.mse_loss(noise, predicted_noise)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
print("Training finished.")

# --- 6. Model Saving ---
model_path = "dipper_simple_obstacle_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# --- 7. Sampling and Visualization ---
@torch.no_grad()
def sample(model, maps):
    model.eval()
    path = torch.randn((maps.shape[0], HORIZON, 2), device=device)
    for i in tqdm(reversed(range(0, TIMESTEPS)), desc="Sampling", total=TIMESTEPS):
        t = torch.full((maps.shape[0],), i, device=device, dtype=torch.long)
        predicted_noise = model(path, maps, t)
        alpha_t = alphas[t][:, None, None]
        alpha_cumprod_t = alphas_cumprod[t][:, None, None]
        beta_t = betas[t][:, None, None]
        term1 = 1 / torch.sqrt(alpha_t)
        term2 = (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        path = term1 * (path - term2)
        if i > 0:
            z = torch.randn_like(path)
            path += torch.sqrt(beta_t) * z
    return path

def plot_results(maps, generated_paths, true_paths):
    generated_paths_scaled = (generated_paths + 1) / 2 * IMG_SIZE
    true_paths_scaled = (true_paths + 1) / 2 * IMG_SIZE
    gen_path = generated_paths_scaled[0].cpu().numpy()
    true_path = true_paths_scaled[0].cpu().numpy()
    kf_x = SimpleKalmanFilter(0.01, 0.7, gen_path[0, 0])
    kf_y = SimpleKalmanFilter(0.01, 0.7, gen_path[0, 1])
    smoothed_path = np.zeros_like(gen_path)
    for i in range(len(gen_path)):
        smoothed_path[i, 0] = kf_x.update(gen_path[i, 0])
        smoothed_path[i, 1] = kf_y.update(gen_path[i, 1])
    map_img = maps[0, 0].cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(1 - map_img, cmap='gray', origin='lower') # Invert map for plotting
    plt.plot(gen_path[:, 0], gen_path[:, 1], 'r:', alpha=0.5, label='Original Generated Path')
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'm-', linewidth=2, label='Smoothed Path (Kalman)')
    plt.plot(true_path[:, 0], true_path[:, 1], 'g--', label='True Path')
    plt.scatter(smoothed_path[0, 0], smoothed_path[0, 1], c='m', marker='o', s=50, label='Start')
    plt.scatter(smoothed_path[-1, 0], smoothed_path[-1, 1], c='m', marker='x', s=50, label='End')
    plt.legend()
    plt.title("DiPPeR with Kalman Filter Smoothing (Simple Obstacle)")
    plt.xlim(0, IMG_SIZE)
    plt.ylim(0, IMG_SIZE)
    plt.show()

# --- 8. Inference ---
print("\n--- Starting Inference ---")
inference_model = ConditionalUNet().to(device)
inference_model.load_state_dict(torch.load(model_path, map_location=device))
print("Loaded trained model for inference.")
print("Generating a new test sample...")
test_map_np, true_traj_np = generate_maze_and_path(IMG_SIZE, HORIZON)
if test_map_np is not None:
    test_maps = torch.from_numpy(test_map_np).float().unsqueeze(0).unsqueeze(0).to(device)
    true_trajectories = (torch.from_numpy(true_traj_np).float() / IMG_SIZE) * 2 - 1
    true_trajectories = true_trajectories.unsqueeze(0).to(device)
    generated_trajectories = sample(inference_model, test_maps)
    plot_results(test_maps, generated_trajectories, true_trajectories)
else:
    print("Failed to generate a test sample.")
