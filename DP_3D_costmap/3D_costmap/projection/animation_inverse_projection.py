import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DualViewMappingSim:
    def __init__(self):
        # --- 1. 월드 설정 ---
        self.map_size = 12.0 
        self.grid_res = 0.1  
        self.grid_dim = int(self.map_size / self.grid_res)
        
        # 전역 지도 (Global Map)
        self.global_map = np.ones((self.grid_dim, self.grid_dim)) * 0.5 
        self.visited_mask = np.zeros((self.grid_dim, self.grid_dim), dtype=bool)
        self.visit_count = np.zeros((self.grid_dim, self.grid_dim), dtype=float)

        # --- GT 경로 생성 ---
        num_frames = 150
        t = np.linspace(0, 12, num_frames)
        self.path_x = t * 0.6 + 2.0
        self.path_y = 2.0 * np.sin(t * 0.7) + 6.0
        self.path_yaw = np.zeros_like(t)
        
        for i in range(len(t)-1):
            dx = self.path_x[i+1] - self.path_x[i]
            dy = self.path_y[i+1] - self.path_y[i]
            self.path_yaw[i] = np.arctan2(dy, dx)
        self.path_yaw[-1] = self.path_yaw[-2]

        # --- 2. 카메라 파라미터 ---
        self.cam_max_range = 3.5 
        self.cam_fov_deg = 60    
        self.cam_fov_rad = np.radians(self.cam_fov_deg)
        self.img_w, self.img_h = 64, 48 

    def world_to_local(self, wx, wy, rx, ry, ryaw):
        dx = wx - rx
        dy = wy - ry
        cos_a = np.cos(ryaw)
        sin_a = np.sin(ryaw)
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        return lx, ly

    def local_to_pixel(self, lx, ly):
        if lx <= 0.5 or lx > self.cam_max_range: return None, None
        
        v_norm = (self.cam_max_range - lx) / (self.cam_max_range - 0.5)
        v = (v_norm * (self.img_h * 2/3)) + (self.img_h / 3)
        
        half_width = lx * np.tan(self.cam_fov_rad / 2)
        u_ratio = 0.5 - (ly / (2 * half_width))
        u = u_ratio * self.img_w
        
        return u, v

    def get_camera_view(self, frame):
        """[왼쪽 화면] GT 경로 투영 + 길 폭 확대"""
        camera_img = np.ones((self.img_h, self.img_w)) * 0.9 # 배경: 위험
        
        rx = self.path_x[frame]
        ry = self.path_y[frame]
        ryaw = self.path_yaw[frame]
        
        camera_img[:int(self.img_h/3), :] = np.nan

        future_idx = range(frame, min(frame + 40, len(self.path_x)))
        path_uvs = []
        
        for i in future_idx:
            wx = self.path_x[i]
            wy = self.path_y[i]
            lx, ly = self.world_to_local(wx, wy, rx, ry, ryaw)
            u, v = self.local_to_pixel(lx, ly)
            if u is not None and 0 <= u < self.img_w and 0 <= v < self.img_h:
                path_uvs.append([u, v])

        if not path_uvs: return camera_img

        path_uvs = np.array(path_uvs)
        
        v_grid, u_grid = np.meshgrid(np.arange(self.img_h), np.arange(self.img_w), indexing='ij')
        mask_ground = v_grid >= self.img_h/3
        pixels_v = v_grid[mask_ground]
        pixels_u = u_grid[mask_ground]
        
        min_dists = np.zeros_like(pixels_v, dtype=float) + 999.0
        
        for pu, pv in path_uvs:
            d2 = (pixels_u - pu)**2 + (pixels_v - pv)**2
            min_dists = np.minimum(min_dists, d2)
            
        min_dists = np.sqrt(min_dists)
        
        # --- 수정된 부분: 길 폭 확대 ---
        # 기존: threshold = 6.0 (좁음)
        # 변경: threshold = 12.0 (넓음)
        threshold = 12.0 
        scores = np.clip(min_dists / threshold, 0, 1) 
        
        final_vals = 0.1 + scores * 0.8
        camera_img[mask_ground] = final_vals
        camera_img[mask_ground] += np.random.normal(0, 0.05, camera_img[mask_ground].shape)
        camera_img = np.clip(camera_img, 0, 1)
        
        return camera_img

    def update_global_map(self, frame, camera_img):
        rx = self.path_x[frame]
        ry = self.path_y[frame]
        ryaw = self.path_yaw[frame]

        idx_x = int(rx / self.grid_res)
        idx_y = int(ry / self.grid_res)
        search_range = int(self.cam_max_range / self.grid_res) + 2
        
        min_x = max(0, idx_x - search_range)
        max_x = min(self.grid_dim, idx_x + search_range)
        min_y = max(0, idx_y - search_range)
        max_y = min(self.grid_dim, idx_y + search_range)

        ix_grid, iy_grid = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
        gx_grid = ix_grid * self.grid_res
        gy_grid = iy_grid * self.grid_res
        
        dx = gx_grid - rx
        dy = gy_grid - ry
        cos_a = np.cos(ryaw)
        sin_a = np.sin(ryaw)
        local_x = dx * cos_a + dy * sin_a
        local_y = -dx * sin_a + dy * cos_a
        
        fov_angle = np.abs(np.arctan2(local_y, local_x))
        fov_margin = 0.85
        valid_mask = ((local_x > 0.5) & (local_x < self.cam_max_range)
                      & (fov_angle < self.cam_fov_rad / 2 * fov_margin))
        if not np.any(valid_mask): return

        valid_lx = local_x[valid_mask]
        valid_ly = local_y[valid_mask]
        
        v_norm = (self.cam_max_range - valid_lx) / (self.cam_max_range - 0.5)
        v_idx = ((v_norm * (self.img_h * 2/3)) + (self.img_h / 3)).astype(int)
        
        half_width = valid_lx * np.tan(self.cam_fov_rad / 2)
        u_ratio = 0.5 - (valid_ly / (2 * half_width))
        u_idx = (u_ratio * self.img_w).astype(int)
        
        in_img_mask = (v_idx >= 0) & (v_idx < self.img_h) & (u_idx >= 0) & (u_idx < self.img_w)
        
        final_v = v_idx[in_img_mask]
        final_u = u_idx[in_img_mask]
        valid_ix = ix_grid[valid_mask][in_img_mask]
        valid_iy = iy_grid[valid_mask][in_img_mask]
        
        observed_vals = camera_img[final_v, final_u]
        not_nan = ~np.isnan(observed_vals)
        
        target_ix = valid_ix[not_nan]
        target_iy = valid_iy[not_nan]
        vals = observed_vals[not_nan]
        
        current_vals = self.global_map[target_iy, target_ix]
        current_count = self.visit_count[target_iy, target_ix]
        visited = self.visited_mask[target_iy, target_ix]

        dist = np.sqrt((target_ix * self.grid_res - rx)**2
                       + (target_iy * self.grid_res - ry)**2)
        confidence = np.clip(1.0 - dist / self.cam_max_range, 0.1, 1.0)

        new_count = current_count + confidence
        new_vals = np.where(visited,
                            (current_vals * current_count + vals * confidence) / new_count,
                            vals)

        self.global_map[target_iy, target_ix] = new_vals
        self.visit_count[target_iy, target_ix] = new_count
        self.visited_mask[target_iy, target_ix] = True

    def animate(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.set_title("1. Wider Camera View (Projected GT)")
        self.im_cam = ax1.imshow(np.zeros((self.img_h, self.img_w)), cmap='RdYlGn_r', vmin=0, vmax=1)
        ax1.set_xlabel("Pixel u")
        ax1.set_ylabel("Pixel v")

        ax2.set_title("2. Precise Global Mapping")
        self.im_map = ax2.imshow(self.global_map, cmap='RdYlGn_r', vmin=0, vmax=1,
                                 origin='lower', extent=[0, self.map_size, 0, self.map_size])
        
        ax2.plot(self.path_x, self.path_y, 'w--', linewidth=2, label='GT Path', zorder=10)
        
        self.robot_dot, = ax2.plot([], [], 'bo', ms=8, zorder=20)
        self.robot_arrow = ax2.quiver([0], [0], [0], [0], color='blue', scale=20, zorder=20)
        self.fov_lines, = ax2.plot([], [], 'k--', alpha=0.5, lw=1, zorder=15)
        
        ax2.legend(loc='upper right')
        plt.tight_layout()

        def update(frame):
            if frame >= len(self.path_x): return
            
            cam_img = self.get_camera_view(frame)
            self.im_cam.set_data(cam_img)
            
            self.update_global_map(frame, cam_img)
            self.im_map.set_data(self.global_map)
            
            rx = self.path_x[frame]
            ry = self.path_y[frame]
            ryaw = self.path_yaw[frame]
            
            self.robot_dot.set_data([rx], [ry])
            self.robot_arrow.set_offsets([rx, ry])
            self.robot_arrow.set_UVC(np.cos(ryaw), np.sin(ryaw))
            
            lx = rx + self.cam_max_range * np.cos(ryaw + self.cam_fov_rad/2)
            ly = ry + self.cam_max_range * np.sin(ryaw + self.cam_fov_rad/2)
            rx_pt = rx + self.cam_max_range * np.cos(ryaw - self.cam_fov_rad/2)
            ry_pt = ry + self.cam_max_range * np.sin(ryaw - self.cam_fov_rad/2)
            self.fov_lines.set_data([rx, lx, rx_pt, rx], [ry, ly, ry_pt, ry])
            
            return self.im_cam, self.im_map, self.robot_dot, self.robot_arrow, self.fov_lines

        ani = FuncAnimation(fig, update, frames=len(self.path_x), interval=50, blit=False)
        plt.show()

sim = DualViewMappingSim()
sim.animate()