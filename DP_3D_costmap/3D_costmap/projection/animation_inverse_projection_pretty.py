import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap


class EnhancedDualViewMappingSim:
    def __init__(self):
        self.map_size = 12.0
        self.grid_res = 0.1
        self.grid_dim = int(self.map_size / self.grid_res)

        self.global_map = np.ones((self.grid_dim, self.grid_dim)) * 0.5
        self.visited_mask = np.zeros((self.grid_dim, self.grid_dim), dtype=bool)
        self.visit_count = np.zeros((self.grid_dim, self.grid_dim), dtype=float)

        num_frames = 170
        t = np.linspace(0, 12, num_frames)
        self.path_x = t * 0.58 + 2.0
        self.path_y = 6.0 + 1.55 * np.sin(t * 0.72) + 0.35 * np.sin(t * 1.75)
        self.path_yaw = np.zeros_like(t)

        for i in range(len(t) - 1):
            dx = self.path_x[i + 1] - self.path_x[i]
            dy = self.path_y[i + 1] - self.path_y[i]
            self.path_yaw[i] = np.arctan2(dy, dx)
        self.path_yaw[-1] = self.path_yaw[-2]

        self.cam_max_range = 3.5
        self.cam_fov_deg = 60
        self.cam_fov_rad = np.radians(self.cam_fov_deg)
        self.img_w, self.img_h = 96, 64

        self.rng = np.random.default_rng(7)
        self.static_camera_texture = self._make_camera_texture()
        self.last_path_uvs = np.empty((0, 2))

        self.cost_cmap = self._make_cost_cmap()
        self.camera_cmap = self._make_camera_cmap()

    def _make_cost_cmap(self):
        colors = [
            "#1f3b73",  # deep blue: very easy terrain
            "#1ba784",  # teal
            "#f7d154",  # sand
            "#f47c48",  # orange
            "#ba2f5d",  # magenta: risky terrain
        ]
        cmap = LinearSegmentedColormap.from_list("terrain_cost_rich", colors, N=256)
        cmap.set_bad("#101522")
        return cmap

    def _make_camera_cmap(self):
        colors = [
            "#153e90",
            "#12b8a6",
            "#d4e157",
            "#ff9f1c",
            "#d7265b",
        ]
        cmap = LinearSegmentedColormap.from_list("camera_cost_rich", colors, N=256)
        cmap.set_bad("#202a44")
        return cmap

    def _make_camera_texture(self):
        low_h, low_w = 8, 12
        coarse = self.rng.normal(0.0, 1.0, (low_h, low_w))
        tex = np.kron(coarse, np.ones((self.img_h // low_h, self.img_w // low_w)))
        tex = tex[: self.img_h, : self.img_w]
        tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-9)
        return tex - 0.5

    def world_to_local(self, wx, wy, rx, ry, ryaw):
        dx = wx - rx
        dy = wy - ry
        cos_a = np.cos(ryaw)
        sin_a = np.sin(ryaw)
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        return lx, ly

    def local_to_pixel(self, lx, ly):
        if lx <= 0.5 or lx > self.cam_max_range:
            return None, None

        v_norm = (self.cam_max_range - lx) / (self.cam_max_range - 0.5)
        v = (v_norm * (self.img_h * 2 / 3)) + (self.img_h / 3)

        half_width = lx * np.tan(self.cam_fov_rad / 2)
        u_ratio = 0.5 - (ly / (2 * half_width))
        u = u_ratio * self.img_w
        return u, v

    def get_camera_view(self, frame):
        camera_img = np.ones((self.img_h, self.img_w)) * np.nan

        rx = self.path_x[frame]
        ry = self.path_y[frame]
        ryaw = self.path_yaw[frame]

        future_idx = range(frame, min(frame + 48, len(self.path_x)))
        path_uvs = []
        for i in future_idx:
            lx, ly = self.world_to_local(self.path_x[i], self.path_y[i], rx, ry, ryaw)
            u, v = self.local_to_pixel(lx, ly)
            if u is not None and 0 <= u < self.img_w and 0 <= v < self.img_h:
                path_uvs.append([u, v])

        self.last_path_uvs = np.array(path_uvs) if path_uvs else np.empty((0, 2))

        v_grid, u_grid = np.meshgrid(np.arange(self.img_h), np.arange(self.img_w), indexing="ij")
        ground_mask = v_grid >= self.img_h / 3
        ground_u = u_grid[ground_mask]
        ground_v = v_grid[ground_mask]

        if len(self.last_path_uvs) > 0:
            min_dists = np.full_like(ground_v, 999.0, dtype=float)
            for pu, pv in self.last_path_uvs:
                d2 = (ground_u - pu) ** 2 + (ground_v - pv) ** 2
                min_dists = np.minimum(min_dists, d2)
            min_dists = np.sqrt(min_dists)
        else:
            min_dists = np.full_like(ground_v, 999.0, dtype=float)

        road_core = np.exp(-((min_dists / 7.5) ** 2))
        road_shoulder = np.exp(-((min_dists / 18.0) ** 2))
        depth = (ground_v - self.img_h / 3) / (self.img_h * 2 / 3)
        lateral = np.abs((ground_u / max(1, self.img_w - 1)) - 0.5) * 2.0

        texture = self.static_camera_texture[ground_mask]
        shimmer = 0.035 * np.sin(frame * 0.15 + ground_u * 0.19 + ground_v * 0.07)
        rocks = 0.05 * np.sin(ground_u * 0.33) * np.cos(ground_v * 0.21 + frame * 0.04)

        cost = (
            0.76
            + 0.12 * depth
            + 0.08 * lateral
            + 0.07 * texture
            + shimmer
            + rocks
            - 0.68 * road_core
            - 0.16 * road_shoulder
        )
        camera_img[ground_mask] = np.clip(cost, 0.02, 0.98)
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
        valid_mask = (
            (local_x > 0.5)
            & (local_x < self.cam_max_range)
            & (fov_angle < self.cam_fov_rad / 2 * 0.9)
        )
        if not np.any(valid_mask):
            return

        valid_lx = local_x[valid_mask]
        valid_ly = local_y[valid_mask]

        v_norm = (self.cam_max_range - valid_lx) / (self.cam_max_range - 0.5)
        v_idx = ((v_norm * (self.img_h * 2 / 3)) + (self.img_h / 3)).astype(int)

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

        dist = np.sqrt((target_ix * self.grid_res - rx) ** 2 + (target_iy * self.grid_res - ry) ** 2)
        confidence = np.clip(1.0 - dist / self.cam_max_range, 0.12, 1.0)
        new_count = current_count + confidence
        new_vals = np.where(
            visited,
            (current_vals * current_count + vals * confidence) / new_count,
            vals,
        )

        self.global_map[target_iy, target_ix] = new_vals
        self.visit_count[target_iy, target_ix] = new_count
        self.visited_mask[target_iy, target_ix] = True

    def animate(self, save_path=None, fps=22):
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(14, 7), facecolor="#0b1020")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.12)
        ax_cam = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[0, 1])

        ax_cam.set_title("Forward Camera Cost Projection", fontsize=13, pad=12)
        self.im_cam = ax_cam.imshow(
            np.zeros((self.img_h, self.img_w)),
            cmap=self.camera_cmap,
            vmin=0,
            vmax=1,
            interpolation="bicubic",
        )
        self.camera_path_line, = ax_cam.plot([], [], color="#f8f9fa", lw=2.5, alpha=0.95)
        self.camera_path_glow, = ax_cam.plot([], [], color="#00e5ff", lw=7.0, alpha=0.22)
        ax_cam.axhline(self.img_h / 3, color="#dbeafe", lw=1.2, alpha=0.35)
        ax_cam.text(2, self.img_h / 3 - 3, "sky / unobserved", color="#cbd5e1", fontsize=8)
        ax_cam.set_xlabel("pixel u")
        ax_cam.set_ylabel("pixel v")
        ax_cam.set_facecolor("#111827")

        ax_map.set_title("Inverse Projection Into Global Cost Map", fontsize=13, pad=12)
        self.im_map = ax_map.imshow(
            self.global_map,
            cmap=self.cost_cmap,
            vmin=0,
            vmax=1,
            origin="lower",
            extent=[0, self.map_size, 0, self.map_size],
            interpolation="bilinear",
        )
        ax_map.plot(self.path_x, self.path_y, color="#e5e7eb", ls="--", lw=1.8, alpha=0.7, label="GT path")
        self.trail_line, = ax_map.plot([], [], color="#38bdf8", lw=3.0, alpha=0.85, label="visited")
        self.robot_dot, = ax_map.plot([], [], marker="o", ms=10, color="#facc15", mec="#111827", mew=1.5)
        self.robot_arrow = ax_map.quiver(
            [0],
            [0],
            [0],
            [0],
            color="#facc15",
            scale=18,
            width=0.006,
            zorder=20,
        )
        self.fov_fill = ax_map.fill([], [], color="#38bdf8", alpha=0.14, zorder=12)[0]
        self.fov_lines, = ax_map.plot([], [], color="#7dd3fc", alpha=0.8, lw=1.4, zorder=15)

        ax_map.set_xlim(0, self.map_size)
        ax_map.set_ylim(0, self.map_size)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("world x [m]")
        ax_map.set_ylabel("world y [m]")
        ax_map.grid(color="#94a3b8", alpha=0.15, lw=0.8)
        ax_map.legend(loc="upper right", framealpha=0.25)
        ax_map.set_facecolor("#0f172a")

        cbar = fig.colorbar(self.im_map, ax=[ax_cam, ax_map], fraction=0.035, pad=0.025)
        cbar.set_label("terrain traversal cost")
        cbar.ax.tick_params(colors="#cbd5e1")
        cbar.outline.set_edgecolor("#475569")

        status = fig.text(0.5, 0.035, "", ha="center", color="#cbd5e1", fontsize=10)
        fig.suptitle("Richer Inverse-Projection Animation", fontsize=16, color="#f8fafc", y=0.98)

        def update(frame):
            cam_img = self.get_camera_view(frame)
            self.im_cam.set_data(cam_img)

            if len(self.last_path_uvs) > 1:
                self.camera_path_glow.set_data(self.last_path_uvs[:, 0], self.last_path_uvs[:, 1])
                self.camera_path_line.set_data(self.last_path_uvs[:, 0], self.last_path_uvs[:, 1])
            else:
                self.camera_path_glow.set_data([], [])
                self.camera_path_line.set_data([], [])

            self.update_global_map(frame, cam_img)
            self.im_map.set_data(self.global_map)

            rx = self.path_x[frame]
            ry = self.path_y[frame]
            ryaw = self.path_yaw[frame]

            self.trail_line.set_data(self.path_x[: frame + 1], self.path_y[: frame + 1])
            self.robot_dot.set_data([rx], [ry])
            self.robot_arrow.set_offsets([rx, ry])
            self.robot_arrow.set_UVC(np.cos(ryaw), np.sin(ryaw))

            left_x = rx + self.cam_max_range * np.cos(ryaw + self.cam_fov_rad / 2)
            left_y = ry + self.cam_max_range * np.sin(ryaw + self.cam_fov_rad / 2)
            right_x = rx + self.cam_max_range * np.cos(ryaw - self.cam_fov_rad / 2)
            right_y = ry + self.cam_max_range * np.sin(ryaw - self.cam_fov_rad / 2)

            fov_x = [rx, left_x, right_x, rx]
            fov_y = [ry, left_y, right_y, ry]
            self.fov_fill.set_xy(np.column_stack([fov_x, fov_y]))
            self.fov_lines.set_data(fov_x, fov_y)

            status.set_text(
                f"frame {frame + 1:03d}/{len(self.path_x)}  |  "
                f"observed cells: {int(self.visited_mask.sum())}"
            )

            return (
                self.im_cam,
                self.camera_path_glow,
                self.camera_path_line,
                self.im_map,
                self.trail_line,
                self.robot_dot,
                self.robot_arrow,
                self.fov_fill,
                self.fov_lines,
                status,
            )

        ani = FuncAnimation(fig, update, frames=len(self.path_x), interval=45, blit=False)
        if save_path:
            ani.save(save_path, writer=PillowWriter(fps=fps), dpi=110)
            plt.close(fig)
        else:
            plt.show()
        return ani


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Animate inverse projection with a richer visual style.")
    parser.add_argument("--save-gif", default=None, help="Output GIF path. If omitted, opens the animation window.")
    parser.add_argument("--fps", type=int, default=22, help="Frames per second when saving a GIF.")
    args = parser.parse_args()

    sim = EnhancedDualViewMappingSim()
    sim.animate(save_path=args.save_gif, fps=args.fps)
