"""
Depth-aware 3D height-map visualizer.

목표:
- path가 지형 앞에 있는 구간은 선명하게 표시
- path가 지형 뒤로 가려지는 구간은 완전히 사라지지 않고 흐릿하게 표시

Usage:
    python3 scripts/visualize_3d.py \
        --pt data/raw/terrain_00001.pt \
        --intents avoid_steep \
        --elev 20 --azim 250 \
        --save results/3d_vis/depth_fade.png

권장 옵션:
    --surface-alpha 0.88
    --offset 0.05
    --hidden-alpha 0.22
    --depth-tol 0.0015
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

# 서버 / SSH 환경에서는 저장 전용 backend 사용
if not os.environ.get("DISPLAY", ""):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ─────────────────────────────────────────────
# Basic utilities
# ─────────────────────────────────────────────

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_pt(pt_path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(pt_path, map_location="cpu")


def _path_to_pixels(path: Any, width: int, height: int, img_size: int | None = None) -> np.ndarray:
    """
    path를 pixel coordinate [N, 2] = (col, row)로 변환한다.

    - 입력이 [-1, 1] 근처이면 normalized coordinate로 간주한다.
    - 아니면 이미 pixel coordinate라고 간주한다.
    """
    p = _to_numpy(path).astype(np.float64)

    if p.size == 0:
        return p.reshape(0, 2)

    if p.ndim == 1 and p.shape[0] == 2:
        p = p.reshape(1, 2)

    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"path shape must be [N, 2], got {p.shape}")

    valid = np.isfinite(p).all(axis=1)
    p = p[valid]
    if len(p) == 0:
        return p.reshape(0, 2)

    # Diffusion path는 보통 [-1, 1] normalized coordinate이다.
    if np.nanmax(np.abs(p)) <= 1.5:
        sx = float(img_size if img_size is not None else width - 1)
        sy = float(img_size if img_size is not None else height - 1)
        col = (p[:, 0] + 1.0) / 2.0 * sx
        row = (p[:, 1] + 1.0) / 2.0 * sy
    else:
        col = p[:, 0]
        row = p[:, 1]

    col = np.clip(col, 0, width - 1)
    row = np.clip(row, 0, height - 1)
    return np.stack([col, row], axis=1)


def _sample_height_bilinear(path_px: np.ndarray, height_map: np.ndarray) -> np.ndarray:
    """pixel coordinate [N,2]=(col,row)에서 height를 bilinear interpolation으로 샘플링."""
    if path_px.size == 0:
        return np.empty((0,), dtype=np.float64)

    h, w = height_map.shape
    rows = np.clip(path_px[:, 1], 0, h - 1)
    cols = np.clip(path_px[:, 0], 0, w - 1)

    r0 = np.floor(rows).astype(int)
    c0 = np.floor(cols).astype(int)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)

    dr = rows - r0
    dc = cols - c0

    z = (
        height_map[r0, c0] * (1.0 - dr) * (1.0 - dc)
        + height_map[r1, c0] * dr * (1.0 - dc)
        + height_map[r0, c1] * (1.0 - dr) * dc
        + height_map[r1, c1] * dr * dc
    )
    return z


def _row_col(pos: Any) -> tuple[float, float] | None:
    if pos is None:
        return None
    p = _to_numpy(pos).astype(np.float64).reshape(-1)
    if p.size < 2:
        return None
    return float(p[0]), float(p[1])


# ─────────────────────────────────────────────
# Occlusion-aware path utilities
# ─────────────────────────────────────────────

def _densify_path(path_px: np.ndarray, height_map: np.ndarray, z_offset: float, step_px: float) -> np.ndarray:
    """
    path segment를 잘게 쪼개서 [N, 3] = (x, y, z) path로 만든다.
    쪼개야 hidden/visible 전환 지점이 자연스럽다.
    """
    if len(path_px) < 2:
        return np.empty((0, 3), dtype=np.float64)

    step_px = max(float(step_px), 0.2)
    pieces: list[np.ndarray] = []

    for i in range(len(path_px) - 1):
        p0 = path_px[i]
        p1 = path_px[i + 1]
        dist = float(np.linalg.norm(p1 - p0))
        n = max(int(np.ceil(dist / step_px)), 1)
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        seg = p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]
        pieces.append(seg)

    pieces.append(path_px[-1:].copy())
    dense_xy = np.concatenate(pieces, axis=0)
    dense_z = _sample_height_bilinear(dense_xy, height_map) + float(z_offset)
    return np.column_stack([dense_xy[:, 0], dense_xy[:, 1], dense_z])


def _project_points(ax: Axes3D, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D data coordinate를 Matplotlib projected coordinate로 변환."""
    m = ax.get_proj()
    sx, sy, sz = proj3d.proj_transform(xyz[:, 0], xyz[:, 1], xyz[:, 2], m)
    return np.asarray(sx), np.asarray(sy), np.asarray(sz)


def _build_terrain_depth_buffer(
    ax: Axes3D,
    height_map: np.ndarray,
    buffer_res: int = 700,
    paint_radius: int = 3,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    현재 view에서 terrain surface의 approximate z-buffer를 만든다.

    Matplotlib 3D는 true z-buffer를 사용자에게 직접 제공하지 않으므로,
    terrain grid vertex를 현재 camera projection으로 투영한 뒤,
    screen-space grid에 가장 가까운 terrain depth를 저장한다.

    proj3d의 projected z는 일반적으로 값이 더 작을수록 camera에 더 가깝다.
    """
    h, w = height_map.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    terrain_xyz = np.column_stack([
        xx.ravel().astype(np.float64),
        yy.ravel().astype(np.float64),
        height_map.ravel().astype(np.float64),
    ])

    sx, sy, sz = _project_points(ax, terrain_xyz)
    valid = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(sz)
    sx, sy, sz = sx[valid], sy[valid], sz[valid]

    # projected coordinate의 범위를 terrain 기준으로 잡는다.
    # margin을 두어 path가 살짝 떠 있어도 buffer 안에 들어오게 한다.
    xmin, xmax = float(np.min(sx)), float(np.max(sx))
    ymin, ymax = float(np.min(sy)), float(np.max(sy))
    xmargin = max((xmax - xmin) * 0.06, 1e-9)
    ymargin = max((ymax - ymin) * 0.06, 1e-9)
    xmin -= xmargin
    xmax += xmargin
    ymin -= ymargin
    ymax += ymargin

    res = int(buffer_res)
    depth = np.full((res, res), np.inf, dtype=np.float64)

    bx = np.round((sx - xmin) / max(xmax - xmin, 1e-12) * (res - 1)).astype(int)
    by = np.round((sy - ymin) / max(ymax - ymin, 1e-12) * (res - 1)).astype(int)

    valid = (0 <= bx) & (bx < res) & (0 <= by) & (by < res)
    bx, by, sz = bx[valid], by[valid], sz[valid]

    # Terrain vertex 하나가 screen-space에서 작은 면적을 차지하도록 주변 pixel에도 paint한다.
    # 이렇게 해야 vertex 사이의 hole 때문에 hidden path가 잘못 visible로 판정되는 것을 줄일 수 있다.
    r = max(int(paint_radius), 0)
    flat = depth.ravel()
    for dy in range(-r, r + 1):
        yy2 = by + dy
        ok_y = (0 <= yy2) & (yy2 < res)
        for dx in range(-r, r + 1):
            xx2 = bx + dx
            ok = ok_y & (0 <= xx2) & (xx2 < res)
            if not np.any(ok):
                continue
            index = yy2[ok] * res + xx2[ok]
            np.minimum.at(flat, index, sz[ok])

    return depth, (xmin, xmax, ymin, ymax)


def _query_depth_buffer(
    sx: np.ndarray,
    sy: np.ndarray,
    depth_buffer: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """projected x/y 위치에서 terrain depth를 조회한다."""
    xmin, xmax, ymin, ymax = bounds
    res = depth_buffer.shape[0]

    bx = np.round((sx - xmin) / max(xmax - xmin, 1e-12) * (res - 1)).astype(int)
    by = np.round((sy - ymin) / max(ymax - ymin, 1e-12) * (res - 1)).astype(int)

    out = np.full_like(sx, np.inf, dtype=np.float64)
    valid = (0 <= bx) & (bx < res) & (0 <= by) & (by < res)
    out[valid] = depth_buffer[by[valid], bx[valid]]
    return out


def _compute_hidden_mask(
    ax: Axes3D,
    dense_xyz: np.ndarray,
    terrain_depth_buffer: np.ndarray,
    terrain_bounds: tuple[float, float, float, float],
    depth_tol: float,
) -> np.ndarray:
    """
    dense path point가 terrain 뒤에 있는지 판정한다.

    projected z 기준:
    - path_depth <= terrain_depth + tol  → terrain보다 camera 쪽에 있음 → visible
    - path_depth  > terrain_depth + tol  → terrain보다 뒤에 있음       → hidden
    """
    sx, sy, path_depth = _project_points(ax, dense_xyz)
    terrain_depth = _query_depth_buffer(sx, sy, terrain_depth_buffer, terrain_bounds)

    has_occluder = np.isfinite(terrain_depth)
    hidden = has_occluder & (path_depth > terrain_depth + float(depth_tol))
    return hidden


def _split_segments_by_visibility(
    dense_xyz: np.ndarray,
    hidden_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """dense path를 hidden segments / visible segments로 나눈다."""
    if len(dense_xyz) < 2:
        empty = np.empty((0, 2, 3), dtype=np.float64)
        return empty, empty

    pts0 = dense_xyz[:-1]
    pts1 = dense_xyz[1:]
    segs = np.stack([pts0, pts1], axis=1)

    # 두 endpoint 중 하나라도 hidden이면 해당 작은 segment는 hidden으로 취급한다.
    seg_hidden = hidden_mask[:-1] | hidden_mask[1:]
    hidden_segs = segs[seg_hidden]
    visible_segs = segs[~seg_hidden]
    return hidden_segs, visible_segs


def _add_segments(
    ax: Axes3D,
    segs: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
    linestyle: str = "solid",
) -> Line3DCollection | None:
    if len(segs) == 0:
        return None

    rgba = to_rgba(color, alpha=alpha)
    lc = Line3DCollection(
        segs,
        colors=[rgba],
        linewidths=float(linewidth),
        linestyles=linestyle,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection3d(lc)
    lc.set_zorder(zorder)
    return lc


# ─────────────────────────────────────────────
# Main visualization
# ─────────────────────────────────────────────

def visualize_3d(
    pt_path: str | Path,
    save_path: str | Path | None = None,
    filter_intents: list[str] | None = None,
    elev: float = 30.0,
    azim: float = 225.0,
    path_z_offset: float = 0.05,
    surface_alpha: float = 0.88,
    hidden_alpha: float = 0.22,
    path_linewidth: float = 3.0,
    visible_outline_width: float = 6.0,
    hidden_linewidth: float | None = None,
    hidden_dashed: bool = False,
    densify_step_px: float = 0.75,
    depth_buffer_res: int = 800,
    depth_buffer_radius: int = 3,
    depth_tol: float = 0.0015,
    marker_size: float = 120.0,
    z_exaggeration: float = 10.0,
    dpi: int = 150,
) -> None:
    pt_path = Path(pt_path)
    data = _load_pt(pt_path)

    height_map = _to_numpy(data["height_map"]).astype(np.float64)
    height_map = np.squeeze(height_map)
    if height_map.ndim != 2:
        raise ValueError(f"height_map must be 2D, got {height_map.shape}")

    raw_paths = data["paths"]
    paths = _to_numpy(raw_paths)
    if paths.ndim == 2 and paths.shape[-1] == 2:
        paths = paths[None, ...]

    img_size = data.get("img_size", None)
    if img_size is not None:
        img_size = int(_to_numpy(img_size).reshape(-1)[0])

    intent_types = list(data.get("intent_types", []))
    start = _row_col(data.get("start_position"))
    goal = _row_col(data.get("goal_position"))
    map_id = data.get("map_id", pt_path.stem)

    h, w = height_map.shape
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    z_min = float(np.nanmin(height_map))
    z_max = float(np.nanmax(height_map))
    z_range = max(z_max - z_min, 1e-8)

    fig = plt.figure(figsize=(14, 10))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # 중요:
    # hidden/visible을 우리가 직접 판단해서 그리므로 Matplotlib 자동 zorder는 끈다.
    ax.computed_zorder = False

    # Surface colors
    z_norm = (height_map - z_min) / z_range
    face_colors = cm.terrain(z_norm)
    face_colors[..., 3] = float(surface_alpha)

    surf = ax.plot_surface(
        xx,
        yy,
        height_map,
        facecolors=face_colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=surface_alpha,
    )
    surf.set_zorder(0)

    # Axis limits and camera must be fixed before projection/depth-buffer computation.
    marker_z_offset = max(path_z_offset * 4.0, 0.04 * z_range)
    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)
    ax.set_zlim(z_min - 0.02 * z_range, z_max + marker_z_offset * 1.8)
    ax.view_init(elev=elev, azim=azim)

    z_aspect = max(z_range * float(z_exaggeration), 1.0)
    ax.set_box_aspect((w, h, z_aspect))

    # Force projection matrix to update.
    fig.canvas.draw()

    terrain_depth_buffer, terrain_bounds = _build_terrain_depth_buffer(
        ax=ax,
        height_map=height_map,
        buffer_res=depth_buffer_res,
        paint_radius=depth_buffer_radius,
    )

    # Paths
    colors = [
        "#FF0000", "#FF3333", "#CC0000", "#FF5555", "#DD1111",
        "#AA0000", "#FF2222", "#EE4444", "#BB0000", "#FF6666",
    ]
    filter_set = set(filter_intents) if filter_intents is not None else None
    plotted_intents: list[str] = []
    legend_proxies: list[Line2D] = []
    path_idx = 0

    if hidden_linewidth is None:
        hidden_linewidth = max(path_linewidth * 0.85, 1.0)

    for i, path in enumerate(paths):
        itype = intent_types[i] if i < len(intent_types) else f"path_{i}"
        if filter_set is not None and itype not in filter_set:
            continue

        path_px = _path_to_pixels(path, width=w, height=h, img_size=img_size)
        if len(path_px) < 2:
            continue

        dense_xyz = _densify_path(
            path_px=path_px,
            height_map=height_map,
            z_offset=path_z_offset,
            step_px=densify_step_px,
        )
        if len(dense_xyz) < 2:
            continue

        hidden_mask = _compute_hidden_mask(
            ax=ax,
            dense_xyz=dense_xyz,
            terrain_depth_buffer=terrain_depth_buffer,
            terrain_bounds=terrain_bounds,
            depth_tol=depth_tol,
        )
        hidden_segs, visible_segs = _split_segments_by_visibility(dense_xyz, hidden_mask)

        color = colors[path_idx % len(colors)]
        hidden_style = "--" if hidden_dashed else "solid"

        # 1) 지형 뒤쪽 path: 완전히 가리지 말고 흐릿하게 overlay한다.
        _add_segments(
            ax,
            hidden_segs,
            color=color,
            linewidth=hidden_linewidth,
            alpha=hidden_alpha,
            zorder=800,
            linestyle=hidden_style,
        )

        # 2) 지형 앞쪽 path: outline + foreground로 선명하게 표시한다.
        _add_segments(
            ax,
            visible_segs,
            color="black",
            linewidth=visible_outline_width,
            alpha=0.88,
            zorder=1000,
            linestyle="solid",
        )
        _add_segments(
            ax,
            visible_segs,
            color=color,
            linewidth=path_linewidth,
            alpha=1.0,
            zorder=1001,
            linestyle="solid",
        )

        if itype not in plotted_intents:
            plotted_intents.append(itype)
            legend_proxies.append(Line2D([0], [0], color=color, linewidth=path_linewidth, label=itype))

        path_idx += 1

    # Start / Goal markers
    marker_proxies: list[Line2D] = []

    def _marker(pos: tuple[float, float], marker: str, fill_color: str, ring_color: str, label: str) -> None:
        row_f, col_f = pos
        ri = int(np.clip(round(row_f), 0, h - 1))
        ci = int(np.clip(round(col_f), 0, w - 1))
        z_pt = float(height_map[ri, ci]) + marker_z_offset

        ax.scatter(
            [ci], [ri], [z_pt],
            c="white",
            s=marker_size * 1.35,
            marker=marker,
            edgecolors="black",
            linewidths=3.0,
            alpha=1.0,
            depthshade=False,
            zorder=1200,
            clip_on=False,
        )
        ax.scatter(
            [ci], [ri], [z_pt],
            c=fill_color,
            s=marker_size,
            marker=marker,
            edgecolors=ring_color,
            linewidths=2.0,
            alpha=1.0,
            depthshade=False,
            zorder=1201,
            clip_on=False,
        )
        marker_proxies.append(
            Line2D(
                [0], [0],
                marker=marker,
                color="w",
                markerfacecolor=fill_color,
                markeredgecolor=ring_color,
                markersize=10,
                linewidth=0,
                label=label,
            )
        )

    if start is not None:
        _marker(start, marker="o", fill_color="#00DD00", ring_color="#005500", label="Start")
    if goal is not None:
        _marker(goal, marker="*", fill_color="#FF6600", ring_color="#8B2500", label="Goal")

    # Labels / legend
    ax.set_xlabel("X (pixels)", labelpad=8)
    ax.set_ylabel("Y (pixels)", labelpad=8)
    ax.set_zlabel("Height (m)", labelpad=8)
    ax.set_title(f"3D Height Map + Depth-faded Path  |  {map_id}", fontsize=13, pad=12)

    handles = legend_proxies + marker_proxies
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=8,
            framealpha=0.90,
            bbox_to_anchor=(0.0, 1.0),
        )

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="3D height-map visualizer with depth-faded path overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--pt", required=True, help=".pt 파일 경로")
    ap.add_argument("--save", default=None, help="저장 경로. 미지정 시 plt.show()")
    ap.add_argument("--intents", nargs="*", default=None, help="표시할 intent 목록. 미지정 시 전체 표시")
    ap.add_argument("--elev", type=float, default=30.0, help="3D view elevation")
    ap.add_argument("--azim", type=float, default=225.0, help="3D view azimuth")
    ap.add_argument("--offset", type=float, default=0.05, help="path를 surface 위로 살짝 띄우는 높이 m")
    ap.add_argument("--surface-alpha", type=float, default=0.88, help="terrain surface 투명도")
    ap.add_argument("--hidden-alpha", type=float, default=0.22, help="지형 뒤쪽 path의 투명도")
    ap.add_argument("--path-linewidth", type=float, default=3.0, help="visible path linewidth")
    ap.add_argument("--visible-outline-width", type=float, default=6.0, help="visible path black outline width")
    ap.add_argument("--hidden-linewidth", type=float, default=None, help="hidden path linewidth. 기본값은 path-linewidth * 0.85")
    ap.add_argument("--hidden-dashed", action="store_true", help="지형 뒤쪽 path를 dashed line으로 표시")
    ap.add_argument("--densify-step-px", type=float, default=0.75, help="path occlusion 판정을 위한 segment sampling 간격 pixel")
    ap.add_argument("--depth-buffer-res", type=int, default=800, help="terrain depth buffer resolution")
    ap.add_argument("--depth-buffer-radius", type=int, default=3, help="terrain vertex를 screen-space buffer에 paint할 radius")
    ap.add_argument("--depth-tol", type=float, default=0.0015, help="occlusion 판정 tolerance")
    ap.add_argument("--marker-size", type=float, default=120.0, help="Start / Goal marker size")
    ap.add_argument("--z-exaggeration", type=float, default=10.0, help="z-axis visual exaggeration")
    ap.add_argument("--dpi", type=int, default=150, help="저장 이미지 DPI")

    args = ap.parse_args()

    pt = Path(args.pt)
    if not pt.exists():
        print(f"Error: 파일을 찾을 수 없습니다 → {pt}", file=sys.stderr)
        sys.exit(1)

    visualize_3d(
        pt_path=pt,
        save_path=args.save,
        filter_intents=args.intents,
        elev=args.elev,
        azim=args.azim,
        path_z_offset=args.offset,
        surface_alpha=args.surface_alpha,
        hidden_alpha=args.hidden_alpha,
        path_linewidth=args.path_linewidth,
        visible_outline_width=args.visible_outline_width,
        hidden_linewidth=args.hidden_linewidth,
        hidden_dashed=args.hidden_dashed,
        densify_step_px=args.densify_step_px,
        depth_buffer_res=args.depth_buffer_res,
        depth_buffer_radius=args.depth_buffer_radius,
        depth_tol=args.depth_tol,
        marker_size=args.marker_size,
        z_exaggeration=args.z_exaggeration,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
