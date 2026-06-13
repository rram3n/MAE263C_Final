#!/usr/bin/env python3
"""
Halftoning robot toolpath generator
===================================

Turns a photo into a set of variable-diameter dots for a 2R + prismatic
("marker") robot.  X,Y is the plane of the paper, Z (the prismatic joint) is
out of plane and controls how big each dot comes out: deeper press -> bigger
dot.

Pipeline
--------
1. Lay a regular square grid of pitch ``dot_pitch`` over the drawing area.
2. For every grid cell, sample the image's local tone.
3. Map tone -> "coverage" in [0,1] using ``cutoff`` (white point) and
   ``gamma`` (response curve), respecting the chosen mode.
4. coverage -> dot diameter (linear, capped at the grid pitch so neighbouring
   dots can never overlap) -> Z depth.  By default Z is linear across the given
   Z range; with ``z_levels`` = N the coverage is quantized into N+1 bands and Z
   becomes the integer level 0..N (0 = lightest band, N = darkest band).
5. Emit per-pen CSV toolpaths (x, y, z) in snake order, plus a to-scale
   preview plot.

Modes
-----
* ``dark``  : black pen, dark areas of the image get the dots.
* ``light`` : black pen, light areas of the image get the dots (inverted).
* ``rgb``   : three passes with red / green / blue pens.  Each grid node is
              assigned exactly ONE pen color in a repeating pattern, so the
              three colors interleave and never overlap.  A dot's size is set by
              its channel's *color dominance* (how much that channel exceeds the
              mean of the other two), so a red area produces a big red dot and no
              green/blue, while white/gray produces no dots at all.

No-overlap guarantee
--------------------
All dots live on a single grid of pitch ``p``.  The nearest two dot centers
are exactly ``p`` apart, and the maximum diameter is capped at ``p`` (``max_dot``
in page units, clamped down to ``p``).  Two maxed-out dots therefore at most
*touch*; with ``max_dot`` < ``p`` you get a guaranteed white gap.  This holds
within a pen and across the three RGB pens.

Dependencies:  numpy, pillow, matplotlib
    pip install numpy pillow matplotlib

Example
-------
    python halftone_robot.py portrait.jpg \
        --x-range 0 200 --y-range 0 250 --z-range 0 4 \
        --mode dark --dot-pitch 3.0 --cutoff 0.9 --gamma 1.2 --save-plot preview.png

    python halftone_robot.py flowers.jpg \
        --x-range 0 200 --y-range 0 200 --z-range 0 3 \
        --mode rgb --dot-pitch 2.5
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    image_path: str
    x_range: tuple[float, float]     # (x_min, x_max) on the paper, page units
    y_range: tuple[float, float]     # (y_min, y_max) on the paper, page units
    z_range: tuple[float, float]     # (z_min, z_max) prismatic depth for a dot

    mode: str = "dark"               # "dark" | "light" | "rgb"

    # ---- the knobs you turn ----
    dot_pitch: float = 3.0           # grid spacing == max dot diameter (page units).
                                     # In adaptive mode this is the COARSE pitch
                                     # used in flat areas.
    cutoff: float = 0.9              # tone threshold / white point in [0,1]
    gamma: float = 1.0               # response curve; >1 thins midtones, <1 fattens
    max_dot: float = 0.0             # max dot diameter in page units; 0 = use the
                                     # full grid pitch. Always clamped to the pitch
                                     # so neighbouring dots never overlap.
    min_dot: float = 0.0             # min dot diameter in page units; drawn dots
                                     # are bumped up to at least this (0 = no floor).
    min_coverage: float = 0.04       # skip dots below this coverage (no tiny specks)
    invert: bool = False             # flip the tone->ink direction (mainly for rgb)

    # ---- adaptive density + dot budget (fewer dots, same quality) ----
    adaptive: bool = False           # quadtree: dense dots only where there's detail
    fine_pitch: float = 1.5          # finest cell size in detailed areas (page units)
    detail_threshold: float = 0.12   # subdivide a cell if its luminance range
                                     # (max-min, [0,1]) exceeds this. Lower = more dots.
    max_dots: int = 0                # hard cap on total dots; keep the biggest. 0 = off

    # ---- Z discretization ----
    # z_levels = 0 -> continuous Z (z linearly spans z_range). If z_levels = N
    # (>=1), coverage is quantized into discrete integer levels running from
    # z_level_min up to N (inclusive). The number of bands is N - z_level_min + 1,
    # coverage [0,1] is split into that many equal bands, and the dot's Z value
    # becomes the level for its band.
    #   z_level_min=0, N=4 -> levels 0..4: 0->0-20%, 1->20-40%, ... 4->80-100%.
    #   z_level_min=2, N=5 -> levels 2..5: 2->0-25%, 3->25-50%, ... 5->75-100%.
    # The dot diameter snaps to its band's mid-width so the preview matches.
    z_levels: int = 0
    z_level_min: int = 0             # lowest integer Z level when discretizing

    # Working resolution: pixels per grid cell used for sampling. Keeps the
    # per-cell tone consistent across source images of any resolution.
    supersample: int = 4

    # ---- output ----
    out_prefix: str = "halftone"
    save_plot: str | None = None     # path to save preview png, or None
    show_plot: bool = True
    travel_z: float | None = None    # optional pen-up Z written between dots


# Pen colors used for rgb mode (channel index 0,1,2 -> R,G,B).
RGB_PENS = [("red", (1.0, 0.0, 0.0)), ("green", (0.0, 0.7, 0.0)), ("blue", (0.0, 0.0, 1.0))]


# --------------------------------------------------------------------------- #
# Geometry: fit the image into the page preserving aspect ratio
# --------------------------------------------------------------------------- #
def compute_layout(cfg: Config, img_w: int, img_h: int):
    """Return grid geometry that fits the image into the page, centered."""
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    page_w = x_max - x_min
    page_h = y_max - y_min
    if page_w <= 0 or page_h <= 0:
        raise ValueError("x-range / y-range must be increasing and non-empty.")

    img_aspect = img_w / img_h
    page_aspect = page_w / page_h

    # Letterbox the image inside the page.
    if img_aspect > page_aspect:           # image is wider -> fit to width
        draw_w = page_w
        draw_h = page_w / img_aspect
    else:                                  # image is taller -> fit to height
        draw_h = page_h
        draw_w = page_h * img_aspect

    ncols = max(1, int(round(draw_w / cfg.dot_pitch)))
    nrows = max(1, int(round(draw_h / cfg.dot_pitch)))

    px = draw_w / ncols                    # actual cell pitch in x
    py = draw_h / nrows                    # actual cell pitch in y

    # Origin of the drawn region (centered in the page). Page y points up,
    # image row 0 is the top, so we start from the top edge and go down.
    ox = x_min + (page_w - draw_w) / 2.0
    oy_top = y_max - (page_h - draw_h) / 2.0

    return dict(ncols=ncols, nrows=nrows, px=px, py=py, ox=ox, oy_top=oy_top,
                draw_w=draw_w, draw_h=draw_h)


# --------------------------------------------------------------------------- #
# Image sampling
# --------------------------------------------------------------------------- #
def sample_image(cfg: Config, ncols: int, nrows: int):
    """Downsample the image to (nrows, ncols) cell averages. Returns R,G,B,gray
    arrays in [0,1], each shaped (nrows, ncols)."""
    img = Image.open(cfg.image_path).convert("RGB")

    # Downscale to a consistent working resolution (a few px per grid cell)
    # before filtering, then BOX-average down to one value per cell.
    ss = max(1, int(cfg.supersample))
    work_w, work_h = ncols * ss, nrows * ss
    if work_w * work_h < img.width * img.height:   # only ever downsample
        img = img.resize((work_w, work_h), Image.BOX)

    # BOX filter averages each cell's pixels -> exactly the per-cell tone.
    small = img.resize((ncols, nrows), Image.BOX)
    arr = np.asarray(small, dtype=np.float64) / 255.0      # (nrows, ncols, 3)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return R, G, B, gray


# --------------------------------------------------------------------------- #
# Tone -> coverage
# --------------------------------------------------------------------------- #
def _clamp01(a):
    return np.clip(a, 0.0, 1.0)


def coverage_from_tone(cfg: Config, R, G, B, gray):
    """Map sampled tone to ink coverage in [0,1] per grid node.

    Returns (coverage, color_index) where color_index is 0/1/2 for rgb mode
    (selecting which pen draws that node) and all-zeros for the black-pen modes.
    """
    nrows, ncols = gray.shape
    eps = 1e-9

    if cfg.mode == "dark":
        # Dark pixels -> ink. Anything brighter than cutoff stays blank.
        cov = _clamp01((cfg.cutoff - gray) / (cfg.cutoff + eps))
        color_index = np.zeros((nrows, ncols), dtype=int)

    elif cfg.mode == "light":
        # Light pixels -> ink (e.g. a white/black pen on dark paper).
        cov = _clamp01((gray - cfg.cutoff) / (1.0 - cfg.cutoff + eps))
        color_index = np.zeros((nrows, ncols), dtype=int)

    elif cfg.mode == "rgb":
        # Assign each grid node one pen color in a repeating diagonal pattern
        # so R/G/B interleave evenly across the page and the three passes never
        # overlap.
        jj, ii = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
        color_index = (ii + 2 * jj) % 3

        # Size each dot by how much *its* channel dominates the other two
        # (color dominance), not by raw brightness: a red pixel -> big red dot,
        # while white/gray (all channels equal) -> no dot of any color.
        chan = np.where(color_index == 0, R, np.where(color_index == 1, G, B))
        others = np.where(color_index == 0, (G + B) / 2.0,
                          np.where(color_index == 1, (R + B) / 2.0, (R + G) / 2.0))
        cov = _clamp01(chan - others)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r} (use dark|light|rgb)")

    # Response curve.
    if cfg.gamma != 1.0:
        cov = np.power(cov, cfg.gamma)

    return cov, color_index


# --------------------------------------------------------------------------- #
# Build the dot list (snake-ordered to minimize travel)
# --------------------------------------------------------------------------- #
@dataclass
class Dot:
    x: float
    y: float
    z: float
    diameter: float
    color_index: int


def coverage_to_z_and_diameter(cfg: Config, cov: float, cell_max: float):
    """Map a coverage value to (z, diameter) for one dot. Sizes are absolute
    page units.

    `cell_max` is the overlap-safe ceiling for this cell (the grid pitch, or the
    cell size in adaptive mode). The diameter is kept within [floor, cap], where
    cap = min(cfg.max_dot, cell_max) (cfg.max_dot = 0 -> fill the cell) and
    floor = min(cfg.min_dot, cap).

    Continuous (cfg.z_levels == 0): diameter scales with coverage between floor
    and cap; z spans z_range so the floor maps to z_min and the cap to z_max
    (depth and dot size stay consistent -- depth is what makes the dot).

    Discretized (cfg.z_levels == N >= 1): coverage is binned into integer levels
    from cfg.z_level_min up to N (inclusive). z is that integer level, and the
    diameter is the absolute size for that level, evenly spaced from floor
    (smallest level) to cap (largest level).
    """
    cap = min(cfg.max_dot, cell_max) if cfg.max_dot > 0 else cell_max
    floor = min(max(cfg.min_dot, 0.0), cap)       # 0 <= floor <= cap

    n = int(cfg.z_levels)
    if n >= 1:
        lo = min(max(0, int(cfg.z_level_min)), n)
        nbands = n - lo + 1
        band = int(cov * nbands)
        if band > nbands - 1:
            band = nbands - 1
        z = float(lo + band)
        frac = band / (nbands - 1) if nbands > 1 else 1.0   # 0..1 across the levels
        diameter = floor + frac * (cap - floor)
    else:
        z_min, z_max = cfg.z_range
        diameter = min(max(cov * cap, floor), cap)
        t = (diameter - floor) / (cap - floor) if cap > floor else 0.0  # keep z consistent
        z = z_min + t * (z_max - z_min)
    return z, diameter


def build_dots(cfg: Config, layout, coverage, color_index):
    ncols, nrows = layout["ncols"], layout["nrows"]
    px, py = layout["px"], layout["py"]
    ox, oy_top = layout["ox"], layout["oy_top"]

    p = min(px, py)

    dots: list[Dot] = []
    for j in range(nrows):
        col_iter = range(ncols) if j % 2 == 0 else range(ncols - 1, -1, -1)  # snake
        for i in col_iter:
            cov = float(coverage[j, i])
            if cov < cfg.min_coverage:
                continue
            x = ox + (i + 0.5) * px
            y = oy_top - (j + 0.5) * py
            z, diameter = coverage_to_z_and_diameter(cfg, cov, p)
            dots.append(Dot(x, y, z, diameter, int(color_index[j, i])))
    max_dot = min(cfg.max_dot, p) if cfg.max_dot > 0 else p   # effective cap, for display
    return dots, max_dot


# --------------------------------------------------------------------------- #
# Adaptive (quadtree) dot placement + dot budget
# --------------------------------------------------------------------------- #
def _scalar_coverage(cfg: Config, value: float) -> float:
    """Gray tone in [0,1] -> coverage in [0,1] for dark/light modes (scalar)."""
    eps = 1e-9
    if cfg.mode == "dark":
        c = (cfg.cutoff - value) / (cfg.cutoff + eps)
    else:                                           # light
        c = (value - cfg.cutoff) / (1.0 - cfg.cutoff + eps)
    c = min(max(c, 0.0), 1.0)
    return c ** cfg.gamma if cfg.gamma != 1.0 else c


def _scalar_rgb_dominance(cfg: Config, r: float, g: float, b: float, col: int) -> float:
    """Color dominance for one pen: how much channel `col` exceeds the mean of
    the other two, in [0,1]. Red pixel -> big red; white/gray -> ~0."""
    vals = (r, g, b)
    others = [vals[k] for k in range(3) if k != col]
    c = min(max(vals[col] - 0.5 * (others[0] + others[1]), 0.0), 1.0)
    return c ** cfg.gamma if cfg.gamma != 1.0 else c


def build_dots_adaptive(cfg: Config, layout):
    """Place dots via quadtree refinement: start at the coarse pitch and split
    a cell into 4 only where local contrast exceeds detail_threshold, down to
    fine_pitch. Each dot is inscribed in its (disjoint) cell, so no overlaps."""
    ox, oy_top = layout["ox"], layout["oy_top"]
    draw_w, draw_h = layout["draw_w"], layout["draw_h"]
    fine = max(cfg.fine_pitch, 1e-6)

    # Working image at a resolution matched to the finest cell (a few px per
    # fine cell), capped, filtered. Drawn region == whole image (aspect kept).
    ss = max(1, int(cfg.supersample))
    cap = 1000
    long_cells = max(draw_w, draw_h) / fine
    work_long = int(min(cap, max(8, long_cells * ss)))
    if draw_w >= draw_h:
        ww, hh = work_long, max(1, round(work_long * draw_h / draw_w))
    else:
        hh, ww = work_long, max(1, round(work_long * draw_w / draw_h))

    img = Image.open(cfg.image_path).convert("RGB").resize((ww, hh), Image.BOX)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    def _slice(u0, v0, u1, v1):
        x0, x1 = int(u0 * ww), max(int(u1 * ww), int(u0 * ww) + 1)
        y0, y1 = int(v0 * hh), max(int(v1 * hh), int(v0 * hh) + 1)
        return y0, y1, x0, x1

    dots: list[Dot] = []

    def emit(u0, v0, u1, v1):
        cw, ch = (u1 - u0) * draw_w, (v1 - v0) * draw_h
        if cfg.mode == "rgb":
            # 3 of the 4 quadrants carry R, G, B; each inscribed in its quadrant.
            for qx, qy, col in ((0, 0, 0), (1, 0, 1), (0, 1, 2)):
                a0 = u0 + qx * (u1 - u0) / 2; a1 = a0 + (u1 - u0) / 2
                b0 = v0 + qy * (v1 - v0) / 2; b1 = b0 + (v1 - v0) / 2
                ys, ye, xs, xe = _slice(a0, b0, a1, b1)
                region = arr[ys:ye, xs:xe, :]
                cov = _scalar_rgb_dominance(cfg, float(region[..., 0].mean()),
                                            float(region[..., 1].mean()),
                                            float(region[..., 2].mean()), col)
                if cov < cfg.min_coverage:
                    continue
                cell = min(cw / 2, ch / 2)
                z, d = coverage_to_z_and_diameter(cfg, cov, cell)
                dots.append(Dot(ox + (a0 + a1) / 2 * draw_w,
                                oy_top - (b0 + b1) / 2 * draw_h, z, d, col))
        else:
            ys, ye, xs, xe = _slice(u0, v0, u1, v1)
            cov = _scalar_coverage(cfg, float(gray[ys:ye, xs:xe].mean()))
            if cov < cfg.min_coverage:
                return
            cell = min(cw, ch)
            z, d = coverage_to_z_and_diameter(cfg, cov, cell)
            dots.append(Dot(ox + (u0 + u1) / 2 * draw_w,
                            oy_top - (v0 + v1) / 2 * draw_h, z, d, 0))

    def recurse(u0, v0, u1, v1, depth):
        cw, ch = (u1 - u0) * draw_w, (v1 - v0) * draw_h
        ys, ye, xs, xe = _slice(u0, v0, u1, v1)
        sub = gray[ys:ye, xs:xe]
        detail = float(sub.max() - sub.min())
        # Split only if there's contrast AND children would stay >= fine_pitch.
        if detail > cfg.detail_threshold and min(cw, ch) > 2 * fine and depth < 12:
            um, vm = (u0 + u1) / 2, (v0 + v1) / 2
            recurse(u0, v0, um, vm, depth + 1)
            recurse(um, v0, u1, vm, depth + 1)
            recurse(u0, vm, um, v1, depth + 1)
            recurse(um, vm, u1, v1, depth + 1)
        else:
            emit(u0, v0, u1, v1)

    nc = max(1, int(round(draw_w / cfg.dot_pitch)))
    nr = max(1, int(round(draw_h / cfg.dot_pitch)))
    for j in range(nr):
        for i in range(nc):
            recurse(i / nc, j / nr, (i + 1) / nc, (j + 1) / nr, 0)

    base = min(draw_w / nc, draw_h / nr)
    max_dot = min(cfg.max_dot, base) if cfg.max_dot > 0 else base   # effective cap, for display
    return dots, max_dot


def apply_budget(dots: list[Dot], max_dots: int) -> list[Dot]:
    """If over budget, keep the most visually important dots (largest diameter).
    Removing dots can never create an overlap."""
    if max_dots and len(dots) > max_dots:
        dots = sorted(dots, key=lambda d: d.diameter, reverse=True)[:max_dots]
    return dots


def snake_sort(dots: list[Dot], band: float) -> list[Dot]:
    """Order dots top-to-bottom in horizontal bands, alternating L->R / R->L,
    to minimise pen travel."""
    if not dots:
        return dots
    y_top = max(d.y for d in dots)
    band = max(band, 1e-6)
    rows: dict[int, list[Dot]] = {}
    for d in dots:
        rows.setdefault(int((y_top - d.y) / band), []).append(d)
    out: list[Dot] = []
    for b in sorted(rows):
        rows[b].sort(key=lambda d: d.x, reverse=(b % 2 == 1))
        out.extend(rows[b])
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv(path: str, dots: list[Dot], cfg: Config):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z"])
        for d in dots:
            if cfg.travel_z is not None:
                w.writerow([f"{d.x:.4f}", f"{d.y:.4f}", f"{cfg.travel_z:.4f}"])
            w.writerow([f"{d.x:.4f}", f"{d.y:.4f}", f"{d.z:.4f}"])
    return path


def write_outputs(cfg: Config, dots: list[Dot]):
    written = []
    if cfg.mode == "rgb":
        for idx, (name, _) in enumerate(RGB_PENS):
            pen_dots = [d for d in dots if d.color_index == idx]
            path = f"{cfg.out_prefix}_{name}.csv"
            write_csv(path, pen_dots, cfg)
            written.append((name, path, len(pen_dots)))
    else:
        path = f"{cfg.out_prefix}_black.csv"
        write_csv(path, dots, cfg)
        written.append(("black", path, len(dots)))
    return written


# --------------------------------------------------------------------------- #
# Preview plot
# --------------------------------------------------------------------------- #
def render_dots(ax, cfg: Config, dots: list[Dot], layout, max_dot):
    """Draw the dot pattern onto an existing matplotlib Axes (used by both the
    CLI preview and the GUI)."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Rectangle

    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range

    ax.clear()
    if cfg.mode == "light":
        ax.set_facecolor("0.15")           # dark paper preview
        fallback = "white"
    else:
        ax.set_facecolor("white")
        fallback = "black"

    # Group circles by pen color for efficient drawing.
    if cfg.mode == "rgb":
        groups = {idx: ([], col) for idx, (_, col) in enumerate(RGB_PENS)}
        for d in dots:
            groups[d.color_index][0].append(Circle((d.x, d.y), d.diameter / 2.0))
        for circles, col in groups.values():
            ax.add_collection(PatchCollection(circles, facecolors=[col], edgecolors="none"))
    else:
        circles = [Circle((d.x, d.y), d.diameter / 2.0) for d in dots]
        ax.add_collection(PatchCollection(circles, facecolors=fallback, edgecolors="none"))

    # Page outline + drawn region.
    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, ec="0.6", ls="--", lw=1))
    ax.add_patch(Rectangle((layout["ox"], layout["oy_top"] - layout["draw_h"]),
                           layout["draw_w"], layout["draw_h"],
                           fill=False, ec="0.8", ls=":", lw=0.8))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (page units)")
    ax.set_ylabel("Y (page units)")
    ax.set_title(f"{os.path.basename(cfg.image_path)}  |  mode={cfg.mode}  |  "
                 f"{len(dots)} dots  |  pitch={cfg.dot_pitch}  max_dot={max_dot:.2f}")


def preview(cfg: Config, dots: list[Dot], layout, max_dot):
    import matplotlib.pyplot as plt

    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    fig, ax = plt.subplots(figsize=(8, 8 * (y_max - y_min) / (x_max - x_min)))
    render_dots(ax, cfg, dots, layout, max_dot)

    if cfg.save_plot:
        fig.savefig(cfg.save_plot, dpi=150, bbox_inches="tight")
        print(f"  saved preview -> {cfg.save_plot}")
    if cfg.show_plot:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
def compute_dots(cfg: Config):
    """Run the full pipeline and return (dots, layout, max_dot) without any I/O."""
    with Image.open(cfg.image_path) as im:
        img_w, img_h = im.size

    layout = compute_layout(cfg, img_w, img_h)
    if cfg.adaptive:
        dots, max_dot = build_dots_adaptive(cfg, layout)
        band = cfg.fine_pitch
    else:
        R, G, B, gray = sample_image(cfg, layout["ncols"], layout["nrows"])
        coverage, color_index = coverage_from_tone(cfg, R, G, B, gray)
        dots, max_dot = build_dots(cfg, layout, coverage, color_index)
        band = min(layout["px"], layout["py"])

    dots = apply_budget(dots, cfg.max_dots)
    dots = snake_sort(dots, band)
    return dots, layout, max_dot


def generate(cfg: Config):
    dots, layout, max_dot = compute_dots(cfg)

    print(f"Grid: {layout['ncols']} x {layout['nrows']} cells "
          f"(pitch {layout['px']:.3f} x {layout['py']:.3f} page units)")
    pitch = min(layout["px"], layout["py"])
    if cfg.max_dot and cfg.max_dot > pitch:
        print(f"  note: max dot {cfg.max_dot:.3f} > pitch {pitch:.3f}; "
              f"clamped to pitch so dots don't overlap")
    print(f"Max dot diameter: {max_dot:.3f}  (z {cfg.z_range[0]}..{cfg.z_range[1]})")
    if dots:
        ds = [d.diameter for d in dots]
        print(f"Dots placed: {len(dots)}   diameter {min(ds):.3f}..{max(ds):.3f}")
    else:
        print("Dots placed: 0  (try lowering --cutoff or --min-coverage)")

    written = write_outputs(cfg, dots)
    for name, path, n in written:
        print(f"  {name:>6} pen: {n:5d} dots -> {path}")

    if cfg.show_plot or cfg.save_plot:
        preview(cfg, dots, layout, max_dot)
    return dots


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        description="Generate variable-diameter halftone dot toolpaths for a "
                    "2R + prismatic drawing robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("image", help="input photo (jpg/png/...)")
    p.add_argument("--x-range", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                   required=True, help="paper X range in page units")
    p.add_argument("--y-range", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                   required=True, help="paper Y range in page units")
    p.add_argument("--z-range", type=float, nargs=2, metavar=("ZMIN", "ZMAX"),
                   required=True, help="prismatic depth range: ZMIN=lightest dot, "
                                       "ZMAX=biggest dot")
    p.add_argument("--mode", choices=["dark", "light", "rgb"], default="dark")

    # knobs
    p.add_argument("--dot-pitch", type=float, default=3.0,
                   help="grid spacing AND max dot diameter (page units)")
    p.add_argument("--cutoff", type=float, default=0.9,
                   help="tone threshold / white point in [0,1]")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="response curve; >1 thins midtones, <1 fattens them")
    p.add_argument("--max-dot", type=float, default=0.0,
                   help="max dot diameter in page units; 0 = full grid pitch "
                        "(always clamped to the pitch so dots never overlap)")
    p.add_argument("--min-dot", type=float, default=0.0,
                   help="min dot diameter in page units (floor for drawn dots)")
    p.add_argument("--min-coverage", type=float, default=0.04,
                   help="skip dots below this coverage")
    p.add_argument("--invert", action="store_true",
                   help="flip tone->ink direction (subtractive feel in rgb mode)")

    # adaptive density + budget
    p.add_argument("--adaptive", action="store_true",
                   help="quadtree density: dense dots only where there's detail")
    p.add_argument("--fine-pitch", type=float, default=1.5,
                   help="finest cell size in detailed areas (adaptive mode)")
    p.add_argument("--detail-threshold", type=float, default=0.12,
                   help="subdivide when cell luminance range exceeds this; lower=more dots")
    p.add_argument("--max-dots", type=int, default=0,
                   help="hard cap on total dots (keeps the biggest); 0 = off")

    # Z discretization
    p.add_argument("--z-levels", type=int, default=0,
                   help="discretize Z into integer levels up to N (equal coverage "
                        "bands); Z output becomes the level. 0 = continuous Z")
    p.add_argument("--z-level-min", type=int, default=0,
                   help="lowest integer Z level when --z-levels is set")

    # output
    p.add_argument("--out-prefix", default="halftone", help="output CSV prefix")
    p.add_argument("--save-plot", default=None, help="save preview PNG to this path")
    p.add_argument("--no-show", action="store_true", help="do not open the preview window")
    p.add_argument("--travel-z", type=float, default=None,
                   help="optional pen-up Z written before each dot")

    a = p.parse_args(argv)
    return Config(
        image_path=a.image,
        x_range=tuple(a.x_range),
        y_range=tuple(a.y_range),
        z_range=tuple(a.z_range),
        mode=a.mode,
        dot_pitch=a.dot_pitch,
        cutoff=a.cutoff,
        gamma=a.gamma,
        max_dot=a.max_dot,
        min_dot=a.min_dot,
        min_coverage=a.min_coverage,
        invert=a.invert,
        adaptive=a.adaptive,
        fine_pitch=a.fine_pitch,
        detail_threshold=a.detail_threshold,
        max_dots=a.max_dots,
        z_levels=a.z_levels,
        z_level_min=a.z_level_min,
        out_prefix=a.out_prefix,
        save_plot=a.save_plot,
        show_plot=not a.no_show,
        travel_z=a.travel_z,
    )


if __name__ == "__main__":
    generate(parse_args())
