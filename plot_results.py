#!/usr/bin/env python3
"""
Reads optimization_log.csv and renders a 4-panel progress chart using Pillow.

Usage:
    python3 plot_results.py                        # reads optimization_log.csv
    python3 plot_results.py my_log.csv             # reads a specific log file
    python3 plot_results.py --watch                # re-renders every 5s (live mode)

Output: optimization_progress.png  (same directory as the log file)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

# ── Canvas layout ─────────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 1400, 960
BG              = (18, 18, 24)          # near-black background
PANEL_BG        = (28, 28, 36)          # panel background
GRID_COLOR      = (50, 52, 65)          # subtle grid lines
AXIS_COLOR      = (100, 105, 130)       # axis / tick labels
TITLE_COLOR     = (220, 225, 240)       # panel titles
ACCENT          = (255, 255, 255)       # axis values

MARGIN_TOP      = 55   # space for main title
MARGIN_LEFT     = 10
MARGIN_RIGHT    = 10
MARGIN_BOTTOM   = 10
GAP             = 14   # gap between panels

# Per-panel inner padding (space for axis labels + ticks)
PAD_LEFT   = 72
PAD_BOTTOM = 42
PAD_TOP    = 38
PAD_RIGHT  = 20

# Colours for each data series
PALETTE = [
    (99,  202, 255),   # blue
    (120, 230, 130),   # green
    (255, 175,  80),   # orange
    (230,  80,  90),   # red
    (180, 130, 255),   # purple
    (255, 230,  80),   # yellow
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_log(path: Path) -> dict[str, list[float]]:
    """Return {column_name: [values...]} from the CSV."""
    cols: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


# ── Drawing primitives ────────────────────────────────────────────────────────

def _try_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "FreeSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_SM  = _try_font(11)
FONT_MD  = _try_font(13)
FONT_LG  = _try_font(15)
FONT_XL  = _try_font(20)


def text_size(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def draw_panel_bg(
    draw: ImageDraw.Draw,
    x0: int, y0: int, x1: int, y1: int,
) -> None:
    draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=PANEL_BG)


def draw_panel(
    draw: ImageDraw.Draw,
    x0: int, y0: int, x1: int, y1: int,
    title: str,
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    x_vals: list[float],
    y_label: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
    n_x_ticks: int = 5,
    n_y_ticks: int = 5,
    fill_area: bool = False,
) -> None:
    """Draw a single line-chart panel."""
    draw_panel_bg(draw, x0, y0, x1, y1)

    # ── Title ─────────────────────────────────────────────────────────────────
    tw, th = text_size(draw, title, FONT_LG)
    draw.text(
        (x0 + (x1 - x0 - tw) // 2, y0 + 8),
        title, fill=TITLE_COLOR, font=FONT_LG,
    )

    # ── Plot area ─────────────────────────────────────────────────────────────
    ax0 = x0 + PAD_LEFT
    ay0 = y0 + PAD_TOP
    ax1 = x1 - PAD_RIGHT
    ay1 = y1 - PAD_BOTTOM

    if ax1 <= ax0 or ay1 <= ay0:
        return

    # ── Data range ────────────────────────────────────────────────────────────
    all_vals = [v for _, ys, _ in series for v in ys]
    if not all_vals or not x_vals:
        return

    data_ymin = y_min if y_min is not None else min(all_vals)
    data_ymax = y_max if y_max is not None else max(all_vals)
    if data_ymax == data_ymin:
        data_ymax = data_ymin + 1.0

    data_xmin = min(x_vals)
    data_xmax = max(x_vals)
    if data_xmax == data_xmin:
        data_xmax = data_xmin + 1.0

    def to_px(xv: float, yv: float) -> tuple[int, int]:
        px = int(ax0 + (xv - data_xmin) / (data_xmax - data_xmin) * (ax1 - ax0))
        py = int(ay1 - (yv - data_ymin) / (data_ymax - data_ymin) * (ay1 - ay0))
        return px, py

    # ── Grid lines ────────────────────────────────────────────────────────────
    # Horizontal
    for i in range(n_y_ticks + 1):
        yv = data_ymin + i * (data_ymax - data_ymin) / n_y_ticks
        _, py = to_px(data_xmin, yv)
        draw.line([(ax0, py), (ax1, py)], fill=GRID_COLOR, width=1)
        label = f"{yv:.1f}"
        lw, lh = text_size(draw, label, FONT_SM)
        draw.text((ax0 - lw - 6, py - lh // 2), label, fill=AXIS_COLOR, font=FONT_SM)

    # Vertical
    for i in range(n_x_ticks + 1):
        xv = data_xmin + i * (data_xmax - data_xmin) / n_x_ticks
        px, _ = to_px(xv, data_ymin)
        draw.line([(px, ay0), (px, ay1)], fill=GRID_COLOR, width=1)
        label = str(int(round(xv)))
        lw, lh = text_size(draw, label, FONT_SM)
        draw.text((px - lw // 2, ay1 + 5), label, fill=AXIS_COLOR, font=FONT_SM)

    # X-axis label ("Generation")
    xlabel = "Generation"
    xlw, xlh = text_size(draw, xlabel, FONT_SM)
    draw.text(
        (ax0 + (ax1 - ax0 - xlw) // 2, ay1 + 22),
        xlabel, fill=AXIS_COLOR, font=FONT_SM,
    )

    # Y-axis label (rotated via separate image)
    if y_label:
        lw, lh = text_size(draw, y_label, FONT_SM)
        label_img = Image.new("RGBA", (lw + 4, lh + 4), (0, 0, 0, 0))
        ld = ImageDraw.Draw(label_img)
        ld.text((2, 2), y_label, fill=AXIS_COLOR, font=FONT_SM)
        rotated = label_img.rotate(90, expand=True)
        # paste onto main image
        ry = ay0 + (ay1 - ay0 - rotated.height) // 2
        draw._image.paste(rotated, (x0 + 6, ry), rotated)

    # ── Axes border ───────────────────────────────────────────────────────────
    draw.rectangle([ax0, ay0, ax1, ay1], outline=GRID_COLOR, width=1)

    # ── Data series ───────────────────────────────────────────────────────────
    for name, ys, color in series:
        if not ys:
            continue
        points = [to_px(x_vals[i], ys[i]) for i in range(min(len(x_vals), len(ys)))]

        # Filled area under the curve (optional)
        if fill_area and len(points) > 1:
            poly = [(ax0, ay1)] + points + [(points[-1][0], ay1)]
            fill_color = color + (30,)  # very transparent
            overlay = Image.new("RGBA", draw._image.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.polygon(poly, fill=fill_color)
            draw._image.alpha_composite(overlay)

        # Line
        if len(points) > 1:
            draw.line(points, fill=color, width=2)

        # Dots at data points (only if not too many)
        if len(points) <= 200:
            r = 3
            for px, py in points:
                draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_x = ax0 + 8
    legend_y = ay0 + 8
    for name, _, color in series:
        sw = 18
        sh = 3
        draw.rectangle(
            [legend_x, legend_y + 5, legend_x + sw, legend_y + 5 + sh],
            fill=color,
        )
        draw.text((legend_x + sw + 5, legend_y), name, fill=TITLE_COLOR, font=FONT_SM)
        _, lh = text_size(draw, name, FONT_SM)
        legend_y += max(lh + 5, 16)


# ── Main render ───────────────────────────────────────────────────────────────

def render(log_path: Path, out_path: Path) -> None:
    data = load_log(log_path)
    gens = data.get("generation", [])
    if not gens:
        print("No data in log yet.")
        return

    img = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # ── Main title ────────────────────────────────────────────────────────────
    title = f"Route Optimizer — GA Progress  (generations: {int(gens[-1])})"
    tw, th = text_size(draw, title, FONT_XL)
    draw.text(((WIDTH - tw) // 2, 14), title, fill=TITLE_COLOR, font=FONT_XL)

    # ── Panel grid layout ─────────────────────────────────────────────────────
    col0 = MARGIN_LEFT
    col1 = WIDTH // 2 + GAP // 2
    col_w = WIDTH // 2 - GAP // 2 - MARGIN_LEFT
    row0 = MARGIN_TOP
    row1 = HEIGHT // 2 + GAP // 2
    row_h = (HEIGHT - MARGIN_TOP - MARGIN_BOTTOM) // 2 - GAP // 2

    panels = [
        # (x0, y0, x1, y1)
        (col0, row0, col0 + col_w, row0 + row_h),
        (col1, row0, col1 + col_w, row0 + row_h),
        (col0, row1, col0 + col_w, row1 + row_h),
        (col1, row1, col1 + col_w, row1 + row_h),
    ]

    def get(col: str) -> list[float]:
        return data.get(col, [])

    # ── Panel 1: TOTFIT ───────────────────────────────────────────────────────
    draw_panel(
        draw, *panels[0],
        title="Fitness (TOTFIT = ω₁·F1 + ω₂·F2 + ω₃·F3)",
        series=[("TOTFIT", get("best_TOTFIT"), PALETTE[0])],
        x_vals=gens,
        y_label="Score",
    )

    # ── Panel 2: Coverage breakdown ───────────────────────────────────────────
    draw_panel(
        draw, *panels[1],
        title="Demand Coverage Breakdown (%)",
        series=[
            ("Direct d0%",   get("d0p"),   PALETTE[1]),
            ("1 transfer d1%", get("d1p"), PALETTE[0]),
            ("2 transfers d2%", get("d2p"), PALETTE[2]),
            ("Unsatisfied%",  get("dunp"),  PALETTE[3]),
        ],
        x_vals=gens,
        y_label="%",
        y_min=0.0,
    )

    # ── Panel 3: F1, F2, F3 components ───────────────────────────────────────
    draw_panel(
        draw, *panels[2],
        title="Fitness Components (F1, F2, F3)",
        series=[
            ("F1 (travel time)",     get("F1"), PALETTE[4]),
            ("F2 (demand coverage)", get("F2"), PALETTE[1]),
            ("F3 (unsatisfied pen.)", get("F3"), PALETTE[2]),
        ],
        x_vals=gens,
        y_label="Score",
        y_min=0.0,
    )

    # ── Panel 4: Average Travel Time ──────────────────────────────────────────
    draw_panel(
        draw, *panels[3],
        title="Average Travel Time (ATT, km-equivalent)",
        series=[("ATT", get("ATT"), PALETTE[2])],
        x_vals=gens,
        y_label="km",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    img.convert("RGB").save(out_path)
    print(f"Saved → {out_path}  (gens={int(gens[-1])}, "
          f"TOTFIT={data['best_TOTFIT'][-1]:.4f}, "
          f"d0={data['d0p'][-1]:.1f}%, un={data['dunp'][-1]:.1f}%)")


def main() -> None:
    args = sys.argv[1:]
    watch = "--watch" in args
    args = [a for a in args if a != "--watch"]

    log_path = Path(args[0]) if args else Path("optimization_log.csv")
    out_path = log_path.parent / "optimization_progress.png"

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    if watch:
        print(f"Watching {log_path} — updating every 5s. Ctrl+C to stop.")
        while True:
            try:
                render(log_path, out_path)
                time.sleep(5)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        render(log_path, out_path)


if __name__ == "__main__":
    main()
