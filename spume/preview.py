# SPDX-License-Identifier: Apache-2.0
"""Point-cloud preview of a run: the liquid particles at a few moments, seen from the side (x, y) and from above (x, z), coloured by speed.

Only for a quick look at a new setup (inflow direction, walls) before a long bake; it needs Pillow.
"""

from pathlib import Path

import numpy as np

PANEL_WIDTH = 520
PANEL_HEIGHT = 380
MARGIN = 18
BACKGROUND = (16, 20, 26)
WALL_COLOUR = (150, 160, 170)
LABEL_COLOUR = (200, 208, 216)

# bit, and which edge of the side view it is drawn on: (axis, high end)
WALL_EDGES = {0: ("x", False), 1: ("x", True), 2: ("y", False), 3: ("y", True)}


def speed_colours(speeds: np.ndarray, top_speed: float) -> np.ndarray:
    """Slow liquid is deep blue, fast liquid is white."""
    fraction = np.clip(speeds / max(top_speed, 1e-6), 0.0, 1.0)[:, None]
    slow = np.array([40.0, 110.0, 200.0])
    fast = np.array([245.0, 250.0, 255.0])
    return (slow + (fast - slow) * fraction).astype(np.uint8)


def draw_view(image, origin, horizontal, vertical, positions, speeds, low, high, axes, walls, top_speed):
    """Draw one projection: `axes` are the two coordinates (0 x, 1 y, 2 z) mapped to the horizontal and vertical directions of the panel."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    span = np.array([high[axes[0]] - low[axes[0]], high[axes[1]] - low[axes[1]]])
    scale = min((horizontal - 2 * MARGIN) / span[0], (vertical - 2 * MARGIN) / span[1])

    def to_pixels(a, b):
        return origin[0] + MARGIN + (a - low[axes[0]]) * scale, origin[1] + vertical - MARGIN - (b - low[axes[1]]) * scale

    left, bottom = to_pixels(low[axes[0]], low[axes[1]])
    right, top = to_pixels(high[axes[0]], high[axes[1]])
    draw.rectangle([left, top, right, bottom], outline=(60, 68, 78))
    solid = [bit for bit in range(6) if (walls >> bit) & 1]
    for bit in solid:
        if axes == (0, 1) and bit < 4:
            axis, is_high = WALL_EDGES[bit]
            if axis == "x":
                x = right if is_high else left
                draw.line([x, top, x, bottom], fill=WALL_COLOUR, width=3)
            else:
                y = top if is_high else bottom
                draw.line([left, y, right, y], fill=WALL_COLOUR, width=3)
        if axes == (0, 2) and bit in (0, 1, 4, 5):
            if bit < 2:
                x = right if bit == 1 else left
                draw.line([x, top, x, bottom], fill=WALL_COLOUR, width=3)
            else:
                y = top if bit == 5 else bottom
                draw.line([left, y, right, y], fill=WALL_COLOUR, width=3)
    colours = speed_colours(speeds, top_speed)
    order = np.argsort(positions[:, 2 if axes[1] == 1 else 1])
    pixels = image.load()
    for index in order:
        a, b = to_pixels(positions[index, axes[0]], positions[index, axes[1]])
        x, y = int(a), int(b)
        if 0 <= x < image.width and 0 <= y < image.height:
            pixels[x, y] = tuple(int(value) for value in colours[index])


def save_preview(path: Path, snapshots, low, high, walls: int, fps: int, frames: int):
    """One row per moment, the side view on the left and the top view on the right."""
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (PANEL_WIDTH * 2, PANEL_HEIGHT * len(snapshots)), BACKGROUND)
    top_speed = max((float(speeds.max()) for _, speeds in snapshots if len(speeds)), default=1.0)
    moments = sorted({frames // 3, 2 * frames // 3, frames - 1})
    for row, (positions, speeds) in enumerate(snapshots):
        if len(positions):
            draw_view(image, (0, row * PANEL_HEIGHT), PANEL_WIDTH, PANEL_HEIGHT, positions, speeds, low, high, (0, 1), walls, top_speed)
            draw_view(image, (PANEL_WIDTH, row * PANEL_HEIGHT), PANEL_WIDTH, PANEL_HEIGHT, positions, speeds, low, high, (0, 2), walls, top_speed)
        label = f"t = {moments[row] / fps:.2f} s   {len(positions)} points   side (x, y) and top (x, z)"
        ImageDraw.Draw(image).text((MARGIN, row * PANEL_HEIGHT + 4), label, fill=LABEL_COLOUR)
    image.save(path)
