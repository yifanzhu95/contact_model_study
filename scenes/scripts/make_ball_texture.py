"""Generate scenes/leap/textures/ball_surface.png — the ball object's skin.

A sphere has no edges or corners, so an untextured one gives a reorientation
task nothing to read: the object can spin through any angle and the render never
changes. The scene used to solve that with a MuJoCo *builtin* (procedurally
generated) checker texture, which works in MuJoCo but is invisible to Pinocchio:
its MJCF parser only records a texture that exists as a FILE on disk
(GeometryObject.meshTexturePath), so the builtin left the Panda3d viewer drawing
a featureless white ball. Same lesson as the cube scene's atlas — anything
MuJoCo-only is lost on the Pinocchio path, so the appearance has to live in a
plain image file both renderers can load.

The pattern is a random patchwork: a grid of saturated random colours, each cell
carrying a random mark (dot / triangle / bar / ring) at a random offset. Dense,
locally unique features everywhere on the surface, so any rotation visibly moves
something regardless of which way the ball turns. The two renderers map it
differently — MuJoCo cube-maps it (the material is type="cube"), Panda3d wraps
it once around its UV sphere — and a random pattern reads correctly under both,
which a layout with a meaningful orientation (letters, a face atlas) would not.

Deterministic: SEED fixes the output, so re-running reproduces the committed PNG
byte-for-byte rather than silently changing the scene's appearance.

    python scenes/scripts/make_ball_texture.py
"""

from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

SEED  = 20260902
CELLS = 8      # grid is CELLS x CELLS patches
SIZE  = 512    # output edge in px (power of two: mipmaps cleanly)

OUT = Path(__file__).parents[1] / "leap" / "textures" / "ball_surface.png"


def _cell_colors(rng) -> np.ndarray:
    """One saturated RGB per cell, hues spread over the wheel then shuffled.

    Sampling hue uniformly at random clumps: neighbouring cells come out nearly
    the same colour often enough to leave washed-out patches. Taking an even
    sweep and shuffling it keeps the full spread while still looking random.
    """
    n = CELLS * CELLS
    hues = (np.arange(n) / n + rng.uniform(0, 1)) % 1.0
    rng.shuffle(hues)
    sat = rng.uniform(0.55, 0.95, n)
    val = rng.uniform(0.55, 1.00, n)
    rgb = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, sat, val)]
    return (np.asarray(rgb) * 255).astype(np.uint8)


def make_ball_texture() -> Image.Image:
    rng    = np.random.default_rng(SEED)
    colors = _cell_colors(rng)
    step   = SIZE / CELLS

    img  = Image.new("RGB", (SIZE, SIZE))
    draw = ImageDraw.Draw(img)

    for i in range(CELLS):
        for j in range(CELLS):
            x0, y0 = j * step, i * step
            x1, y1 = x0 + step, y0 + step
            base = tuple(int(c) for c in colors[i * CELLS + j])
            draw.rectangle([x0, y0, x1, y1], fill=base)

            # Mark colour: black or white, whichever contrasts with the patch, so
            # every mark stays visible whatever hue it landed on.
            lum  = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
            ink  = (20, 20, 20) if lum > 140 else (240, 240, 240)
            cx   = x0 + step * rng.uniform(0.32, 0.68)
            cy   = y0 + step * rng.uniform(0.32, 0.68)
            r    = step * rng.uniform(0.14, 0.24)
            kind = rng.integers(0, 4)
            if kind == 0:                                  # filled dot
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink)
            elif kind == 1:                                # triangle
                a = rng.uniform(0, 2 * np.pi)
                pts = [(cx + r * np.cos(a + k * 2 * np.pi / 3),
                        cy + r * np.sin(a + k * 2 * np.pi / 3)) for k in range(3)]
                draw.polygon(pts, fill=ink)
            elif kind == 2:                                # bar
                a  = rng.uniform(0, np.pi)
                dx, dy = r * np.cos(a), r * np.sin(a)
                draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)],
                          fill=ink, width=max(2, int(step * 0.12)))
            else:                                          # ring
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             outline=ink, width=max(2, int(step * 0.09)))

    # Grid lines last, so they sit on top of any mark that ran to a cell edge.
    for k in range(CELLS + 1):
        p = k * step
        w = max(1, int(step * 0.05))
        draw.line([(p, 0), (p, SIZE)], fill=(25, 25, 25), width=w)
        draw.line([(0, p), (SIZE, p)], fill=(25, 25, 25), width=w)

    return img


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    make_ball_texture().save(OUT)
    print(f"wrote {OUT}")
