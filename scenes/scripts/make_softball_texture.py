"""Generate scenes/leap/textures/softball_surface.png — a yellow softball skin.

The ball geom needs a skin that reads as a rotation (see the comment in
env_leap_eval_ball.xml), and a softball's seam does that well: one closed,
chiral, asymmetric curve, so no two orientations look alike.

This is a plain EQUIRECTANGULAR map: image X is longitude (0..2pi, wrapping left
to right), image Y is latitude (+z pole on the top row, -z on the bottom). It is
consumed through the UV-mapped sphere mesh that scenes/scripts/make_uv_sphere.py
writes, NOT by either renderer's built-in sphere projection — because neither of
those works here and the two disagree with each other:

  - MuJoCo's type="cube" stamps one square onto all 6 cube faces before
    projecting, so a single continuous seam comes out as six copies of itself,
    one per face. Its type="2d" on a primitive is a planar projection, which
    smears badly anywhere off-axis.
  - Panda3d's sphere primitive (panda3d_viewer/geometry.py::make_capsule) emits
    tcoord = (u/pi, v/2pi) with u the polar angle — equirectangular, but
    TRANSPOSED: latitude along image X, longitude along image Y. An earlier
    version of this script drew for that convention; carrying UVs on the mesh
    instead is what lets one image serve both renderers.

Putting the UVs on the mesh means this file gets to use the conventional layout
and both renderers agree on it. It also means a straight "belt" drawn across the
image really is a ring around the ball, which is what the very first version of
this script wrongly assumed.

The drawing itself works on the sphere, not in the image: instead of stroking 2D
polylines, every pixel is turned back into a direction and coloured by its
ANGULAR DISTANCE to the seam curve. The seam then has a constant width *on the
ball*, and the longitude wrap and the latitude pinch near the poles both fall
out for free rather than needing special cases.

Deterministic by construction: the seam is an analytic curve and the stitches
are placed by arc length, so there is no RNG to seed — re-running reproduces the
committed PNG byte-for-byte.

    python scenes/scripts/make_softball_texture.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import numpy as np

SIZE        = 512   # output edge in px (power of two: mipmaps cleanly)
SUPERSAMPLE = 4     # render at SIZE*4, box-average down: cheap, exact AA

# Seam curve lobe weights. A + B = 1 puts the curve exactly on the unit sphere
# (see _curve). A/B = 3 is the critical ratio at which the seam stops doubling
# back on itself in longitude; sitting just below it gives ~7.4 deg of backtrack,
# which is what reads as the two interlocking horseshoes of a real ball. The
# resulting max latitude is 63.9 deg, leaving 26 deg of clearance from the poles
# — where this mapping is singular and the 16-ring sphere degenerates to a fan.
A, B = 0.72, 0.28

STITCHES = 56    # ticks around the seam. A regulation ball has 88; fewer and
                 # bolder survives 512 px and mipmapping better.

# All widths are half-angles in RADIANS ON THE SPHERE, not pixels, because the
# mapping is anisotropic: one pixel spans 0.352 deg of latitude vertically but
# only 0.703*sin(theta) deg of arc horizontally, so a constant-pixel feature
# would visibly fatten and thin as the seam turned. In radians it stays put.
GROOVE_HALF    = 0.075   # recessed channel the thread sits in (4.3 deg)
SEAM_HALF      = 0.012   # the red seam line itself (0.7 deg)
STITCH_INNER   = 0.014   # tick's near end — just outside the seam line
STITCH_OUTER   = 0.068   # tick's far end (stays inside GROOVE_HALF)
STITCH_SKEW    = 0.035   # along-seam run of a tick, giving it its slant
STITCH_HALF    = 0.016   # thread half-thickness (0.9 deg)
STITCH_OUTLINE = 0.008   # dark halo outside the thread; holds the shape under
                         # minification, where bare red bleeds into yellow

# Every channel stays below 255 so the material's specular="0.3" shininess="0.4"
# highlight reads as a highlight instead of clipping to a flat white blob.
YELLOW  = (240, 230,  45)   # optic softball yellow, slightly green-leaning
GROOVE  = (214, 198,  40)   # channel: ~10% darker, low contrast
RED     = (198,  32,  38)   # seam line and thread
DARKRED = (132,  20,  26)   # thread outline

SEAM_SAMPLES = 6000   # see _max_cosine for how this number is chosen
TICK_SAMPLES = 24

OUT = Path(__file__).parents[1] / "leap" / "textures" / "softball_surface.png"

_TWO_ROOT_AB = 2.0 * np.sqrt(A * B)


def _curve(t: np.ndarray) -> np.ndarray:
    """The baseball/tennis-ball seam curve, exactly on the unit sphere.

    With x + iy = A e^{it} + B e^{-3it} we get x^2 + y^2 = A^2 + B^2 + 2AB cos4t
    = (A+B)^2 - 4AB sin^2(2t) = (A+B)^2 - z^2, so |P| = A + B = 1 identically.
    That is why no normalisation is needed anywhere downstream.
    """
    return np.stack([A * np.cos(t) + B * np.cos(3 * t),
                     A * np.sin(t) - B * np.sin(3 * t),
                     _TWO_ROOT_AB * np.sin(2 * t)], axis=-1)


def _tangent(t: np.ndarray) -> np.ndarray:
    """Unit tangent. Automatically tangent to the sphere too, since |P| constant
    implies P . P' = 0 — which is what makes the stitch frame below orthonormal."""
    d = np.stack([-A * np.sin(t) - 3 * B * np.sin(3 * t),
                   A * np.cos(t) - 3 * B * np.cos(3 * t),
                   2 * _TWO_ROOT_AB * np.cos(2 * t)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _pixel_directions(s: int) -> tuple[np.ndarray, int, int]:
    """Unit vector per pixel, restricted to the rows the seam can reach.

    Standard equirectangular: column -> longitude, row -> polar angle with row 0
    at the +z pole. make_uv_sphere.py writes the matching vt, flipping the
    latitude axis there (OBJ v=0 is the image BOTTOM, PIL row 0 is the TOP) so
    the two stay in agreement — that flip lives on exactly one side of the pair.

    Rows outside the seam's latitude band are base yellow by construction, so
    skipping them is free; it also doubles as a check that the curve clears the
    poles, since the returned band must sit strictly inside the image.
    """
    phi = 2 * np.pi * (np.arange(s) + 0.5) / s              # longitude -> X
    theta = np.pi * (np.arange(s) + 0.5) / s                # polar angle -> Y

    reach = np.arcsin(_TWO_ROOT_AB)                         # max latitude
    margin = GROOVE_HALF + STITCH_HALF + STITCH_OUTLINE
    lo, hi = np.pi / 2 - reach - margin, np.pi / 2 + reach + margin
    rows = np.nonzero((theta >= lo) & (theta <= hi))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    assert r0 > 0 and r1 < s, "seam reaches the poles; check A/B"

    st, ct = np.sin(theta[r0:r1]), np.cos(theta[r0:r1])
    dirs = np.empty((r1 - r0, s, 3), np.float32)
    dirs[..., 0] = st[:, None] * np.cos(phi)[None, :]
    dirs[..., 1] = st[:, None] * np.sin(phi)[None, :]
    dirs[..., 2] = ct[:, None]
    return dirs.reshape(-1, 3), r0, r1


def _max_cosine(dirs: np.ndarray, points: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Per-pixel cosine of the angular distance to the nearest sample point.

    Kept as a cosine rather than an angle on purpose: max-dot is monotone in
    -distance, so "within radius r" is just "cos > cos(r)". That dodges arccos's
    precision cliff near 1 in float32, and lets one field be thresholded at
    several radii for free (groove, line, outline, thread).

    The matmul is a single BLAS sgemm; the broadcast form (M,1,3)*(1,N,3) is far
    slower and allocates an (M,N,3) temporary. Chunking bounds that temporary to
    chunk*N*4 bytes.

    Sampling rule for callers: keep the spacing between consecutive points below
    ~0.2 * (the smallest radius this field is thresholded at). Coarser than that
    and the painted width visibly scallops between samples.
    """
    out = np.empty(len(dirs), np.float32)
    pts = np.ascontiguousarray(points.T, np.float32)
    for k in range(0, len(dirs), chunk):
        out[k:k + chunk] = (dirs[k:k + chunk] @ pts).max(axis=1)
    return out


def _stitch_points() -> np.ndarray:
    """Sample points for every tick, spaced along the seam by ARC LENGTH.

    |P'(t)| swings by a factor of 2/sqrt(3) over the curve, so spacing stitches
    uniformly in t clumps them noticeably at the horseshoe turns. Reparameterise
    by cumulative chord length instead (20k segments puts the chord-vs-arc error
    around 1e-7 relative, which is well past caring).

    Each stitch is two ticks mirrored across the seam and slanted the same way,
    so the pair opens into the chevron that reads as baseball stitching.
    """
    t = np.linspace(0, 2 * np.pi, 20001)
    p = _curve(t)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    t_k = np.interp((np.arange(STITCHES) + 0.5) * s[-1] / STITCHES, s, t)

    c = _curve(t_k)                  # (K,3) — also the outward normal, since |c|=1
    tang = _tangent(t_k)             # (K,3) — along the seam
    across = np.cross(tang, c)       # (K,3) — across it; c,tang,across orthonormal

    f = np.linspace(0.0, 1.0, TICK_SAMPLES)
    alpha = -STITCH_SKEW / 2 + f * STITCH_SKEW
    chunks = []
    for side in (1.0, -1.0):
        beta = side * (STITCH_INNER + f * (STITCH_OUTER - STITCH_INNER))
        q = (c[None] + alpha[:, None, None] * tang[None]
                     + beta[:, None, None] * across[None])
        chunks.append(q.reshape(-1, 3))
    q = np.concatenate(chunks)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)   # project back to sphere


def make_softball_texture() -> Image.Image:
    s = SIZE * SUPERSAMPLE
    dirs, r0, r1 = _pixel_directions(s)
    rows, cols = r1 - r0, s

    seam = _curve(np.linspace(0, 2 * np.pi, SEAM_SAMPLES, endpoint=False))
    assert np.allclose(np.linalg.norm(seam, axis=1), 1.0)

    seam_cos = _max_cosine(dirs, seam.astype(np.float32)).reshape(rows, cols)
    stitch_cos = _max_cosine(dirs, _stitch_points().astype(np.float32)).reshape(rows, cols)

    img = np.empty((s, s, 3), np.uint8)
    img[:] = YELLOW
    band = img[r0:r1, :]
    # Painted outside-in, so each layer sits on top of the one it belongs to:
    # channel, then the line in it, then the thread crossing over both.
    band[seam_cos   > np.cos(GROOVE_HALF)]                  = GROOVE
    band[seam_cos   > np.cos(SEAM_HALF)]                    = RED
    band[stitch_cos > np.cos(STITCH_HALF + STITCH_OUTLINE)] = DARKRED
    band[stitch_cos > np.cos(STITCH_HALF)]                  = RED

    # P(t+pi) = (-x,-y,z) is a pi rotation about +z, i.e. a half-width shift in
    # longitude at unchanged latitude, so the image must repeat exactly twice
    # across its width. Cheapest possible proof that the mapping is right.
    assert np.array_equal(img, np.roll(img, s // 2, axis=1))

    # BOX, not LANCZOS: at an exact integer factor BOX is a true box average,
    # while LANCZOS's negative lobes halo saturated red against saturated yellow.
    return Image.fromarray(img).resize((SIZE, SIZE), Image.Resampling.BOX)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    make_softball_texture().save(OUT)
    print(f"wrote {OUT}")
