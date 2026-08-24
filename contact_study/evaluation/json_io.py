"""JSON writing for result records that carry bulk numeric arrays.

Every result writer in this repo used ``json.dump(..., indent=2)``, which is what
makes a result file readable: one key per line, diffable, greppable. Applied to a
per-control-step trajectory it is also what makes the file unusable — indent=2
puts every scalar of a (2000, 23) qpos block on its own line, turning a 2 MB
array into ~40 MB of whitespace.

`dump` keeps indent=2 for the record's structure and writes anything wrapped in
`compact()` on ONE line. The stdlib encoder has no per-subtree indent hook, so
this happens in two passes: each compact block is encoded separately with
separators=(",", ":"), a placeholder string takes its place in the indented
encode, and the blocks are spliced back in afterwards. The placeholder embeds a
uuid4 minted per dump() call, so it cannot collide with real string data.
(The token is deliberately ASCII: json.dumps escapes non-ASCII by default, which
would stop the placeholder from matching the text it has to be swapped into.)

Floats are rounded to `precision` SIGNIFICANT digits before encoding. The default
7 is exactly float32's precision, so nothing the GPU produced is lost (planner
means, covariances and deltas are all float32); Python's float repr is
shortest-round-trip, so 7 significant digits is also 7 mantissa characters on the
wire. precision=0 disables rounding entirely, which is what a bit-exact replay of
the float64 qpos/ctrl arrays needs.

Two hazards this module exists to absorb:
  - a bare np.float32 anywhere in the record is not JSON-serializable and would
    otherwise raise at write time, i.e. at the END of a two-hour sweep cell;
  - the stdlib writes non-finite floats as bare NaN/Infinity, which json.load
    accepts but jq and every JS consumer reject. sanitize_nonfinite maps them to
    null. (gaussian_kl already returns inf on a non-PD covariance.)
"""

from __future__ import annotations

from pathlib import Path
import json
import uuid

import numpy as np


__all__ = ["compact", "unwrap", "dump", "dumps"]


class _Compact:
    """Marker: encode this subtree on a single line, at its own precision.

    Transparent to in-memory consumers as well as to the encoder. The recorded
    trajectory blocks are built out of these and are read back in-process by
    replay code and tests, so a marker they cannot index would force every reader
    to know about this module. Indexing, len(), iteration and np.asarray() all
    pass straight through to the wrapped value; the wrapped array is NOT copied
    or converted here, so wrapping a (2000, 23) block stays free until dump time.
    """
    __slots__ = ("value", "precision")

    def __init__(self, value, precision):
        self.value     = value
        self.precision = precision

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self.value, dtype=dtype)
        return a.copy() if copy else a

    def __getitem__(self, k):  return self.value[k]
    def __len__(self):         return len(self.value)
    def __iter__(self):        return iter(self.value)

    def tolist(self):
        v = self.value
        return v.tolist() if isinstance(v, np.ndarray) else list(v)

    def __repr__(self):
        return f"compact({self.value!r})"


def compact(x, *, precision: int | None = None):
    """Mark `x` to be written on one line. `precision` overrides dump()'s."""
    return _Compact(x, precision)


def unwrap(x):
    """The value inside a compact() marker (or x itself when it is not one)."""
    return x.value if isinstance(x, _Compact) else x


# ---------------------------------------------------------------------------
# numeric preparation
# ---------------------------------------------------------------------------
def _sig_round(a: np.ndarray, sig: int) -> np.ndarray:
    """Round to `sig` significant digits. NaN/Inf/0 pass through untouched."""
    a   = np.asarray(a, dtype=np.float64)
    out = a.copy()
    nz  = np.isfinite(a) & (a != 0.0)
    if nz.any():
        v        = a[nz]
        mag      = np.floor(np.log10(np.abs(v)))
        with np.errstate(over="ignore", invalid="ignore"):
            f = 10.0 ** (sig - 1 - mag)
            r = np.round(v * f) / f
        # A subnormal input (|v| < ~1e-308) overflows the scale factor to inf and
        # would come back NaN — silently corrupting a real number into a null.
        # Keep the original value whenever the scaling did not survive.
        out[nz] = np.where(np.isfinite(r), r, v)
    return out


def _nullify(x):
    """Replace non-finite floats with None in a nested list (rare path)."""
    if isinstance(x, list):
        return [_nullify(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _prepare_array(a: np.ndarray, sig: int, sanitize: bool):
    if a.dtype.kind == "f":
        b   = _sig_round(a, sig) if sig else np.asarray(a, dtype=np.float64)
        out = b.tolist()
        if sanitize and not np.isfinite(b).all():
            out = _nullify(out)
        return out
    if a.dtype.kind in "iub":
        return a.tolist()
    return a.tolist()


def _try_array(x):
    """np.asarray(x) when x is a rectangular numeric/bool sequence, else None."""
    try:
        arr = np.asarray(x)
    except (ValueError, TypeError):
        return None
    return arr if arr.dtype.kind in "fiub" and arr.size else None


def _prepare(x, sig: int, sanitize: bool):
    """Recursively convert numpy types to JSON-native ones, rounding floats."""
    if isinstance(x, np.ndarray):
        return _prepare_array(x, sig, sanitize)
    if isinstance(x, dict):
        return {k: _prepare(v, sig, sanitize) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        arr = _try_array(x)
        if arr is not None:
            return _prepare_array(arr, sig, sanitize)
        return [_prepare(v, sig, sanitize) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if not np.isfinite(v):
            return None if sanitize else v
        return float(_sig_round(np.array(v), sig)) if sig else v
    return x


# ---------------------------------------------------------------------------
# the two-pass encoder
# ---------------------------------------------------------------------------
def _extract(x, blocks: list, token: str):
    """Replace every _Compact subtree with a placeholder, collecting the values."""
    if isinstance(x, _Compact):
        blocks.append(x)
        return f"{token}{len(blocks) - 1}@@"
    if isinstance(x, dict):
        return {k: _extract(v, blocks, token) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_extract(v, blocks, token) for v in x]
    return x


def dumps(obj, *, indent: int = 2, precision: int = 7,
          sanitize_nonfinite: bool = True) -> str:
    """Encode `obj`, writing compact()-marked subtrees on one line each."""
    token  = f"@@compact:{uuid.uuid4().hex}:"
    blocks: list[_Compact] = []
    outer  = _extract(obj, blocks, token)

    text = json.dumps(_prepare(outer, precision, sanitize_nonfinite), indent=indent)
    for i, blk in enumerate(blocks):
        sig     = precision if blk.precision is None else blk.precision
        encoded = json.dumps(
            _prepare(blk.value, sig, sanitize_nonfinite), separators=(",", ":")
        )
        # The placeholder is a JSON string in `text`; swap it quotes and all.
        text = text.replace(f'"{token}{i}@@"', encoded, 1)
    return text


def dump(obj, path, *, indent: int = 2, precision: int = 7,
         sanitize_nonfinite: bool = True) -> Path:
    """dumps() straight to `path`, creating the parent directory. Returns path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(dumps(obj, indent=indent, precision=precision,
                      sanitize_nonfinite=sanitize_nonfinite))
    return path
