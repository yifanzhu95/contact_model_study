"""Renderers — drawing a state, independent of which simulator produced it.

A renderer owns its own model, built from the task's MJCF, so simulators carry
no rendering code. Concrete renderers are imported directly from their modules
so that importing this subpackage does not require a GL context.
"""

from ContactModelStudy.Renderers.RendererBase import (
    RendererBase,
    RendererBaseConfig,
    VideoRendererBase,
    VideoRendererBaseConfig,
)

__all__ = [
    "RendererBase",
    "RendererBaseConfig",
    "VideoRendererBase",
    "VideoRendererBaseConfig",
]
