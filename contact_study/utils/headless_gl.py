"""Headless OpenGL environment for offscreen rendering on GPU / HPC nodes.

The Pinocchio eval sim records video through Panda3D's `p3headlessgl` pipe
(contact_models/pinocchio_sim.py::_setup_viewer), which needs an EGL context
bound to the NVIDIA *device*. Three things break that on a cluster node:

  * DISPLAY being set (e.g. under `xvfb-run`) makes Mesa's EGL pick its X11
    platform and fail against the virtual screen:
        libEGL warning: DRI3: Screen seems not DRI3 capable
        :display:gsg:glgsg(error): Unable to detect OpenGL version
    surfacing as `RuntimeError: Failed to capture image from viewer`;
  * a conda-installed Mesa libEGL shadowing the driver's libglvnd one;
  * MuJoCo grabbing the EGL device first (MUJOCO_GL=egl).

So the environment that actually renders is:

    unset DISPLAY
    export MUJOCO_GL=disable
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

LD_LIBRARY_PATH is captured by the dynamic loader at exec time, so editing
os.environ mid-process does nothing for the libEGL that Panda3D dlopens later.
configure_headless_gl() therefore re-execs the interpreter once with the
corrected environment rather than just setting variables; _SENTINEL keeps that
to at most one restart.

It is deliberately conservative: nothing happens unless this looks like an
NVIDIA glvnd Linux node, and any value already present in the environment is
left alone — so a workstation run, or an explicit `MUJOCO_GL=egl python -m ...`,
behaves exactly as it did before. CONTACT_STUDY_HEADLESS_GL=0 opts out entirely.

Not for the Drake eval sim: its VTK renderer wants the real X display that
`xvfb-run` provides, which is the opposite of what this sets up.
"""

from __future__ import annotations

import os
import sys

# glvnd vendor file for the NVIDIA EGL driver. Its presence doubles as the test
# for "this node has a driver EGL worth forcing".
NVIDIA_EGL_VENDOR = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
# Where that driver's libEGL.so.1 lives — ahead of any conda Mesa build.
SYSTEM_GL_LIBDIR = "/usr/lib64"

_SENTINEL = "CONTACT_STUDY_HEADLESS_GL_APPLIED"
_OPT_OUT = "CONTACT_STUDY_HEADLESS_GL"


def _target_env(mujoco_gl: str) -> dict[str, str | None]:
    """The env this process should be running with; None means "must be unset"."""
    target: dict[str, str | None] = {
        "DISPLAY": None,          # keep Mesa's EGL off the X11 platform
        "MUJOCO_GL": mujoco_gl,
        "__EGL_VENDOR_LIBRARY_FILENAMES":
            os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", NVIDIA_EGL_VENDOR),
    }
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    target["LD_LIBRARY_PATH"] = (
        ld if SYSTEM_GL_LIBDIR in ld.split(":")
        else ":".join(p for p in (SYSTEM_GL_LIBDIR, ld) if p)
    )
    return target


def _reexec(env: dict[str, str]) -> None:
    """Restart this interpreter on the same command line, under `env`."""
    # orig_argv (3.10+) is the real command line, interpreter flags and all, so
    # `python -u -m pkg.mod ...` comes back as itself. Rebuild it by hand only if
    # it is missing: `-m` has to be reconstructed from __main__'s spec, since
    # re-running the file by path would put drivers/ on sys.path instead of the
    # repo root and break the `contact_study.*` imports.
    argv = list(getattr(sys, "orig_argv", ()))
    if not argv:
        spec = getattr(sys.modules["__main__"], "__spec__", None)
        launcher = ["-m", spec.name] if spec is not None else [sys.argv[0]]
        argv = [sys.executable] + launcher + sys.argv[1:]
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, argv, env)


def configure_headless_gl(mujoco_gl: str = "disable", *, verbose: bool = True) -> bool:
    """Put this process in the environment Panda3D's EGL pipe needs.

    mujoco_gl: MUJOCO_GL for the render path in use — "egl" when MuJoCo itself
        is the renderer, "disable" for every other eval sim, so MuJoCo does not
        take the EGL device Panda3D wants.

    Returns True if the process is (now) set up for headless GL, False if this
    machine isn't one the fix applies to. Does not return when it re-execs.
    Call before warp/CUDA init: a re-exec throws away everything done so far.
    """
    if os.environ.get(_OPT_OUT, "").lower() in ("0", "off", "false", "no"):
        return False
    if not sys.platform.startswith("linux") or not os.path.exists(NVIDIA_EGL_VENDOR):
        return False
    if os.environ.get(_SENTINEL):   # already re-exec'd, or already correct
        return True

    target = _target_env(mujoco_gl)
    stale = {k: v for k, v in target.items() if os.environ.get(k) != v}
    os.environ[_SENTINEL] = "1"     # inherited by the child below
    if not stale:
        return True

    env = dict(os.environ)
    for key, val in target.items():
        if val is None:
            env.pop(key, None)
        else:
            env[key] = val
    if verbose:
        shown = ", ".join(f"{k}=unset" if v is None else f"{k}={v}"
                          for k, v in stale.items())
        print(f"[headless-gl] restarting with {shown}", flush=True)
    _reexec(env)
    return True                     # unreachable
