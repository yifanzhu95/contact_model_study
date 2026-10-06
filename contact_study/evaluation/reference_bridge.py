"""Load the engine-neutral Kamino reference protocol in the main environment.

The main study and Newton/Kamino intentionally use incompatible MuJoCo and Warp
versions.  Only the NumPy/standard-library protocol modules are imported here;
no Newton or Kamino physics module is loaded into the main process.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KAMINO_REFERENCE_ROOT = REPOSITORY_ROOT / "kamino_reference"
KAMINO_REFERENCE_SRC = KAMINO_REFERENCE_ROOT / "src"


def enable_reference_protocol() -> Path:
    """Make the shared, pure-Python protocol package importable.

    This is deliberately a source-path bridge rather than a package dependency:
    installing ``kamino_reference`` into the main environment would also install
    Newton's Warp/MuJoCo versions and break the M1--M4 stack.
    """

    package = KAMINO_REFERENCE_SRC / "kamino_feasibility" / "reference_planning.py"
    if not package.is_file():
        raise FileNotFoundError(
            "Kamino reference protocol is missing. Expected " f"{package}"
        )
    source = str(KAMINO_REFERENCE_SRC)
    if source not in sys.path:
        sys.path.insert(0, source)
    return KAMINO_REFERENCE_SRC


def reference_paths() -> dict[str, Path]:
    """Canonical paths used by validation and the comparison CLI."""

    return {
        "root": KAMINO_REFERENCE_ROOT,
        "contract": KAMINO_REFERENCE_ROOT / "configs" / "optimizer_v3_20_repeat.json",
        "inputs_index": KAMINO_REFERENCE_ROOT / "configs" / "optimizer_v3_inputs.json",
        "validator": KAMINO_REFERENCE_ROOT / "scripts" / "validate_delivery.py",
        "coordinator": (
            KAMINO_REFERENCE_ROOT / "scripts" / "reference_optimizer_v2_coordinator.py"
        ),
        "workers": (
            KAMINO_REFERENCE_ROOT / "scripts" / "run_reference_optimizer_workers.py"
        ),
        "default_python": KAMINO_REFERENCE_ROOT / ".venv" / "bin" / "python",
    }
