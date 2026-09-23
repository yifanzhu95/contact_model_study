"""ContactModelStudy — contact model fidelity for sampling-based MPC.

Refactored package layout; see ``refactor_progress.md`` at the repository root
for what has been ported from ``contact_study`` so far. Subpackages are imported
explicitly (``from ContactModelStudy.Simulators.Simulator import Simulator``)
rather than re-exported here, so importing the package never pulls in MuJoCo,
Warp, Drake or Pinocchio.
"""

__version__ = "0.1.0"
