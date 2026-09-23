"""Tasks — what to do, and what it costs.

A task owns its scene path, its initial conditions and its cost function, and
owns no simulator or renderer. Concrete tasks are imported from their own
modules; only the base classes are re-exported here, so importing this
subpackage pulls in neither MuJoCo nor Warp.
"""

from ContactModelStudy.Tasks.TaskBase import TaskBase, TaskRole

__all__ = ["TaskBase", "TaskRole"]
