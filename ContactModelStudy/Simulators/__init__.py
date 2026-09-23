"""Simulator wrappers — one module per backend.

Each backend module is imported directly by the caller so that a heavy or
optional dependency (Warp, Drake, Pinocchio) is only imported when that
simulator is actually used.
"""

from ContactModelStudy.Simulators.Simulator import Simulator, SimulatorConfig
from ContactModelStudy.Simulators.VectorizedSimulator import (
    VectorizedSimulator,
    VectorizedSimulatorConfig,
)

__all__ = [
    "Simulator",
    "SimulatorConfig",
    "VectorizedSimulator",
    "VectorizedSimulatorConfig",
]
