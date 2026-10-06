"""The committed handoff must be internally complete without CUDA."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_delivery.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "kamino_delivery_validator", _VALIDATOR_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load delivery validator: {_VALIDATOR_PATH}")
_VALIDATOR_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VALIDATOR_MODULE)
validate_delivery = _VALIDATOR_MODULE.validate_delivery


def test_compact_delivery_is_self_consistent() -> None:
    result = validate_delivery()
    assert result["ready"], result
    assert result["checks"]["contract_hash_matches_index"]
    assert result["checks"]["development_fingerprint_matches"]
    assert result["details"]["frozen_problem"]["samples"] == 128
