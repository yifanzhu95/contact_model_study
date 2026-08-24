"""Reading sweep cell files, old shape or new.

The four sweep workers (horizon / num_rollout / cntrl_freq / contact_timeconst)
used to write a bare JSON list of AggregatedResult dicts via
metrics.save_results. They now write metrics.cell_record's dict — the same shape
run_kl_divergence_cell.py already used — which keeps the per-episode records
(end_reason, time_out, the recorded trajectory) alongside the aggregate.

Every results/ directory on disk and inside the results_*.zip archives holds the
old shape, so both are accepted. Anything unrecognized (a meta.json swept up by a
`*.json` glob) yields [] and is skipped rather than merged as a bogus record.
"""

from __future__ import annotations

import json
from pathlib import Path


__all__ = ["load_aggregates", "load_cell"]


def load_cell(path: str | Path) -> dict | None:
    """The whole cell record when the file is in the new shape, else None."""
    with open(path) as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) and isinstance(obj.get("aggregate"), (dict, list)) else None


def load_aggregates(path: str | Path) -> list[dict]:
    """Aggregate rows from a sweep cell file, whichever shape it is in."""
    with open(path) as f:
        obj = json.load(f)

    if isinstance(obj, list):                       # old: save_results output
        return obj
    if isinstance(obj, dict):
        agg = obj.get("aggregate")
        if isinstance(agg, dict):
            return [agg]
        if isinstance(agg, list):
            return agg
        if "model_label" in obj:                    # a lone aggregate dict
            return [obj]
    return []
