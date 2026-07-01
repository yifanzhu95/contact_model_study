"""combine_results.py

Merge the per-cell JSON files written by run_param_cell.py into the same two
artifacts experiments/run_param_search.py produces, plus a ranked summary:

  - <prefix>_rich.json : one row per cell (model, weights, success_rate, ...)
  - <prefix>_agg.json  : AggregatedResult list (same schema as run_experiment.py)

Run this after the SLURM job array finishes (the submit script chains it as an
afterok dependency):

    python combine_results.py --indir results/param_search_run \
        --output results/param_search_grasp_reorient

If --output is omitted it defaults to "<indir>/combined".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contact_study.evaluation.metrics import (
    EpisodeResult, aggregate_episodes, save_results,
)


def load_cells(indir: Path) -> list[dict]:
    """Load and index-sort every cell_*.json in indir."""
    files = sorted(indir.glob("cell_*.json"))
    if not files:
        raise SystemExit(f"no cell_*.json files found in {indir}")
    cells = []
    for f in files:
        with open(f) as fh:
            cells.append(json.load(fh))
    cells.sort(key=lambda c: c.get("combo_index", 0))
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir",  type=str, required=True,
                   help="Directory containing the per-cell cell_*.json files.")
    p.add_argument("--output", type=str, default=None,
                   help="Output prefix (suffixes _rich.json / _agg.json added). "
                        "Default: <indir>/combined")
    p.add_argument("--top_n",  type=int, default=15,
                   help="How many top cells to print in the ranked summary.")
    args = p.parse_args()

    indir  = Path(args.indir)
    cells  = load_cells(indir)
    prefix = args.output or str(indir / "combined")

    # ---- rich rows (drop the per-episode detail for the compact rich file) --
    rich_rows = []
    aggregated = []
    for c in cells:
        rich_rows.append({k: c[k] for k in (
            "model", "label", "overrides", "full_weights", "success_rate",
            "mean_steps_to_success", "mean_step_ms", "std_step_ms",
            "mean_elapsed_s", "n_episodes",
        ) if k in c})

        # Rebuild EpisodeResult objects so we can reuse the shared aggregator,
        # keeping the _agg.json schema identical to the other experiment scripts.
        episodes = [EpisodeResult(**e) for e in c.get("episodes", [])]
        if episodes:
            aggregated.append(
                aggregate_episodes(episodes, c["task"], c["label"], "B"))

    rich_path = f"{prefix}_rich.json"
    agg_path  = f"{prefix}_agg.json"

    Path(rich_path).parent.mkdir(parents=True, exist_ok=True)
    with open(rich_path, "w") as f:
        json.dump(rich_rows, f, indent=2)
    print(f"Saved {len(rich_rows)} rich rows -> {rich_path}")

    if aggregated:
        save_results(aggregated, agg_path)

    # ---- ranked summary: success rate desc, then step time asc --------------
    ranked = sorted(rich_rows, key=lambda r: (-r["success_rate"], r["mean_step_ms"]))
    col_w  = max((len(r["label"]) for r in ranked), default=10) + 2

    print(f"\n{'='*65}")
    print(f"  Combined {len(cells)} cells  (task={cells[0].get('task', '?')})")
    print(f"  Top-{args.top_n} by success rate")
    print(f"{'='*65}")
    print(f"  {'label':<{col_w}}  {'succ%':>6}  {'step_ms':>9}  {'mean_steps':>10}")
    print(f"  {'-'*col_w}  {'-'*6}  {'-'*9}  {'-'*10}")
    for row in ranked[:args.top_n]:
        ms = row.get("mean_steps_to_success")
        ms_str = f"{ms:.1f}" if ms is not None else "—"
        print(f"  {row['label']:<{col_w}}  "
              f"{row['success_rate']*100:>5.1f}%  "
              f"{row['mean_step_ms']:>8.3f}ms  "
              f"{ms_str:>10}")


if __name__ == "__main__":
    main()
