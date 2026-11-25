"""Append the completed search winner as comments, preserving active Python code."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


WEIGHT_NAMES = (
    'w_quat', 'w_pos_x', 'w_pos_y', 'w_pos_z', 'w_velo', 'w_contact',
    'w_joint', 'w_joint_velo', 'w_fallen', 'w_quat_term', 'w_pos_term',
    'w_fallen_term',
)
MARKER = '# BEGIN completed cube friction BO: mass070g_T0.1_noise0.05_750trials'


def make_block(summary, config):
    if summary['n_trials'] != config['n_calls'] or config['n_calls'] != 750:
        raise ValueError('Refusing to publish weights before all 750 trials complete')
    best = summary['best']
    if best['status'] != 'done' or not math.isfinite(best['objective']):
        raise ValueError('Best trial is not a valid completed trial')
    weights = {name: best['cost_weights'].get(name, config['row'].get(name))
               for name in WEIGHT_NAMES}
    if any(value is None or not math.isfinite(float(value)) for value in weights.values()):
        raise ValueError('Incomplete or non-finite winning weights')
    lines = [MARKER,
             '# Uncalibrated high/high; shared M1-M4 weights; M6 excluded.',
             '# Cube/tips/palm/other sliding friction: 0.3 / 0.5 / 0.4 / 0.4.',
             '# Mass=0.07 kg; temperature=0.1; noise_sigma=0.05; samples=256.',
             f"# Best trial={best['trial']}; objective={best['objective']!r} (minimized).",
             '# "cost_weights": {']
    lines.extend(f'#     "{name}": {float(value)!r},' for name, value in weights.items())
    lines.extend(['# },', '# END completed cube friction BO', ''])
    return '\n'.join(lines), weights


def append_winner(target, summary, config):
    block, weights = make_block(summary, config)
    target = Path(target)
    with target.open('a+b') as stream:
        stream.seek(0)
        before = stream.read()
        current = before.decode('utf-8')
        if MARKER in current:
            if block not in current:
                raise RuntimeError('A different completed winner block already exists')
            after = before
            appended = False
        else:
            after = before + ('\n\n' + block).encode('utf-8')
            before_ast = ast.dump(ast.parse(before), include_attributes=False)
            after_ast = ast.dump(ast.parse(after), include_attributes=False)
            if before_ast != after_ast:
                raise RuntimeError('Comment append would change active Python code')
            # Recheck the open file and pathname before the single append write.
            stream.seek(0)
            if stream.read() != before or os.fstat(stream.fileno()).st_ino != target.stat().st_ino:
                raise RuntimeError('Target changed during publication; results remain saved')
            stream.write(after[len(before):])
            stream.flush()
            os.fsync(stream.fileno())
            appended = True
    return dict(applied_at_utc=datetime.now(timezone.utc).isoformat(), target=str(target),
                before_sha256=hashlib.sha256(before).hexdigest(),
                after_sha256=hashlib.sha256(after).hexdigest(),
                active_code_ast_unchanged=True, appended=appended,
                trial=summary['best']['trial'], objective=summary['best']['objective'],
                weights=weights, trial_budget=config['n_calls'],
                baseline_500_preserved=config['baseline_500_results'])


def publish(config, revision, root, save):
    revision, root = Path(revision), Path(root)
    summary = json.loads((Path(config['results']) / config['cell'] / 'bo_summary.json').read_text())
    # Save a standalone winner even if the user is editing the target file at completion.
    block, weights = make_block(summary, config)
    save(revision / 'final_best_weights.json', dict(trial=summary['best']['trial'],
         objective=summary['best']['objective'], weights=weights, commented_block=block))
    manifest = json.loads(Path(config['source_manifest']).read_text())
    record = append_winner(manifest['target_file'], summary, config)
    save(revision / 'applied_comment_weights.json', record)
    save(root / 'applied_comment_weights.json', record)
