"""Prepare isolated Duck/Ball BO searches from the frozen Cube workflow; run no episodes."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np


REPO = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
OBJECTS = ('duck', 'ball')
TEMPERATURES = (0.1, 0.05, 0.075, 0.15, 0.2)
FRICTION = dict(object=0.3, fingertip=0.5, palm=0.4, finger=0.4)
TIPS = {'if_tip', 'mf_tip', 'rf_tip', 'th_tip'}
UNCHANGED_FIELDS = (
    'actuator_gainprm', 'actuator_biasprm', 'actuator_dynprm', 'actuator_dyntype',
    'dof_damping', 'dof_armature', 'dof_frictionloss', 'jnt_range', 'jnt_pos', 'jnt_axis',
    'body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'geom_bodyid', 'geom_type',
    'geom_size', 'geom_pos', 'geom_quat', 'geom_condim', 'geom_priority',
    'geom_contype', 'geom_conaffinity', 'geom_solref', 'geom_solimp',
    'mesh_vert', 'mesh_face',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_scene(source, target, mass):
    original = mujoco.MjModel.from_xml_path(str(source))
    assert (original.nq, original.nv, original.nu, original.na) == (23, 22, 16, 0)
    spec = mujoco.MjSpec.from_file(str(source))
    spec.meshdir = str((source.parent / spec.meshdir).resolve())
    spec.texturedir = str((source.parent / spec.texturedir).resolve())
    obj = original.body('obj').id
    original_mass = float(original.body_mass[obj])
    if not np.isclose(mass, original_mass, rtol=0, atol=1e-12):
        body = spec.body('obj')
        body.explicitinertial = True
        body.mass = mass
        body.ipos = original.body_ipos[obj]
        body.iquat = original.body_iquat[obj]
        body.inertia = original.body_inertia[obj] * (mass / original_mass)
    groups = {name: [] for name in FRICTION}
    for body in spec.bodies:
        for geom in body.geoms:
            if not (geom.contype or geom.conaffinity):
                continue
            if body.name == 'obj':
                group = 'object'
            elif body.name == 'palm':
                group = 'palm'
            elif body.name.startswith(('if_', 'mf_', 'rf_', 'th_')):
                group = 'fingertip' if geom.name in TIPS else 'finger'
            else:
                continue
            geom.friction = [FRICTION[group], 0.005, 0.0001]
            groups[group].append(geom.name)
    assert all(groups.values()), groups
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(spec.to_xml())
    compiled = mujoco.MjModel.from_xml_path(str(target))
    assert (compiled.nq, compiled.nv, compiled.nu, compiled.na) == (23, 22, 16, 0)
    for name in UNCHANGED_FIELDS:
        np.testing.assert_allclose(getattr(compiled, name), getattr(original, name),
                                   rtol=2e-5, atol=1e-7, err_msg=name)
    np.testing.assert_allclose(compiled.body_mass[obj], mass, rtol=1e-6)
    np.testing.assert_allclose(compiled.body_inertia[obj],
                               original.body_inertia[obj] * mass / original_mass, rtol=2e-5)
    hand = np.arange(original.nbody) != obj
    for name in ('body_mass', 'body_inertia'):
        np.testing.assert_allclose(getattr(compiled, name)[hand], getattr(original, name)[hand], rtol=2e-5)
    changed = set()
    for group, names in groups.items():
        for name in names:
            gid = compiled.geom(name).id
            changed.add(gid)
            np.testing.assert_allclose(compiled.geom_friction[gid], [FRICTION[group], .005, .0001])
    untouched = [i for i in range(original.ngeom) if i not in changed]
    np.testing.assert_allclose(compiled.geom_friction[untouched], original.geom_friction[untouched])
    return dict(path=str(target), sha256=digest(target), source=str(source),
                original_mass_kg=original_mass, mass_kg=mass,
                inertia=compiled.body_inertia[obj].tolist(), groups=groups,
                uncalibrated=True, servo_activations=0,
                hand_dynamics_and_geometry_unchanged=True,
                object_geometry_and_COM_unchanged=True)


def prepare(args):
    source = args.cube_experiment.resolve()
    output = args.out.resolve()
    if output.exists():
        raise SystemExit(f'Refusing to overwrite existing preparation: {output}')
    base = json.loads((source / 'execution_revisions/mps750_20261115/config.json').read_text())
    original = json.loads((source / 'manifest.json').read_text())
    frozen = source / 'code'
    for rel, expected in original['source_snapshot_sha256'].items():
        assert digest(frozen / rel) == expected, rel
    masses = dict(duck=args.duck_mass, ball=args.ball_mass)
    if any(not np.isfinite(m) or m <= 0 for m in masses.values()):
        raise SystemExit('Object masses must be finite positive kilograms')
    output.mkdir(parents=True)
    code = output / 'code'
    shutil.copytree(frozen, code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__), output / 'prepare.py')
    shutil.copy2(REPO / 'experiments/run_prepared_object_bo.py', output / 'run_search.py')
    records, rows, targets = {}, [], {}
    for obj in OBJECTS:
        scene_root = output / 'scenes' / obj
        records[obj] = {}
        for role, filename in [('rollout', f'env_leap_rollout_{obj}_high_high.xml'),
                               ('eval', f'env_leap_eval_{obj}.xml')]:
            records[obj][role] = make_scene(code / 'scenes/leap' / filename,
                                           scene_root / 'leap' / filename, masses[obj])
        target = REPO / 'ContactModelStudy/Tasks' / f'{obj.capitalize()}Reorient.py'
        targets[obj] = dict(path=str(target), before_sha256=digest(target))
        shutil.copy2(target, output / target.name.replace('.py', '.before.py'))
        for temp in TEMPERATURES:
            row = dict(base['row'])
            row.update(label=f'{obj}_mass{masses[obj]*1000:g}g_T{temp:g}',
                       task=f'{obj}_reorient', scenes_dir=str(scene_root), temperature=temp)
            rows.append(row)
    # Temperature 0.1 for each object first; every other cell is separately selectable.
    rows.sort(key=lambda row: TEMPERATURES.index(row['temperature']))
    csv_path = output / 'searches.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    status='prepared_not_started', cube_reference=str(source),
                    source_git_commit=original['source_git_commit'], python=base['python'],
                    root=str(REPO), out=str(output), code=str(code), csv=str(csv_path),
                    results=str(output / 'search_results'), searches=rows,
                    source_snapshot_sha256={str(p.relative_to(code)): digest(p)
                        for p in code.rglob('*') if p.is_file()},
                    scenes=records, targets=targets, masses_kg=masses,
                    friction_sliding=FRICTION, friction_torsional=.005, friction_rolling=.0001,
                    temperatures=list(TEMPERATURES), noise_sigma=.05, n_samples=256,
                    n_calls=750, n_initial_points=120, models=['M1', 'M2', 'M3', 'M4'],
                    objective=original['objective'], physical_gpu_indices=base['gpu_indices'],
                    gpus=base['gpus'], preferred_planners_per_gpu=20,
                    fallback_planners_per_gpu=16, workers=48, trials_in_flight=3,
                    mps_bin=base['mps_bin'],
                    execution_validation='Not run: Duck/Ball concurrency requires runtime validation.',
                    configured_masses_source='Existing scene values unless overridden in prepare arguments.',
                    no_hardware=True, episodes_run=0, trials_started=0,
                    completion_action='Export the selected cell winner and a commented weight block; do not activate weights.',
                    requested_horizon_s=.352, resolved_control_intervals=5, resolved_horizon_s=.32)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'status.json', dict(status='prepared_not_started', trials_started=0, episodes_run=0))
    lines = ['Prepared Duck/Ball friction BO. No search has been started.', '',
             'Frozen Cube code; original uncalibrated high/high hands; M1-M4, worst-model objective.',
             'Object-specific initial poses, targets, COM handling, thresholds and contact-model settings retained.',
             f'Mass (kg): {masses}. Sliding friction object/tips/palm/fingers: .3/.5/.4/.4.',
             'Each cell: 750 trials; 120 initial points (1 object-default seed + 119 random);',
             '4 fixed episodes per model; 16 episodes per trial; 1000 steps maximum.',
             '256 samples; sigma=.05; 1 iteration; warm-start=none; dt=.004; control dt=.064.',
             'Horizon matches Cube exactly: requested .352, resolves to 5 actions / .32 seconds.',
             'Separate shared M1-M4 weight set at every fixed temperature.',
             'The two Cube masses are NOT assigned to these objects; masses above come from their scenes.',
             'No GPU allocations or MPS services occur during preparation or --check.',
             'The 20-planner Cube memory test is not a Duck/Ball memory benchmark.', '',
             'CPU validation only:',
             f"CUDA_VISIBLE_DEVICES='' {base['python']} {output/'run_search.py'} --check", '',
             'Future starts (only after requested; one selected cell per invocation):']
    for index, row in enumerate(rows):
        lines.append(f"cell {index}: {row['label']}")
        lines.append(f"  {base['python']} {output/'run_search.py'} --cell {index}")
    (output / 'README.txt').write_text('\n'.join(lines) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cube-experiment', type=Path,
                        default=REPO/'results/isolated_bo/cube_friction_20261115')
    parser.add_argument('--out', type=Path,
                        default=REPO/'results/isolated_bo/duck_ball_friction_20261116')
    parser.add_argument('--duck-mass', type=float, default=.05)
    parser.add_argument('--ball-mass', type=float, default=.14)
    args = parser.parse_args()
    manifest = prepare(args)
    print(json.dumps(dict(manifest=str(Path(manifest['out'])/'manifest.json'),
                          searches=len(manifest['searches']), status=manifest['status']), indent=2))


if __name__ == '__main__':
    main()
