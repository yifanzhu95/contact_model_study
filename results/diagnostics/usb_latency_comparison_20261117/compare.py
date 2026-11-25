"""Offline before/after USB latency comparison; never opens hardware."""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
PAIRS = {
    'empty_steps': ('target_response_20261116_empty_hand', 'target_response_20261117_empty_hand_usb1_retry'),
    'cube_steps': ('target_response_20261116_cube_003', 'target_response_20261117_cube_003_usb1'),
    'cube_sine': ('target_response_20261116_cube_sine008', 'target_response_20261117_cube_sine008_usb1'),
}


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), p95=float(np.percentile(a, 95)), max=float(a.max())) if len(a) else None


def load_run(name):
    path = DIAG / name
    meta = json.loads((path / 'experiment.json').read_text())
    if meta['status'] != 'complete' or not meta.get('torque_off_verified'):
        raise ValueError(f'{name}: not a completed, torque-off verified response test')
    rows = list(csv.DictReader((path / 'comparison.csv').open()))
    read, health, stamps, peak_current, errors = [], [], [], 0, []
    for line in (path / 'bridge_telemetry.jsonl').open():
        r = json.loads(line)
        if r.get('event') != 'cycle':
            continue
        if 'read_duration_s' in r:
            read.append(r['read_duration_s'] * 1000)
        if 'health_duration_s' in r:
            health.append(r['health_duration_s'] * 1000)
        if 'state_timestamp_us' in r:
            stamps.append(r['state_timestamp_us'])
        if r.get('current_raw_native'):
            peak_current = max(peak_current, max(map(abs, r['current_raw_native'])))
        if r.get('error'):
            errors.append(r['error'])
    return dict(path=str(path), usb=meta.get('usb_adapter'), model=meta['model'],
                hardware_config=meta['hardware_config'], amplitude_rad=meta['amplitude_rad'],
                cycles=meta['cycles'], waveform=meta.get('waveform', 'steps'),
                closing_only=meta.get('closing_only', False), rate_hz=meta['rate_hz'],
                joints=[{k: v if k == 'joint' else float(v) if v else None for k, v in r.items()} for r in rows],
                read_ms=stats(read), health_ms=stats(health),
                feedback_interval_ms=stats(np.diff(stamps) * .001), peak_current_raw=peak_current,
                errors=errors, torque_off_verified=True,
                telemetry=json.loads((path / 'bridge_telemetry.status.json').read_text()))


def main():
    report = dict(pairs={}, caveats=[
        'Historical 16 ms USB setting was verified during the original session, but was not embedded in each old recording.',
        'Before/after are separate runs, not randomized repeated trials; loaded grasp placement can differ.',
        'Outer command update rate and target sequences are matched. The unthrottled bridge read/write rate changes with USB latency.',
        'Phase lag includes motor dynamics and encoder/USB reporting; reduced measured lag is not proof of faster mechanics.',
        'Fingertip positions are encoder-based calibrated forward kinematics, not optical measurements.',
        'Current values include preparation and reset. Cube stability requires operator observation; no object tracking was recorded.',
    ])
    flat = []
    for label, names in PAIRS.items():
        if not (DIAG / names[1] / 'comparison.csv').exists():
            continue
        before, after = map(load_run, names)
        for key in ('model', 'hardware_config', 'amplitude_rad', 'cycles', 'waveform', 'closing_only', 'rate_hz'):
            if before[key] != after[key]:
                raise ValueError(f'{label}: unmatched {key}')
        report['pairs'][label] = dict(before=before, after=after)
        for b, a in zip(before['joints'], after['joints'], strict=True):
            assert b['joint'] == a['joint']
            for metric in b:
                if metric != 'joint':
                    flat.append(dict(protocol=label, joint=b['joint'], metric=metric,
                                     before_16ms=b[metric], after_1ms=a[metric]))
    (HERE / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    if not flat:
        print('No completed matched repeats yet.')
        return
    with (HERE / 'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [('fast_mean_publish_to_write_ms', 'Publish → write (ms)'),
               ('fast_phase_lag_ms', 'Held target → encoder phase lag (ms)'),
               ('fast_mean_joint_error_deg', 'Fast target error (deg)'),
               ('settled_mean_joint_error_deg', 'Settled target error (deg)')]
    text = ['# USB latency comparison', '', 'Unchanged motor settings and matched command sequences.', '']
    for label, pair in report['pairs'].items():
        b, a = pair['before'], pair['after']
        text += [f'## {label}', '', '| Metric | 16 ms | 1 ms |', '|---|---:|---:|']
        for key, title in [('read_ms', 'Serial read (ms)'), ('feedback_interval_ms', 'Feedback interval (ms)'), ('health_ms', 'Health register read (ms)')]:
            text.append(f"| {title}, mean | {b[key]['mean']:.3f} | {a[key]['mean']:.3f} |")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        x = np.arange(4)
        for ax, (key, title) in zip(axes.flat, metrics):
            old = [r[key] for r in b['joints']]; new = [r[key] for r in a['joints']]
            ax.bar(x-.18, old, .36, label='USB 16 ms')
            ax.bar(x+.18, new, .36, label='USB 1 ms')
            ax.set_xticks(x, ['Index', 'Middle', 'Ring', 'Thumb']); ax.set_ylabel(title)
            ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.2)
            text.append(f'| {title}, mean across four joints | {np.mean(old):.3f} | {np.mean(new):.3f} |')
        axes.flat[0].legend(); fig.suptitle(label.replace('_', ' ') + ': matched command-response tests')
        fig.savefig(HERE / f'{label}.png', dpi=180); fig.savefig(HERE / f'{label}.pdf'); plt.close(fig)
        text += ['', f'![Comparison]({label}.png)', '']
    text += ['## Interpretation limits', ''] + ['- ' + c for c in report['caveats']]
    (HERE / 'comparison.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({k: dict(before_read_ms=v['before']['read_ms']['mean'], after_read_ms=v['after']['read_ms']['mean']) for k, v in report['pairs'].items()}, indent=2))


if __name__ == '__main__':
    main()
