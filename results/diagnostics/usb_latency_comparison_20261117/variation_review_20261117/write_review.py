"""Summarize the offline analysis and retain the broader historical inventory."""
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def main():
    report = json.loads((OUT / 'report.json').read_text())
    rows = list(csv.DictReader((OUT / 'episodes.csv').open()))
    current = [r for r in rows if r['cohort'].startswith('usb1')]
    assert len(rows) == 140 and len(current) == 50
    assert Counter(r['outcome'] for r in current) == {'success': 8, 'failure': 33, 'timeout': 9}
    assert not any(r['source'].endswith('022352_015441/controller.json') and r['source_episode'] == '6' for r in rows)
    assert not any(r['source'].endswith('013433_789897/controller.json') and r['source_episode'] == '9' for r in rows)
    assert len({(r['source'], r['source_episode']) for r in rows}) == len(rows)
    for source in report['sources']:
        assert hashlib.sha256((ROOT / source['source']).read_bytes()).hexdigest() == source['sha256']
    for r in rows:
        assert (r['both_pose_ever'] == 'True') == (r['success'] == 'True')

    inventory_path = ROOT / 'results/diagnostics/hardware_physical_parameter_review_20261116/inventory.json'
    inventory = json.loads(inventory_path.read_text())
    groups = inventory['groups']
    # This is a preserved exploratory inventory, not an added causal comparison.
    with (OUT / 'historical_all_variants.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(groups[0]))
        writer.writeheader()
        for g in groups:
            writer.writerow({**g, 'sources': json.dumps(g['sources'])})

    rotation = []
    for era in ['historical', 'usb1']:
        for direction in ['-X', '+X', '-Y', '+Y', '-Z', '+Z']:
            rr = [r for r in rows if r['cohort'].startswith(era) and r['off90'] == 'False' and r['direction'] == direction]
            rotation.append(dict(era=era, direction=direction, episodes=len(rr), successes=sum(r['success'] == 'True' for r in rr)))
    with (OUT / 'rotation_totals.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rotation[0])); writer.writeheader(); writer.writerows(rotation)

    text = [
        '# M2 parameter review — 17 November 2026',
        '',
        'Offline analysis only. No simulation or hardware was run; no production settings or raw logs were changed.',
        '',
        'The final episode (6) of run `20261117_022352_015441` is excluded by user request, although it was logged successful. The current-cutoff-contaminated episode 9 of `20261117_013433_789897` and aborted/uncheckpointed episodes are also excluded. The latest retained batch is therefore **2/5**, not 3/6.',
        '',
        '## Scope and comparison discipline',
        '',
        f'Recomputed **140 comparable M2 episodes**: 50 current USB-1ms episodes and 90 historical episodes. Checked saved weights, P011 joint-cost reference, physics/contact configuration, hardware settings, transform, thresholds and scene XML for equality apart from clipping, temperature and pacing. These runs use trial-314 weights explicitly passed on the command line, irrespective of the weight block currently active in CubeReorient.py. All use a calibrated high/high hand, 70 g cube, sliding coefficients cube/tip/palm/other = .3/.5/.4/.4, 512 samples, noise .05, shifted warm start, .32 s horizon and .064 s model control timestep.',
        '',
        'The established curated 24+24 historical baseline is preserved. Historical USB settings were not embedded in each run, so historical/current rates are reported separately. GPU/tracker load, placements and run order remain uncontrolled. The broader historical inventory is retained as an appendix; it includes exploratory configurations and is not pooled with this matched comparison.',
        '',
        '## Current USB-1ms variants',
        '',
        '| Clip (rad) | Temperature | Pacing | Success | Drop | Timeout | Actions clipped | Mean planning | Effective Hz |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    order = ['usb1_c0.2_T0.1_U', 'usb1_c0.1_T0.1_U', 'usb1_c0.2_T0.15_U', 'usb1_c0.2_T0.1_P', 'usb1_c0.15_T0.1_P']
    for key in order:
        g = report['cohorts'][key]; _, clip, temp, pacing = key.split('_')
        text.append(f"| ±{clip[1:]} | {temp[1:]} | {'Yes' if pacing == 'P' else 'No'} | {g['successes']}/{g['episodes']} ({100*g['success_rate']:.1f}%) | {g['drops']} | {g['timeouts']} | {100*g['clipped_action_fraction']:.1f}% | {g['planning_ms']['mean']:.1f} ms | {g['effective_hz']:.2f} |")
    text += [
        '',
        'An action is counted clipped if at least one of its 16 joint deltas exceeds the bounds. This is distinct from the fraction of individual joint values clipped. Pooled fractions are step-weighted: long timeouts count more heavily, which is why per-episode and early-window metrics are also saved.',
        '',
        '### 1. Clipping trades completion against retention',
        '',
        'At ±.1, five of eleven episodes timed out and five dropped; six of eleven finished without a drop. At ±.2 unpaced, two of 24 timed out and eighteen dropped; six of 24 finished without a drop. Median time to drop among episodes that dropped was 21.3 s at ±.1 versus 7.8 s at ±.2. These are descriptive finish outcomes, not fixed-time survival estimates, because success stops an episode early.',
        '',
        '±.1 clips 97.7% of actions and 29.2% of individual joint deltas; ±.2 unpaced clips 39.0% and 3.7%, respectively. With tight clipping, the executed action often differs substantially from the action whose rollout was scored. Lower measured target error under ±.1 does not demonstrate better dynamics: the target is mechanically constrained to remain closer to the measured joint state.',
        '',
        '±.15 paced has the best observed current rate, 2/5, but its 95% Wilson interval is approximately 12–77%. It includes two -X starts, two +X starts and one -Y start, with no +Y or Z starts. The older ±.15 unpaced batch was 0/6. There is no established optimum at ±.15 and no evidence that success improves monotonically as clipping is changed.',
        '',
        '### 2. Pacing is barely active',
        '',
        'At ±.2, pacing waited on 24/1140 steps (2.11%), for .1546 s total: .185% of manipulation time. At ±.15, it waited on 3/1320 steps (.227%), for .00358 s total: .00363% of manipulation time. Mean planning remains roughly 72–74 ms and effective control frequency roughly 13.4–13.8 Hz. The pacing option does not make a planning-bound loop run at 15.625 Hz. These tiny waits provide little support for attributing success-rate differences to pacing.',
        '',
        '### 3. Temperature has a narrow, non-monotonic useful range',
        '',
        '| Temperature, clip ±.2, unpaced | Historical success | Current USB-1ms success |',
        '|---|---|---|',
        '| .05 | 0/6 | Not tested in this cohort |',
        '| .1 | 11/24 | 4/24 |',
        '| .125 | 6/12 | Not tested |',
        '| .15 | 0/6 | 0/3 |',
        '',
        'Historical .125 is a plausible intermediate candidate, not a proven improvement over .1. Raising temperature to .15 reduced clipping and produced more current timeouts, but has not demonstrated more reliable completion. Lowering it to .05 also did poorly historically. Smoother or smaller commands are not by themselves a success metric.',
        '',
        '## Orientation and position patterns',
        '',
        '### 4. Starting direction is a major source of unequal difficulty',
        '',
        'The labels below denote the dominant signed starting rotation axis relative to the identity goal, restricted to starting angles 75–105°. Two current starts outside that interval are omitted only from this direction table, not from overall success rates.',
        '',
        '| Starting rotation | Historical success | Current success |',
        '|---|---|---|',
    ]
    for direction in ['-X', '+X', '-Y', '+Y', '-Z', '+Z']:
        a, b = [r for r in rotation if r['direction'] == direction]
        text.append(f"| {direction} | {a['successes']}/{a['episodes']} | {b['successes']}/{b['episodes']} |")
    text += [
        '',
        '+Z is 0/20 across the two eras; -X is 1/27. Current -Y is 5/11, while current +X is 1/10. Thus an apparent parameter win can come from its rotation mix. These results point toward rotation-specific contact/regrasp difficulties and warrant judging each candidate on all six starts. They do not identify a defective joint or prove a particular physical mechanism.',
        '',
        '### 5. The clearest near-success positional bottleneck is +X',
        '',
        '**35 of the 42 current non-successes never reached the orientation threshold at all.** Position-only changes therefore address a subset, not the main population of failures. This does not imply that position objectives are unrelated to rotation: they can influence the actions that prevent reaching orientation in the first place.',
        '',
        'Success requires simultaneous position <20 mm, quaternion error <.04 (about 23.07°), and velocity norm <10. Among the 42 current non-successes, five reached the orientation and velocity thresholds while their closest position was 22.16–27.99 mm. At those same instants, every episode had +X as its largest position-error component. Median signed errors were approximately X=+21.2 mm, Y=+5.3 mm and Z=-6.3 mm. One case also had substantial Y error (+16.5 mm). These are concurrent samples; orientation and position minima from different times were not combined.',
        '',
        'For these near misses, the stage cost is strongly anisotropic: w_pos_z=38.686 versus w_pos_x=4.350. At the component-wise median offsets, their weighted absolute penalties are about .242 for Z and .092 for X despite the larger X displacement. The terminal position term is isotropic, so this is not a complete cost decomposition or proof of the cause. It does, however, motivate a modest isolated X-position weight test (4.3496 → 5.5) before changing all position weights or success criteria. The remaining 37 non-successes are not explained by this narrow near-miss pattern.',
        '',
        'Successful episodes finish at median position error 17.9 mm and orientation error 19.5°, close to the existing boundaries. Current timeouts finish at median 27.8 mm and 30.9°. Drop endpoints have median 167.9 mm and 117.5°; averaging these together with successes would mainly measure whether the cube fell, rather than near-goal control quality. The simultaneous-error scatter uses the best maximum normalized position/orientation/velocity score within each episode to avoid presenting unrelated minima as a near success.',
        '',
        '### 6. Useful correlations, with limits',
        '',
        '- Orientation progress during the first second is the strongest measured early marker: successful episodes improved by a median 32.4°, versus -2.8° for all non-successes. Its episode-level Spearman correlation with success is +.45; after centering ranks within clipping/temperature/pacing configurations it is +.44. This is a progress indicator, not evidence that forcing faster joint motion will help.',
        '- Early clipped-joint fraction correlates with drops within configurations (centered rank r≈+.38). The pooled correlation is near zero, because tight-clip configurations both clip more and retain the cube longer. This reversal shows why a single pooled clipping/performance correlation would be misleading. Within a fixed bound, unusually large requested changes may signal a struggling plan; clipping may also be a consequence of that struggle.',
        '- Larger early target changes correlate modestly with drops in the pooled data (Spearman +.33), weakening to +.17 within configurations. Measured hand-speed RMS has little association with success (+.04 pooled, +.07 within configurations). The evidence supports coherent rotation progress more clearly than simply slowing all fingers.',
        '- Initial position error has weak association with success (-.08 pooled), and whole-episode mean object age has essentially none (+.03 pooled). This does not establish that placement or delay is harmless: initial position varies over a narrow range, contact state is unmeasured, and similar average delay can affect different motions differently.',
        '',
        'All correlations use episodes as the unit, not autocorrelated timesteps. There are only eight current successes; no causal inference, optimal thresholds or independently validated predictor is claimed. Rank centering removes configuration mean-rank differences, not orientation or run-order confounding.',
        '',
        '## What the broader variations do and do not establish',
        '',
        f'The preserved earlier inventory covers {inventory["unique_completed_episodes"]} unique completed hardware episodes in {len(groups)} parameter groups. See historical_all_variants.csv for model, weights, mass, surface friction, clipping, sample count, temperature, noise, horizon and warm-start labels, with source paths. This point-in-time inventory includes exploratory runs and predates the current USB cohorts; it is not a new fully re-curated 724-episode comparison.',
        '',
        '- The original M2 hardware comparison achieved 5/10 with old model mass/friction, but a repeated old-physics baseline achieved 1/6. New friction alone at old mass achieved 2/6, and corrected mass/plastic .4 achieved 2/6 with original weights. These do not identify friction or justify reverting the known 70 g mass just to improve success rate.',
        '- Lowering fingertip/plastic friction together to .4/.3 with trial-314 weights and ±.1 clipping produced 0/6, against 1/6 at .5/.4 in the comparable historical ±.1 batch. There is no demonstrated benefit from lowering friction further to induce stronger grasps; success cannot uniquely identify contact coefficients.',
        '- With current physics and ±.2 clipping, the selected original-weight paced batch was 3/12, while the trial-314 paced baseline was 6/24. These are both 25%; they do not establish superiority of either weight set. The trial-314 unpaced baseline was 11/24. Separate simulation weight/reference comparisons used different execution assumptions and cannot be substituted for hardware success rates.',
        '- The historical trial-314 256-sample shifted batch was 2/6 versus the curated 512-sample unpaced 11/24. This is insufficient to establish a sample-count benefit; 256 samples can be a useful latency/computation ablation, not an automatic cure for drops.',
        '- Hardware step/sine tests established that USB latency reduction sped up feedback and command transmission, while measured servo phase lag and settled tracking offsets changed little. Current manipulation telemetry likewise shows roughly 1.6 ms publish-to-write time, 2.93–2.94 ms feedback intervals, and mean absolute joint target error 2.35–3.22°. Communication speed is not identical to physical command following.',
        '',
        'The exact bridge-written held target, not an estimated controller target, was used for current following-error statistics. These are joint-time-weighted tracking errors, not settled errors or independent joint samples. Smaller clipping mechanically limits the held-target error. Original object samples are typically approximately 148–160 ms old by plan completion even though freshly read hand feedback at publication averages approximately 1.5–1.6 ms. State-to-action alignment remains an engineering opportunity, not a demonstrated outcome correlation.',
        '',
        '## Recommended order of work',
        '',
        '1. Retain noise .05, 512 samples, .32 s horizon, trial-314 weights, P011, mass .07 and friction .3/.5/.4/.4 while resolving the clipping question. Complete balanced six-direction repetitions at ±.15/T=.1, including +Y and both Z directions. Pair or interleave with ±.2/T=.1; do not infer a pacing benefit from negligible waits.',
        '2. For a distinct targeted cost experiment, hold the selected bound/temperature fixed and change only w_pos_x to 5.5. Judge whether the +X near misses decrease without more drops or worse orientation. This proposal addresses five near misses, not all failure modes. Keep current success thresholds and target height unchanged for that comparison.',
        '3. If testing temperature next instead, use .125 as an isolated candidate; do not combine it immediately with a clipping or cost-weight change. Historical .125 is promising but has no current USB-1ms confirmation.',
        '4. Investigate pose time alignment / delay-aware planning separately, and use the hard +Z and -X starts as regression cases. Better communication timing alone did not resolve the manipulation failures. No safety limit or motor-gain change is supported by this analysis.',
        '',
        '## Artifacts',
        '',
        '- [All plots and metric tables](index.html)',
        '- [Full machine-readable report](report.json)',
        '- [Episode-level metrics](episodes.csv)',
        '- [Selection and exclusions](selection.json)',
        '- [Near-miss samples](near_misses.csv)',
        '- [Exploratory correlations](correlations.csv)',
        '- [All earlier parameter groups](historical_all_variants.csv)',
        '- Every plot is saved separately as PNG and PDF.',
    ]
    (OUT / 'review.md').write_text('\n'.join(text) + '\n')
    page = OUT / 'index.html'
    s = page.read_text().split('<h2>Interpretation and recommendations</h2>', 1)[0]
    # Keep the detailed narrative accessible without an additional rendering dependency.
    s += '<h2>Interpretation and recommendations</h2><p><a href="review.md">Readable analysis (Markdown)</a> · <a href="historical_all_variants.csv">All earlier parameter groups</a></p><pre style="white-space:pre-wrap">' + html.escape('\n'.join(text)) + '</pre>'
    page.write_text(s)
    (OUT / 'validation.json').write_text(json.dumps(dict(
        current_episodes=50, historical_episodes=90, current_successes=8,
        user_final_episode_excluded=True, known_current_fault_excluded=True,
        raw_source_hashes_unchanged=True, duplicate_selected_episodes=0,
        historical_inventory_groups=len(groups),
        no_simulation_or_hardware_run=True), indent=2) + '\n')
    print('Validated selection, thresholds and source hashes; saved review.md and historical appendix.')


if __name__ == '__main__':
    main()
