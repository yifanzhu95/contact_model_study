"""Plot saved baseline planning horizons; no simulation or hardware access."""
import csv
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mpl_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_plot():
    report = json.loads((OUT / 'report.json').read_text())
    accuracies = ['high', 'med', 'low']
    colors = {'M1': '#3977ad', 'M2': '#de8836'}
    rows = []
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for j, model in enumerate(['M1', 'M2']):
        values, seconds = [], []
        for accuracy in accuracies:
            source = report['source_manifest'][model + '_' + accuracy]
            period = source['rollout_config']['ctrl_time_step']
            horizon = source['horizon_s']
            steps = round(horizon / period)
            assert np.isclose(period, .064) and np.isclose(steps * period, horizon)
            rows.append(dict(model=model, hand_accuracy=accuracy, horizon_steps=steps,
                             horizon_s=horizon, control_timestep_s=period))
            values.append(steps)
            seconds.append(horizon)
        positions = np.arange(3) + (j - .5) * .36
        ax.bar(positions, values, width=.34, color=colors[model], label=model)
        for x, steps, duration in zip(positions, values, seconds):
            ax.text(x, steps + .10, f'{steps} steps\n{duration:.3f} s', ha='center',
                    va='bottom', fontsize=11)
    ax.set(xticks=range(3), xticklabels=['High', 'Medium', 'Low'],
           xlabel='Hand geometry accuracy (object geometry: high)',
           ylabel='Planning horizon (control steps)', ylim=(0, 6.3), yticks=range(7),
           title='Planning horizons used in the M1–M2 hardware comparison')
    right = ax.secondary_yaxis('right', functions=(lambda x: x * .064,
                                                  lambda x: x / .064))
    right.set_ylabel('Planning horizon (seconds)')
    right.set_yticks(np.arange(7) * .064)
    right.set_yticklabels([f'{x:.3f}' for x in np.arange(7) * .064])
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='upper left', frameon=False)
    fig.text(.5, .025, '1 control step = 0.064 s  •  512 samples  •  Accepted baselines; M1 two-step tests excluded',
             ha='center', fontsize=10, color='#444444')
    fig.tight_layout(rect=(0, .06, 1, 1))
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT / 'plots' / f'planning_horizons.{suffix}', dpi=180,
                    bbox_inches='tight')
    plt.close(fig)
    with (OUT / 'planning_horizons.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == '__main__':
    rows = make_plot()
    index = OUT / 'index.html'
    text = index.read_text()
    if 'plots/planning_horizons.png' not in text:
        section = ('<h2>Planning horizons</h2><p>One control step is 0.064 seconds. '
                   '<a href="planning_horizons.csv">Data CSV</a> · '
                   '<a href="plots/planning_horizons.pdf">PDF</a></p>'
                   '<img src="plots/planning_horizons.png">\n')
        text = text.replace('<h2>Episodes and videos</h2>', section + '<h2>Episodes and videos</h2>')
        index.write_text(text)
    print(json.dumps(rows, indent=2))
