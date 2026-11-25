"""Select a physical NVIDIA GPU without importing a CUDA runtime."""
import csv
import io
import subprocess
import time


def gpu_inventory():
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
        rows = []
        for index, uuid, used, total, utilization in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            rows.append(dict(index=int(index), uuid=uuid.strip(), memory_used_mib=int(used),
                             memory_total_mib=int(total), utilization_percent=int(utilization)))
        if not rows:
            raise ValueError('No NVIDIA GPUs were reported')
        return rows
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError(f'Cannot select GPU: {exc}') from exc


def select_gpu(selection, preferred_uuid=None, exclude_uuids=()):
    """Resolve host indices to UUIDs so host and Docker address the same card.

    Auto selection ranks the peak utilization across three samples, then used
    memory. Auto can exclude the tracking GPU so MPPI uses another card.
    This is a launch-time choice, not live migration or a GPU reservation.
    """
    requested = str(selection).strip().lower()
    if requested.startswith('cuda:'):
        requested = requested[5:]
    if requested != 'auto' and (not requested.isascii() or not requested.isdigit()):
        raise ValueError('--gpu/--device requires auto or a physical GPU index, e.g. 3 or cuda:3')
    samples = [gpu_inventory()]
    if requested == 'auto':
        for _ in range(2):
            time.sleep(.2)
            samples.append(gpu_inventory())
    devices = {row['index']: dict(row) for row in samples[0]}
    for sample in samples[1:]:
        if {row['index']: row['uuid'] for row in sample} != {i: d['uuid'] for i, d in devices.items()}:
            raise ValueError('GPU inventory changed while selecting a device; retry')
        for row in sample:
            device = devices[row['index']]
            for key in ('utilization_percent', 'memory_used_mib'):
                device[key] = max(device[key], row[key])
    if requested == 'auto' and preferred_uuid is not None:
        matches = [d for d in devices.values() if d['uuid'] == preferred_uuid]
        if not matches:
            raise ValueError('The active tracker GPU is unavailable; use --restart-tracking to select again')
        chosen = matches[0]
    elif requested == 'auto':
        candidates = [d for d in devices.values() if d['uuid'] not in exclude_uuids]
        if not candidates:
            raise ValueError('No separate GPU is available for MPPI; explicitly set --mppi-gpu to the tracking GPU to share it')
        chosen = min(candidates, key=lambda d: (d['utilization_percent'], d['memory_used_mib'], d['index']))
    else:
        if int(requested) not in devices:
            raise ValueError(f'GPU {requested} is unavailable; available indices: {sorted(devices)}')
        chosen = devices[int(requested)]
    return dict(requested=str(selection), **chosen, cuda_visible_devices=chosen['uuid'], local_device='cuda:0')
