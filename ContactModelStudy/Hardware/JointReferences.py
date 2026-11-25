"""Object-specific measured joint-cost references, separate from initialization."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from .Hand import vector
from .Scene import JOINTS, ROOT

DEFAULT_REFERENCE_DIR = ROOT / 'configs/leap_joint_references'
OBJECTS = ('cube','ball','duck')


def reference_path(obj):
    if obj not in OBJECTS:
        raise ValueError(f'Unsupported reference object: {obj}')
    return DEFAULT_REFERENCE_DIR / f'{obj}.json'


def load_reference(obj, path=None):
    """Missing default means legacy home; an explicitly requested file must exist."""
    selected = reference_path(obj) if path is None else Path(path).expanduser()
    if not selected.exists() and path is None:
        return None
    payload = selected.read_bytes()
    record = json.loads(payload)
    if record.get('schema_version') != 1 or record.get('object') != obj:
        raise ValueError(f'Wrong schema/object in joint reference {selected}')
    if record.get('joint_order') != list(JOINTS):
        raise ValueError(f'Wrong joint order in joint reference {selected}')
    reference = vector(record.get('joint_reference'), 'joint_reference')
    return dict(path=str(selected.resolve()),sha256=hashlib.sha256(payload).hexdigest(),
                joint_reference=reference.tolist(),record=record)


def save_reference(obj, reference, metadata, path=None):
    """Atomically replace the per-object file; retain an exact copy of the old one."""
    selected = reference_path(obj) if path is None else Path(path).expanduser()
    reference = vector(reference, 'joint_reference')
    now = datetime.now(timezone.utc)
    record = {**metadata,'schema_version':1,'object':obj,
              'captured_at_utc':now.isoformat(),'joint_order':list(JOINTS),
              'joint_reference':reference.tolist()}
    serialized = json.dumps(record,indent=2,allow_nan=False)+'\n'
    selected.parent.mkdir(parents=True,exist_ok=True)
    if selected.exists():
        history = selected.parent/'history'
        history.mkdir(exist_ok=True)
        shutil.copyfile(selected,history/f'{selected.stem}_{now.strftime("%Y%m%dT%H%M%S_%fZ")}.json')
    temporary = selected.with_name(selected.name+'.tmp')
    try:
        temporary.write_text(serialized)
        temporary.replace(selected)
    finally:
        temporary.unlink(missing_ok=True)
    return selected
