"""Install verified split exports, then remove only the prior Desktop ZIP."""
import hashlib
import json
import os
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
DESKTOP=Path('/home/chetan/Desktop')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    exports=json.loads((HERE/'M2_split_12_episode_exports_20261121.json').read_text())
    assert len(exports)==2 and {e['batch'] for e in exports}=={'batch_01','batch_02'}
    installed=[]
    for item in exports:
        source=Path(item['path'])
        assert source.parent==HERE and source.name.startswith('M2_12_completed_episodes_20261121_batch_')
        assert source.stat().st_size==item['bytes'] and sha(source)==item['sha256']
        destination=DESKTOP/source.name
        if not destination.exists():
            pending=destination.with_suffix('.zip.partial')
            with source.open('rb') as incoming,pending.open('xb') as outgoing:
                shutil.copyfileobj(incoming,outgoing,4*1024*1024)
            assert pending.stat().st_size==item['bytes'] and sha(pending)==item['sha256']
            os.link(pending,destination)  # Publish atomically without overwriting.
            pending.unlink()
        assert destination.stat().st_size==item['bytes'] and sha(destination)==item['sha256']
        installed.append(dict(path=str(destination),bytes=item['bytes'],sha256=item['sha256']))
    # Both replacement ZIPs must be intact before the old Desktop file is removed.
    old=DESKTOP/'M2_24_completed_episodes_20261121.zip'
    old_expected=json.loads((HERE/'M2_24_completed_episodes_20261121.zip.verification.json').read_text())
    if old.exists():
        assert sha(old)==old_expected['sha256'],'Old Desktop filename now contains different data; refusing deletion.'
        old.unlink()
    print(json.dumps(dict(installed=installed,old_desktop_zip_removed=not old.exists()),indent=2))


if __name__=='__main__':main()
