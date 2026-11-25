"""Remove only earlier Nov 21 hardware-run videos, preserving all other files.

Default: print inventory. --delete: save inventory with hashes, then delete the
listed media. The fixed day/cutoff deliberately prevent a broader cleanup.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'run_leap_hardware.py').is_file())
DAY='20261121'
CUTOFF='leap_experiment_20261121_030146_687194'
SUFFIXES={'.mkv','.mp4','.avi','.mov','.webm'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delete',action='store_true');args=parser.parse_args()
    provenance=json.loads((OUT/'sources.json').read_text())
    protected=[]
    for video in provenance['videos']:
        for item in video['files']:
            p=OUT/item['path']
            assert p.is_file() and sha(p)==item['sha256'],p
            protected.append(str(p.relative_to(ROOT)))
    assert len([p for p in protected if Path(p).suffix=='.mkv'])==12
    candidates=[];logs={}
    for run in sorted((ROOT/'results').glob(f'**/leap_experiment_{DAY}_*')):
        if not run.is_dir() or not re.fullmatch(r'leap_experiment_\d{8}_\d{6}_\d{6}',run.name):continue
        if run.name>=CUTOFF:continue
        media=[p for p in (run/'video').rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES]
        if not media:continue
        for p in media:
            assert p.resolve().is_relative_to((run/'video').resolve())
            candidates.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
        for filename in ['controller.json','bridge.log','controller.log','launch.json']:
            p=run/filename
            if p.is_file():logs[str(p.relative_to(ROOT))]=sha(p)
    result=dict(scope='Only video media from hardware run folders dated 2026-11-21 before the selected 12-episode batch. Other dates, logs, timestamps and selected videos preserved.',
        client_timezone='America/Phoenix',cutoff=CUTOFF,created_at_utc=datetime.now(timezone.utc).isoformat(),
        files=candidates,file_count=len(candidates),bytes=sum(c['bytes'] for c in candidates),
        protected_aggregate_files=protected,preserved_log_sha256=logs,status='planned')
    manifest=OUT/'video_cleanup_20261121.json'
    if not args.delete:
        print(json.dumps(result,indent=2));return
    if not candidates:
        print('No earlier-today videos remain; previous cleanup manifest retained.');return
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    for candidate in candidates:
        p=ROOT/candidate['path'];assert p.stat().st_size==candidate['bytes']
        p.unlink()
    for path,digest in logs.items():assert sha(ROOT/path)==digest
    for video in provenance['videos']:
        for item in video['files']:assert sha(OUT/item['path'])==item['sha256']
    result.update(status='deleted',verification='All listed media absent; selected 12 videos and all checked source logs retain their hashes.',
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    assert all(not (ROOT/c['path']).exists() for c in candidates)
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','protected_aggregate_files','preserved_log_sha256']},indent=2))


if __name__=='__main__':main()
