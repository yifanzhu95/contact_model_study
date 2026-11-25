"""Split the verified 24-episode export into two self-contained batch ZIPs."""
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'M2_24_completed_episodes_20261121.zip'
SOURCE_PREFIX='M2_24_completed_episodes_20261121/'
OUTPUTS=[('batch_01','030146_030714'),('batch_02','031802_032712')]


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    results=[]
    with zipfile.ZipFile(SOURCE) as source:
        manifest=json.loads(source.read(SOURCE_PREFIX+'manifest.json'))
        old_inventory={item['path']:item for item in manifest['files']}
        rows=list(csv.DictReader(io.StringIO(source.read(SOURCE_PREFIX+'episodes.csv').decode())))
        for batch,suffix in OUTPUTS:
            stem=f'M2_12_completed_episodes_20261121_{batch}_{suffix}'
            destination=HERE/(stem+'.zip');temporary=HERE/(stem+'.zip.partial')
            if destination.exists() or temporary.exists():raise FileExistsError(destination)
            selected=[r for r in rows if r['batch']==batch];assert len(selected)==12
            episodes=[];inventory=[];names=set()
            with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as output:
                def add_data(name,data,origin=None):
                    assert name not in names;names.add(name)
                    if isinstance(data,str):data=data.encode()
                    output.writestr(stem+'/'+name,data)
                    inventory.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),source_archive_member=origin))
                def add_json(name,value,origin=None):add_data(name,json.dumps(value,indent=2,allow_nan=False)+'\n',origin)
                def copy_member(old,new=None):
                    new=old if new is None else new
                    assert new not in names;names.add(new)
                    incoming_info=source.getinfo(SOURCE_PREFIX+old)
                    info=zipfile.ZipInfo(stem+'/'+new,date_time=incoming_info.date_time)
                    info.compress_type=incoming_info.compress_type
                    digest=hashlib.sha256();size=0
                    with source.open(incoming_info) as incoming,output.open(info,'w',force_zip64=True) as outgoing:
                        for block in iter(lambda:incoming.read(4*1024*1024),b''):
                            outgoing.write(block);digest.update(block);size+=len(block)
                    assert digest.hexdigest()==old_inventory[old]['sha256'],old
                    inventory.append(dict(path=new,bytes=size,sha256=digest.hexdigest(),source_archive_member=old))

                for number,row in enumerate(selected,1):
                    old_folder=row['folder'];outcome=row['outcome']
                    group='successes' if outcome=='success' else 'failures'
                    new_folder=f'{group}/episode_{number:02d}_{outcome}'
                    episodes.append(dict(row,episode=number,source_combined_archive_episode=int(row['episode']),folder=new_folder))
                    for old in old_inventory:
                        if not old.startswith(old_folder+'/'):continue
                        new=new_folder+old[len(old_folder):]
                        if old.endswith('/episode_log.json') or old.endswith('/metrics.json'):
                            value=json.loads(source.read(SOURCE_PREFIX+old))
                            value.update(archive_episode=number,source_combined_archive_episode=int(row['episode']))
                            add_json(new,value,old)
                        else:copy_member(old,new)
                # Preserve each batch's complete aggregate and original process logs.
                for old in old_inventory:
                    if (old.startswith(f'batches/{batch}/') or old.startswith(f'calibration/{batch}/')
                        or old.startswith(f'configuration/{batch}/')
                        or old=='configuration/calibrated_hand_used.xml'
                        or old.startswith('calibration/method5_manual_fit_')):
                        copy_member(old)
                stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(episodes[0]))
                writer.writeheader();writer.writerows(episodes);add_data('episodes.csv',stream.getvalue())
                counts={key:sum(e['outcome']==key for e in episodes) for key in ['success','failure','timeout']}
                expected=dict(success=7,failure=5,timeout=0) if batch=='batch_01' else dict(success=6,failure=5,timeout=1)
                assert counts==expected,counts
                provenance=next(s for s in manifest['sources'] if s['batch']==batch)
                add_json('summary.json',dict(batch=batch,completed_episodes=12,successes=counts['success'],
                    drops=counts['failure'],timeouts=counts['timeout'],success_rate=counts['success']/12,
                    published_actions=sum(int(e['steps']) for e in episodes),
                    manipulation_duration_s=sum(float(e['duration_s']) for e in episodes),source_batch=provenance))
                add_data('README.txt',
                    f'M2 — TWELVE COMPLETED EPISODES — {batch} ({suffix})\n\n'
                    f"{counts['success']} successes, {counts['failure']} drops, {counts['timeout']} timeout(s).\n"
                    'successes/ and failures/ contain episode videos, full measured joint/object\n'
                    'state and action logs, exact video-frame timestamps, recording metadata and metrics.\n'
                    'Episodes are numbered 01–12 in chronological order within THIS batch.\n'
                    'source_combined_archive_episode preserves the earlier 24-episode export ID.\n\n'
                    'batches/ contains this batch aggregate, original process logs, summaries,\n'
                    'per-step/per-episode CSVs and analysis plots. Some comparison plots include\n'
                    'the preceding batch for context; the selected data/videos here are only 12 episodes.\n'
                    'Later aborted attempts are excluded from selected videos and counts, although\n'
                    'original source process logs retain their diagnostic messages.\n\n'
                    'calibration/ contains saved tracker-publication calibration, the effective\n'
                    'selected camera-to-world transform and the verified Method 5 manual-fit candidate.\n'
                    'configuration/ contains frozen hardware/planner settings, costs, measured limits,\n'
                    'joint-cost reference, rollout/evaluation scene XML and calibrated hand XML.\n'
                    'Scene asset paths refer to the original checkout; mesh assets are not bundled.\n\n'
                    'Paced M2, 512 samples, horizon 0.256s, control step 0.064s, temperature 0.1,\n'
                    'noise 0.05, shifted warm start, action-rate 10, blending .75, limit cost 100,\n'
                    '+/-0.2 delta clip followed by measured limits; w_pos_y=1 and w_contact=80.\n'
                    'Quaternions use wxyz. State/action arrays include saved joint order and timestamps.\n'
                    'Hardware clipping/target metrics are controller estimates; bridge telemetry was off.\n'
                    'The earlier batch averaged 16.3ms joint-feedback age; the later averaged 1.54ms.\n'
                    'Videos are original MJPEG-copy Matroska files without reencoding; these runs\n'
                    'predate the subsequent one-second recording-padding change.\n'
                    'manifest.json records SHA-256 for every payload file. Source data was not changed.\n')
                add_data('split_export_script.py',Path(__file__).read_bytes())
                output.writestr(stem+'/manifest.json',json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    batch=batch,episode_count=12,outcomes=counts,source=provenance,
                    source_archive=SOURCE.name,files=inventory),indent=2)+'\n')
            with zipfile.ZipFile(temporary) as check:
                assert check.testzip() is None
                assert sum(p.endswith('/video.mkv') for p in check.namelist())==12
                assert not any(f'batches/{"batch_02" if batch=="batch_01" else "batch_01"}/' in p for p in check.namelist())
                members=len(check.namelist())
            temporary.rename(destination)
            result=dict(path=str(destination),bytes=destination.stat().st_size,sha256=file_sha(destination),
                        batch=batch,videos=12,outcomes=counts,archive_members=members,crc_check='passed')
            destination.with_suffix('.zip.verification.json').write_text(json.dumps(result,indent=2)+'\n')
            results.append(result);print(json.dumps(result),flush=True)
    (HERE/'M2_split_12_episode_exports_20261121.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
