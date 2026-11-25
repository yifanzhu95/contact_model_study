"""Archive the two explicitly selected completed M2 batches, without mutation.

Run from this checkout. Creates a ZIP in results/exports; copying to Desktop
is a separate step. Stores JPEG-copy videos without redundant compression.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/baseline_trial314_shift512_h0256_paced_measured_limits'
BATCHES=['combined_12_completed_20261121_030146_030714','combined_12_completed_20261121_031802_032712']
DEST=Path(__file__).with_name('M2_24_completed_episodes_20261121.zip')
PREFIX='M2_24_completed_episodes_20261121/'


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    if DEST.exists():raise FileExistsError(f'Refusing to overwrite {DEST}')
    temporary=DEST.with_suffix('.zip.partial')
    if temporary.exists():raise FileExistsError(f'Refusing to overwrite {temporary}')
    inventory=[];episode_rows=[];sources=[];calibration_sources={};hashes_by_batch={}
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as archive:
        names=set()
        def add_data(name,data,source=None):
            if name in names:raise ValueError(f'Duplicate archive path {name}')
            names.add(name)
            if isinstance(data,str):data=data.encode()
            archive.writestr(PREFIX+name,data)
            inventory.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),source=source))
        def add_json(name,data,source=None):
            add_data(name,json.dumps(data,indent=2,allow_nan=False)+'\n',source)
        def add_file(name,path,expected=None):
            if name in names:raise ValueError(f'Duplicate archive path {name}')
            names.add(name);digest=hashlib.sha256();size=0
            info=zipfile.ZipInfo.from_file(path,PREFIX+name)
            info.compress_type=zipfile.ZIP_STORED if path.suffix.lower() in ('.mkv','.mp4','.png','.pdf') else zipfile.ZIP_DEFLATED
            with path.open('rb') as incoming,archive.open(info,'w',force_zip64=True) as output:
                for block in iter(lambda:incoming.read(4*1024*1024),b''):
                    output.write(block);digest.update(block);size+=len(block)
            actual=digest.hexdigest()
            if expected is not None:assert actual==expected,(path,expected,actual)
            inventory.append(dict(path=name,bytes=size,sha256=actual,source=str(path.relative_to(ROOT))))

        for batch_number,batch_name in enumerate(BATCHES,1):
            folder=BASE/batch_name;batch=f'batch_{batch_number:02d}'
            doc=json.loads((folder/'controller.json').read_text());provenance=json.loads((folder/'sources.json').read_text())
            report=json.loads((folder/'analysis/report.json').read_text())
            assert len(doc['episodes'])==12 and all(e['completed'] for e in doc['episodes'])
            videos={v['episode']:v for v in provenance['videos']}
            metric_rows={r['episode']:r for r in report['episodes']}
            for filename in ['controller.json','summary.json','episodes.csv','steps.csv','sources.json','analyze_and_group.py']:
                add_file(f'batches/{batch}/{filename}',folder/filename)
            for path in sorted((folder/'analysis').rglob('*')):
                if path.is_file() and path.suffix not in ('.pyc',):
                    add_file(f'batches/{batch}/analysis/{path.relative_to(folder/"analysis")}',path)
            add_file(f'batches/{batch}/recording.original.json',folder/'video/recording.json')
            # Keep original process logs, failure traces and launch snapshots.
            # Per-episode logs below contain only the selected completed data.
            snapshots=[]
            for source in provenance['sources']:
                archive_path=source['archived_logs'];src=folder/archive_path
                for path in sorted(src.iterdir()):
                    if path.is_file():
                        expected=source['source_hashes'].get(path.name)
                        add_file(f'batches/{batch}/{archive_path}/{path.name}',path,expected)
                launch=json.loads((src/'launch.json').read_text());snapshots.append(launch['input_snapshots'])
            snapshot=snapshots[0]
            for snap in snapshots:
                for key in ['hardware_config','camera_config','selected_camera_config','calibrated_hand_sha256','joint_limits']:
                    assert snap[key]==snapshot[key],(batch,key)
            add_json(f'calibration/{batch}/effective_pose_transform.json',doc['pose_transform'])
            add_json(f'calibration/{batch}/tracker_publication_camera_config.json',snapshot['camera_config'])
            add_json(f'calibration/{batch}/selected_camera_config.json',snapshot['selected_camera_config'])
            add_json(f'configuration/{batch}/hardware_config.json',doc['hardware_config'])
            add_json(f'configuration/{batch}/joint_limits_used.json',doc['joint_limits'])
            add_json(f'configuration/{batch}/joint_cost_reference.json',dict(joint_order=doc['joint_order'],reference_rad=doc['joint_reference']))
            add_json(f'configuration/{batch}/planner_and_task.json',{key:doc[key] for key in
                ['args','weights','planner_config','rollout_config','success_thresholds','failure_z','execution',
                 'action_rate_cost','temporal_plan_blending','effective_ctrl_range','model_ctrl_range','metric_definitions']})
            for role,scene in doc['scenes'].items():
                add_data(f'configuration/{batch}/{role}_scene_used.xml',scene['scene_xml'])
            selection=doc['pose_transform']['selected_calibration']
            calibration_sources[selection['source']]=selection['source_sha256']
            hand=Path(doc['args']['calibrated_hand']);expected=snapshot['calibrated_hand_sha256']
            assert sha(hand)==expected,'Current calibrated hand differs from the saved model; use frozen scenes instead.'
            if expected not in hashes_by_batch:
                add_file('configuration/calibrated_hand_used.xml',hand,expected)
                hashes_by_batch[expected]=batch
            for episode in doc['episodes']:
                number=episode['episode']+1;global_number=len(episode_rows)+1
                outcome=episode['finish_reason'];group='successes' if outcome=='success' else 'failures'
                identifier=f'episode_{global_number:02d}_{outcome}'
                destination=f'{group}/{identifier}';video=videos[number]
                assert video['source_run']==episode['source_run'] and video['source_episode']==episode['source_episode']
                row=dict(episode=global_number,batch=batch,batch_episode=number,source_run=episode['source_run'],
                    source_episode=episode['source_episode'],outcome=outcome,steps=episode['n_steps'],
                    duration_s=episode['manipulation_wall_time_s'],
                    initial_rotation=metric_rows[number]['rotation'],
                    final_position_error_mm=metric_rows[number]['final_position_error_mm'],
                    final_orientation_error_deg=metric_rows[number]['final_angle_deg'],folder=destination)
                episode_rows.append(row)
                log=dict(episode,archive_episode=global_number,archive_batch=batch,
                    joint_order=doc['joint_order'],metric_definitions=doc['metric_definitions'])
                add_json(destination+'/episode_log.json',log)
                add_json(destination+'/metrics.json',dict(metric_rows[number],archive_episode=global_number,batch=batch))
                add_json(destination+'/recording.json',video['recording'])
                for item in video['files']:
                    source=folder/item['path']
                    target=('video.mkv' if source.suffix=='.mkv' else 'frames.csv' if source.name.endswith('.frames.csv') else 'ffmpeg.log')
                    add_file(destination+'/'+target,source,item['sha256'])
            sources.append(dict(batch=batch,source_aggregate=str(folder.relative_to(ROOT)),completed=12,
                successes=sum(e['success'] for e in doc['episodes']),source_runs=[s['run'] for s in provenance['sources']],
                excluded_attempts=provenance.get('excluded_episodes',[])))
            print(f'Archived {batch}: 12 completed episodes',flush=True)
        for i,(source,digest) in enumerate(calibration_sources.items(),1):
            add_file(f'calibration/method5_manual_fit_{i:02d}_used.json',Path(source),digest)
        assert len(episode_rows)==24
        assert len({(r['source_run'],r['source_episode']) for r in episode_rows})==24
        counts={key:sum(r['outcome']==key for r in episode_rows) for key in ['success','failure','timeout']}
        assert counts==dict(success=13,failure=10,timeout=1),counts
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(episode_rows[0]));writer.writeheader();writer.writerows(episode_rows)
        add_data('episodes.csv',stream.getvalue())
        add_json('summary.json',dict(completed_episodes=24,successes=13,drops=10,timeouts=1,
            success_rate=13/24,published_actions=sum(r['steps'] for r in episode_rows),
            manipulation_duration_s=sum(r['duration_s'] for r in episode_rows),source_batches=sources))
        add_data('README.txt',
            'M2: TWO COMPLETED 12-EPISODE BATCHES, NOVEMBER 21 2026\n\n'
            '24 unique completed episodes: 13 successes, 10 drops, 1 timeout.\n'
            'successes/ contains the 13 successful episode videos and logs.\n'
            'failures/ contains the 10 drop episodes and the explicitly labeled timeout.\n'
            'Each episode folder includes video.mkv, exact frame capture/receive timestamps,\n'
            'full episode_log.json (measured hand/object states, requested/executed-estimate targets,\n'
            'planning times, feedback timestamps, errors and clipping), and metrics.json.\n\n'
            'batches/ holds both full aggregate logs, source-process logs, analysis plots and metrics.\n'
            'Later aborted attempts can appear in original source process logs but are EXCLUDED\n'
            'from the 24 episode folders, counts, and selected videos. A newer four-episode run\n'
            'is not part of these two twelve-episode batches.\n\n'
            'calibration/ contains SAVED effective consumer transforms, tracker-publication\n'
            'calibration snapshots and the hash-verified Method 5 manual calibration candidate.\n'
            'configuration/ contains saved hardware/planner settings, costs, joint limits,\n'
            'joint-cost reference, frozen rollout/evaluation XML and calibrated hand XML.\n'
            'No mesh assets or camera raw RGB-D datasets are included; frozen scene XML retains\n'
            'original repository asset paths. Analysis scripts require the original checkout.\n\n'
            'Both batches: paced M2, 512 samples, 0.256s horizon, 0.064s control step,\n'
            'temperature 0.1, noise 0.05, shifted warm start, action-rate 10, blending .75,\n'
            'joint-limit penalty 100, +/-0.2 clipping followed by measured absolute bounds.\n'
            'Trial314-derived costs with w_pos_y=1 and w_contact=80.\n'
            'Raw data includes position/quaternion units and joint order. Quaternions use wxyz.\n'
            'Clip/target metrics are controller estimates; exact bridge telemetry was off.\n'
            'The earlier batch averaged 16.3ms joint age; the later batch averaged 1.54ms.\n'
            'USB latency observations were made after the runs, not snapshotted during them.\n\n'
            'Videos are original JPEG-copy Matroska files, without reencoding. These runs\n'
            'predate the later one-second recording-padding change.\n'
            'manifest.json lists original paths, byte sizes and SHA-256 of all payload files.\n'
            'No source logs, videos, calibration or production settings were changed.\n')
        add_file('export_script.py',Path(__file__))
        archive.writestr(PREFIX+'manifest.json',json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
            episode_count=24,outcomes=counts,sources=sources,files=inventory),indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        bad=archive.testzip();assert bad is None,bad
        names=archive.namelist()
        assert sum(n.endswith('/video.mkv') for n in names)==24
        assert sum('/successes/' in n and n.endswith('/video.mkv') for n in names)==13
        assert sum('/failures/' in n and n.endswith('/video.mkv') for n in names)==11
    temporary.rename(DEST)
    result=dict(path=str(DEST),bytes=DEST.stat().st_size,sha256=sha(DEST),archive_members=len(names),
                videos=24,successes=13,drops=10,timeouts=1,crc_check='passed')
    DEST.with_suffix('.zip.verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
