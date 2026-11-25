#!/usr/bin/env python3
"""Split a recorded LEAP run into manipulation-only episodes, without re-encoding.

Both original camera capture times and video presentation times must fall in
the logged [episode start, episode finish) interval. Every output JPEG packet
is checked against its source hash. Originals are never modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def probe_packets(path, ffprobe):
    result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_packets', '-show_data_hash', 'sha256',
        '-show_entries', 'stream=codec_name:packet=pts_time,data_hash,flags',
        '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    if len(value.get('streams', [])) != 1 or value['streams'][0]['codec_name'] != 'mjpeg':
        raise ValueError('This lossless splitter requires the recorder\'s MJPEG video')
    packets = value['packets']
    if not packets or any('K' not in packet['flags'] for packet in packets):
        raise ValueError('Expected independently decodable JPEG packets')
    return packets


def epoch(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError('Episode timestamps must include a time zone')
    return stamp.timestamp()


def selected_frames(frames, origin, start, finish):
    result = [i for i, frame in enumerate(frames)
        if start <= float(frame['capture_timestamp']) < finish
        and start <= origin+float(frame['video_time_s']) < finish]
    if result and result != list(range(result[0], result[-1]+1)):
        raise ValueError('Episode frames are not contiguous; inspect camera timestamps')
    return result


def split_run(run, output=None):
    run = Path(run).resolve()
    source = run/'video/camera.mkv'
    if not source.is_file():
        raise ValueError('No whole-run camera.mkv exists. New recordings already save episode_XX.mkv clips under video/.')
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise ValueError('FFmpeg and ffprobe must be installed')
    controller = json.loads((run/'controller.json').read_text())
    recording = json.loads((run/'video/recording.json').read_text())
    if recording['status'] != 'complete':
        raise ValueError('Recording must be finished before splitting')
    fps = float(recording['fps'])
    origin = float(recording['first_capture_timestamp'])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid recorded frame rate')
    with (run/'video/frames.csv').open(newline='') as stream:
        frames = list(csv.DictReader(stream))
    if [int(row['frame_index']) for row in frames] != list(range(len(frames))):
        raise ValueError('Source frame CSV is not contiguous')
    packets = probe_packets(source, ffprobe)
    if len(frames) != len(packets) or len(frames) != recording['video_frames']:
        raise ValueError('Video packets and frame timestamp log do not match')
    plans, skipped = [], []
    for episode in controller['episodes']:
        number = episode['episode']+1
        if not episode.get('completed') or not episode.get('finished_at_utc'):
            skipped.append(dict(episode=number, reason='incomplete or missing finish timestamp'))
            continue
        start, finish = epoch(episode['started_at_utc']), epoch(episode['finished_at_utc'])
        if finish <= start:
            raise ValueError(f'Episode {number} has an invalid time interval')
        indices = selected_frames(frames, origin, start, finish)
        if not indices:
            skipped.append(dict(episode=number, reason='no captured frames inside manipulation interval'))
            continue
        plans.append((episode, start, finish, indices))
    if not plans:
        raise ValueError('No completed episodes overlap the recording')
    output = Path(output).resolve() if output else run/'video/episodes'
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(source_video=str(source), source_controller=str(run/'controller.json'),
        source_video_bytes=source.stat().st_size, source_video_frames=len(frames), fps=fps,
        source_modified=False, reencoded=False, packet_hash_verification='SHA256 for every output packet',
        selection='capture time and original presentation time both within [episode start, episode finish)',
        skipped=skipped, episodes=[])
    fields = ['clip_frame_index', 'clip_time_s', 'source_video_frame_index', *frames[0].keys()]
    for episode, start, finish, indices in plans:
        number, outcome = episode['episode']+1, episode['finish_reason']
        name = f'episode_{number:02d}_{outcome}'
        target = output/(name+'.mkv')
        temporary = output/(name+'.partial.mkv')
        # Output-side seeking discards packets before the exact requested PTS,
        # including packets from an earlier Matroska seek cluster. MJPEG frames
        # are all keyframes; frame-count and hash checks catch boundary errors.
        command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-threads', '1', '-i', str(source), '-ss', packets[indices[0]]['pts_time'],
            '-map', '0:v:0', '-frames:v', str(len(indices)), '-c:v', 'copy',
            '-an', '-avoid_negative_ts', 'make_zero', '-n', str(temporary)]
        subprocess.run(command, check=True)
        clip_packets = probe_packets(temporary, ffprobe)
        actual = [p['data_hash'] for p in clip_packets]
        expected = [packets[i]['data_hash'] for i in indices]
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f'Episode {number}: output frames do not exactly match selected source packets')
        temporary.rename(target)
        with (output/(name+'.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for j, (index, packet) in enumerate(zip(indices, clip_packets)):
                writer.writerow(dict(clip_frame_index=j, clip_time_s=packet['pts_time'],
                    source_video_frame_index=index, **frames[index]))
        entry = dict(episode=number, outcome=outcome, video=target.name, timestamps=name+'.csv',
            episode_started_at_utc=episode['started_at_utc'], episode_finished_at_utc=episode['finished_at_utc'],
            manipulation_duration_s=finish-start, clip_duration_s=len(indices)/fps,
            first_source_video_frame=indices[0], last_source_video_frame=indices[-1],
            video_frames=len(indices), file_bytes=target.stat().st_size,
            first_capture_timestamp=float(frames[indices[0]]['capture_timestamp']),
            last_capture_timestamp=float(frames[indices[-1]]['capture_timestamp']),
            all_packet_hashes_match=True)
        manifest['episodes'].append(entry)
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'Episode {number:02d}: {outcome:7s} {len(indices)/fps:7.3f} s, {len(indices)} frames; verified', flush=True)
    print(f'Saved {len(manifest["episodes"])} clips: {output}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    split_run(args.run_directory, args.output_dir)


if __name__ == '__main__':
    main()
