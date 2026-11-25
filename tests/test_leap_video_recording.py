"""CPU-only recording checks: compressed packet identity, timing and isolation."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

import pytest

from ContactModelStudy.Hardware.VideoRecording import (
    JPEGFrame, CopyVideoWriter, EpisodeVideoWriter, decode_message, offer_latest, recording_stats)

ROOT = Path(__file__).resolve().parents[1]


def jpeg(value):
    import cv2
    import numpy as np
    ok, encoded = cv2.imencode('.jpg', np.full((48,64,3), value, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def require_ffmpeg():
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('FFmpeg/ffprobe not installed')
    return ffmpeg, ffprobe


def test_receive_queue_is_bounded_and_preserves_newest_frames():
    pending = queue.Queue(maxsize=2)
    assert offer_latest(pending, 1) == 0
    assert offer_latest(pending, 2) == 0
    assert offer_latest(pending, 3) == 1
    assert pending.qsize() == 2
    assert [pending.get_nowait(), pending.get_nowait()] == [2,3]


def test_message_validation_preserves_compressed_bytes_and_timestamp():
    packet = jpeg(70)
    frame = decode_message([b'\0'*64, packet, struct.pack('=d', 100.25)], 4, 101)
    assert frame.jpeg is packet
    assert frame.capture_timestamp == 100.25 and frame.received_timestamp == 101
    assert frame.sequence == 4
    for bad in [[b'\0'*64, packet], [b'',packet,struct.pack('=d',100)],
                [b'\0'*64,b'broken',struct.pack('=d',100)],
                [b'\0'*64,packet,struct.pack('=d',float('nan'))]]:
        with pytest.raises(ValueError):
            decode_message(bad, 0, 0)


def test_video_packets_are_unchanged_and_missing_slots_repeat_last_jpeg(tmp_path):
    ffmpeg,ffprobe = require_ffmpeg()
    packets = [jpeg(v) for v in [10,120,230]]
    stats = recording_stats(10,8)
    writer = CopyVideoWriter(tmp_path,10,ffmpeg,stats)
    try:
        for i,(stamp,packet) in enumerate(zip([100.,100.1,100.3],packets)):
            writer.write(JPEGFrame(i,stamp,stamp+.01,packet))
    finally:
        writer.close()
    probe = subprocess.run([ffprobe,'-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name:packet=pts_time,data_hash','-show_packets',
        '-show_data_hash','sha256','-of','json',str(tmp_path/'camera.mkv')],
        capture_output=True,text=True,check=True)
    video = json.loads(probe.stdout)
    assert video['streams'][0]['codec_name'] == 'mjpeg'
    expected = [packets[0],packets[1],packets[1],packets[2]]
    assert [p['data_hash'].split(':')[1].lower() for p in video['packets']] == [hashlib.sha256(p).hexdigest() for p in expected]
    assert [float(p['pts_time']) for p in video['packets']] == pytest.approx([0,.1,.2,.3],abs=.001)
    rows = list(csv.DictReader((tmp_path/'frames.csv').open()))
    assert [r['repeated_frame'] for r in rows] == ['0','0','1','0']
    assert [float(r['capture_timestamp']) for r in rows] == [100,100.1,100.1,100.3]
    assert stats['video_frames'] == 4 and stats['repeated_video_frames'] == 1


def test_resampling_and_clock_gaps_never_create_unbounded_work(tmp_path):
    ffmpeg,_ = require_ffmpeg()
    stats = recording_stats(10,8)
    writer = CopyVideoWriter(tmp_path,10,ffmpeg,stats)
    packet = jpeg(30)
    try:
        writer.write(JPEGFrame(0,100,100,packet))
        writer.write(JPEGFrame(1,100.01,100,packet))
        assert stats['subsampled_frames'] == 1
        with pytest.raises(ValueError,match='gap exceeded'):
            writer.write(JPEGFrame(2,10000,100,packet))
        assert stats['video_frames'] == 1
    finally:
        writer.close()


@pytest.fixture
def jpeg_publisher():
    require_ffmpeg()
    import zmq
    with tempfile.TemporaryDirectory(prefix='video-ipc-') as ipc:
        context = zmq.Context()
        publisher = context.socket(zmq.XPUB)
        try:
            publisher.setsockopt(zmq.LINGER,0)
            endpoint = 'ipc://'+str(Path(ipc)/'stream')
            publisher.bind(endpoint)
            yield publisher, endpoint
        finally:
            publisher.close(linger=0)
            context.term()


def test_live_subscriber_stops_cleanly_and_muxer_failure_is_reported(tmp_path,jpeg_publisher):
    ffmpeg,ffprobe = require_ffmpeg()
    import zmq
    publisher,endpoint=jpeg_publisher
    for index, executable in enumerate([ffmpeg, '/bin/false']):
        output = tmp_path/f'video{index}'
        windows=tmp_path/f'windows{index}.json'
        windows.write_text(json.dumps(dict(episode_windows=[dict(episode=0,
            started_at_utc=datetime.fromtimestamp(100,timezone.utc).isoformat(),
            finished_at_utc=None,confirmed_through=200,outcome='running')])))
        process = subprocess.Popen([sys.executable,str(ROOT/'record_leap_video.py'),
            '--endpoint',endpoint,'--output-dir',str(output),'--fps','10','--ffmpeg',executable,
            '--episode-windows-file',str(windows)],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        try:
            # XPUB subscription notification avoids a slow-joiner sleep.
            deadline = time.monotonic()+6
            while time.monotonic()<deadline:
                if publisher.poll(100,zmq.POLLIN) and publisher.recv()==b'\x01':break
            else:pytest.fail('Recorder did not subscribe')
            packet=jpeg(80)
            for i in range(12):
                publisher.send_multipart([b'\0'*64,packet,struct.pack('=d',100+i*.1)])
                time.sleep(.005)
            status = {}
            while time.monotonic()<deadline:
                try:status=json.loads((output/'recording.json').read_text())
                except (OSError,ValueError):pass
                if (index==0 and status.get('source_frames_written',0)>0) or (index==1 and process.poll() is not None):break
                time.sleep(.02)
            if index==0:
                assert process.poll() is None
                process.send_signal(signal.SIGINT)
            process.communicate(timeout=8)
            status=json.loads((output/'recording.json').read_text())
            if index==0:
                assert process.returncode == 0
                assert status['status']=='complete' and status['video_frames']>0
                assert not (output/'camera.mkv').exists()
                subprocess.run([ffprobe,'-v','error',str(output/'episode_01.mkv')],check=True,capture_output=True)
            else:
                assert process.returncode == 1
                assert status['status']=='failed' and status['error']
        finally:
            if process.poll() is None:
                process.kill();process.communicate(timeout=5)


def test_episode_gate_excludes_preparation_holds_and_unconfirmed_times(tmp_path):
    ffmpeg,ffprobe=require_ffmpeg()
    windows=tmp_path/'goal.json'
    stats=recording_stats(10,8)
    writer=EpisodeVideoWriter(tmp_path,10,ffmpeg,stats,windows)
    image=jpeg(50)
    def frame(stamp):return JPEGFrame(int(stamp*100),stamp,stamp+.05,image)
    def window(episode,start,end,through=None,outcome='running'):
        return dict(episode=episode,started_at_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),
            finished_at_utc=None if end is None else datetime.fromtimestamp(end,timezone.utc).isoformat(),
            confirmed_through=through,outcome=outcome)
    def publish(entries):
        pending=windows.with_suffix('.tmp');pending.write_text(json.dumps(dict(episode_windows=entries)));pending.replace(windows)
    try:
        assert writer.write(frame(99)) is False  # No episode yet.
        publish([window(0,100,None,100.1)])
        assert writer.write(frame(99.9)) is False
        assert writer.write(frame(100)) is True
        assert writer.write(frame(100.2)) is None  # Not confirmed; nothing written yet.
        assert stats['video_frames']==1
        publish([window(0,100,None,100.3)])
        assert writer.write(frame(100.2)) is True
        publish([window(0,100,100.25,outcome='success')])
        assert writer.write(frame(100.3)) is False  # End excludes the success hold.
        assert stats['episodes'][0]['status']=='complete'
        assert writer.write(frame(110)) is False
        publish([window(0,100,100.25,outcome='success'),window(1,120,None,120.2)])
        assert writer.write(frame(119.9)) is False  # Next preparation.
        assert writer.write(frame(120)) is True  # New file; no 20 s gap duplication.
        publish([window(0,100,100.25,outcome='success'),window(1,120,120.1,outcome='interrupted')])
        assert writer.write(frame(120.2)) is False  # Exclude Ctrl-C return to zero.
    finally:
        writer.close()
    assert len(stats['episodes'])==2 and stats['video_frames']==4
    assert not (tmp_path/'camera.mkv').exists()
    for index,(start,end) in enumerate([(100,100.25),(120,120.1)],1):
        rows=list(csv.DictReader((tmp_path/f'episode_{index:02d}.frames.csv').open()))
        assert all(start<=float(r['capture_timestamp'])<end for r in rows)
        origin=float(rows[0]['capture_timestamp'])
        assert all(start<=origin+float(r['video_time_s'])<end for r in rows)
        subprocess.run([ffprobe,'-v','error',str(tmp_path/f'episode_{index:02d}.mkv')],check=True,capture_output=True)


def test_padded_episode_records_lead_and_tail_without_changing_control_timestamps(tmp_path):
    ffmpeg,_=require_ffmpeg();windows=tmp_path/'goal.json'
    stats=recording_stats(10,8);writer=EpisodeVideoWriter(tmp_path,10,ffmpeg,stats,windows)
    stamp=lambda t:datetime.fromtimestamp(t,timezone.utc).isoformat()
    entry=dict(episode=0,started_at_utc=stamp(101),finished_at_utc=None,
        recording_started_at_utc=stamp(100),recording_finished_at_utc=None,confirmed_through=100.2,outcome='running')
    def publish():
        pending=windows.with_suffix('.tmp');pending.write_text(json.dumps(dict(episode_windows=[entry])));pending.replace(windows)
    packet=jpeg(70)
    def frame(t):return JPEGFrame(int(t*100),t,t+.01,packet)
    try:
        publish()
        assert writer.write(frame(99.9)) is False
        assert writer.write(frame(100.1)) is True  # Before control starts.
        assert writer.write(frame(100.5)) is None  # No future permission.
        entry['confirmed_through']=101.5;publish()
        assert writer.write(frame(101.2)) is True
        entry.update(finished_at_utc=stamp(102),outcome='success',confirmed_through=102.2);publish()
        assert writer.write(frame(102.1)) is True  # After control ends, while holding.
        assert writer.write(frame(102.3)) is None
        entry.update(recording_finished_at_utc=stamp(103),confirmed_through=None);publish()
        assert writer.write(frame(102.9)) is True
        assert writer.write(frame(103.1)) is False
    finally:writer.close()
    episode=stats['episodes'][0]
    assert episode['status']=='complete'
    assert episode['started_at_utc']==stamp(101) and episode['finished_at_utc']==stamp(102)
    assert episode['recording_started_at_utc']==stamp(100) and episode['recording_finished_at_utc']==stamp(103)
    rows=list(csv.DictReader((tmp_path/'episode_01.frames.csv').open()))
    assert min(float(r['capture_timestamp']) for r in rows)<101
    assert max(float(r['capture_timestamp']) for r in rows)>102
