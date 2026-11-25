"""Best-effort JPEG stream recording, isolated from vision and hardware control.

The host subscriber copies the JPEG bytes already published for supervision.
FFmpeg muxes them into Matroska with stream copy: no decoding, encoding or GPU.
An eight-frame queue keeps a stalled disk/muxer from accumulating old frames.
Only controller-confirmed episode intervals, including stationary padding, are written.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import struct
import subprocess
import threading
import time


@dataclass(frozen=True)
class JPEGFrame:
    sequence: int
    capture_timestamp: float
    received_timestamp: float
    jpeg: bytes


def offer_latest(pending, frame):
    """Never wait for the writer. Evict the oldest pending frame on overflow."""
    try:
        pending.put_nowait(frame)
        return 0
    except queue.Full:
        try:
            pending.get_nowait()
            dropped = 1
        except queue.Empty:
            dropped = 0  # The writer drained it between the two operations.
        pending.put_nowait(frame)  # There is exactly one producer.
        return dropped


def decode_message(parts, sequence, received_timestamp):
    """Validate framing without decoding the JPEG or importing NumPy/OpenCV."""
    if len(parts) != 3 or len(parts[0]) != 64 or len(parts[2]) != 8:
        raise ValueError('Expected 4x4 float32 pose, JPEG and float64 epoch timestamp')
    stamp = struct.unpack('=d', parts[2])[0]
    jpeg = parts[1]
    if not math.isfinite(stamp) or stamp <= 0:
        raise ValueError('Invalid camera timestamp')
    if len(jpeg) > 8*1024*1024 or not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9'):
        raise ValueError('Invalid or oversized JPEG')
    return JPEGFrame(sequence, stamp, received_timestamp, jpeg)


class CopyVideoWriter:
    """Wall-time-preserving CFR video made entirely from existing JPEG packets.

    Source times select output slots. Missing slots repeat the last JPEG;
    multiple arrivals in the same slot are subsampled. Exact capture times and
    repeated-frame flags are retained in CSV, including the source sequence.
    """
    def __init__(self, directory, fps, ffmpeg, stats, stem='camera'):
        self.directory = Path(directory)
        self.fps = fps
        self.ffmpeg = ffmpeg
        self.stats = stats
        self.process = None
        self.csv_stream = None
        self.error_stream = None
        self.csv = None
        self.origin = None
        self.previous = None
        self.slot = -1
        self.stem = stem

    def _start(self):
        csv_name = 'frames.csv' if self.stem == 'camera' else self.stem+'.frames.csv'
        error_name = 'ffmpeg.log' if self.stem == 'camera' else self.stem+'.ffmpeg.log'
        self.csv_stream = (self.directory/csv_name).open('x', newline='', buffering=262144)
        self.csv = csv.writer(self.csv_stream)
        self.csv.writerow(['frame_index', 'video_time_s', 'capture_timestamp',
                           'received_timestamp', 'source_sequence', 'repeated_frame'])
        self.error_stream = (self.directory/error_name).open('x')
        command = [self.ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-threads', '1', '-f', 'image2pipe', '-framerate', str(self.fps),
                   '-vcodec', 'mjpeg', '-i', 'pipe:0', '-map', '0:v:0',
                   '-c:v', 'copy', '-an', '-n', str(self.directory/(self.stem+'.mkv'))]
        # An independent group lets the parent finish the container cleanly on
        # launcher Ctrl-C. Killing the recorder also closes the sole pipe writer.
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self.error_stream, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))

    def _packet(self, frame, repeated):
        self.process.stdin.write(frame.jpeg)
        self.slot += 1
        self.csv.writerow([self.slot, f'{self.slot/self.fps:.9f}',
            f'{frame.capture_timestamp:.9f}', f'{frame.received_timestamp:.9f}',
            frame.sequence, int(repeated)])
        self.stats['video_frames'] += 1
        self.stats['repeated_video_frames'] += int(repeated)

    def write(self, frame):
        if self.origin is None:
            self.origin = frame.capture_timestamp
            self.stats['first_capture_timestamp'] = self.origin
            self._start()
        slot = int(math.floor((frame.capture_timestamp-self.origin)*self.fps + .5))
        if slot <= self.slot:
            self.stats['subsampled_frames'] += 1
            return
        if self.previous is not None and slot-self.slot > math.ceil(5*self.fps):
            raise ValueError('Camera timestamp gap exceeded 5 s; stopping recording instead of writing a large duplicate burst')
        while self.slot + 1 < slot:
            self._packet(self.previous, True)
        self._packet(frame, False)
        self.previous = frame
        self.stats['source_frames_written'] += 1
        self.stats['last_capture_timestamp'] = frame.capture_timestamp

    def abort(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()

    def close(self):
        error = None
        try:
            if self.process is not None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    code = self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1)
                    raise RuntimeError('Video muxer did not finish within 3 s')
                if code:
                    error = RuntimeError(f'Video muxer exited with code {code}; see ffmpeg.log')
        finally:
            for stream in (self.csv_stream, self.error_stream):
                if stream is not None:
                    stream.close()
        if error:
            raise error


def recording_stats(fps, capacity):
    return dict(status='waiting_for_frames', fps=fps, queue_capacity=capacity,
        codec='mjpeg', container='matroska', reencoding=False,
        frames_received=0, invalid_messages=0, queue_dropped_frames=0,
        source_frames_written=0, video_frames=0, repeated_video_frames=0,
        subsampled_frames=0, first_capture_timestamp=None, last_capture_timestamp=None,
        error=None)


class EpisodeVideoWriter:
    """Write confirmed episode intervals and optional stationary padding.

    A frame newer than the controller's active-time watermark waits in bounded
    memory, never on disk. End timestamps supersede the watermark, so holds and
    unconfirmed holds/resets cannot leak in even if telemetry is delayed or the controller
    crashes. Each episode has an independent muxer and time origin.
    """
    def __init__(self, directory, fps, ffmpeg, stats, windows_file):
        self.directory, self.fps, self.ffmpeg = Path(directory), fps, ffmpeg
        self.stats, self.windows_file = stats, Path(windows_file)
        self.windows, self.version = [], None
        self.current = self.current_window = None
        self.closed = set()
        stats.update(mode='episodes_only', episodes=[], outside_episode_frames=0,
                     unconfirmed_frames_dropped=0, status='waiting_for_episode')

    @property
    def process(self):
        return None if self.current is None else self.current.process

    def refresh(self):
        try:
            info = self.windows_file.stat()
            version = (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)
            if version == self.version:
                return
            data = json.loads(self.windows_file.read_text())
            windows = []
            for entry in data.get('episode_windows', []):
                start = datetime.fromisoformat(entry.get('recording_started_at_utc', entry['started_at_utc'])).timestamp()
                end_stamp = entry.get('recording_finished_at_utc', entry.get('finished_at_utc'))
                end = datetime.fromisoformat(end_stamp).timestamp() if end_stamp else None
                through = end if end is not None else float(entry['confirmed_through'])
                if not all(math.isfinite(v) for v in (start, through)) or through < start:
                    raise ValueError('Invalid confirmed episode interval')
                windows.append(dict(entry, start=start, end=end, through=through))
            self.windows, self.version = windows, version
        except (OSError, ValueError, KeyError, TypeError):
            return  # Fail closed: an old watermark permits no new frames.

    def _aggregate(self):
        for key in ['source_frames_written', 'video_frames', 'repeated_video_frames', 'subsampled_frames']:
            self.stats[key] = sum(e[key] for e in self.stats['episodes'])
        starts = [e['first_capture_timestamp'] for e in self.stats['episodes'] if e['first_capture_timestamp'] is not None]
        ends = [e['last_capture_timestamp'] for e in self.stats['episodes'] if e['last_capture_timestamp'] is not None]
        self.stats['first_capture_timestamp'] = min(starts) if starts else None
        self.stats['last_capture_timestamp'] = max(ends) if ends else None

    def _finish_current(self):
        if self.current is None:
            return
        writer = self.current
        window = next((w for w in self.windows if w['episode']==self.current_window['episode']), self.current_window)
        try:
            writer.close()
            writer.stats.update(status='complete' if window['end'] is not None else 'incomplete',
                started_at_utc=window['started_at_utc'],
                finished_at_utc=window.get('finished_at_utc'), outcome=window['outcome'],
                video_duration_s=writer.stats['video_frames']/self.fps)
            for key in ('recording_started_at_utc', 'recording_finished_at_utc'):
                if key in window:writer.stats[key]=window[key]
        except Exception as exc:
            writer.stats.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            self.closed.add(window['episode'])
            self.current = self.current_window = None
            self._aggregate()

    def idle(self):
        """Finish a closed clip even if tracking stopped sending new frames."""
        self.check_muxer()
        self.refresh()
        if self.current is not None:
            window = next((w for w in self.windows if w['episode']==self.current_window['episode']), None)
            if window and window['end'] is not None and time.time() > window['end']+.5:
                self._finish_current()

    def write(self, frame):
        self.check_muxer()
        self.refresh()
        for window in self.windows:
            if frame.capture_timestamp < window['start']:
                break
            if window['end'] is not None and frame.capture_timestamp >= window['end']:
                continue
            if window['episode'] in self.closed:
                break
            planned_time = frame.capture_timestamp
            if self.current is not None and self.current_window['episode']==window['episode']:
                planned_time = self.current.origin + math.floor((frame.capture_timestamp-self.current.origin)*self.fps+.5)/self.fps
            if window['end'] is None and max(frame.capture_timestamp,planned_time) > window['through']:
                return None  # Controller has not yet confirmed this time as active.
            if window['end'] is not None and planned_time >= window['end']:
                break
            if self.current is None or self.current_window['episode'] != window['episode']:
                self._finish_current()
                stem=f'episode_{window["episode"]+1:02d}'
                stats=recording_stats(self.fps,self.stats['queue_capacity'])
                stats.update(episode=window['episode']+1,status='recording',
                    started_at_utc=window['started_at_utc'],outcome=window['outcome'],
                    video=stem+'.mkv',timestamps=stem+'.frames.csv')
                self.stats['episodes'].append(stats)
                self.current_window=window
                self.current=CopyVideoWriter(self.directory,self.fps,self.ffmpeg,stats,stem)
            self.current.write(frame)
            self._aggregate()
            return True
        if self.current is not None:
            window=next((w for w in self.windows if w['episode']==self.current_window['episode']),None)
            if window and window['end'] is not None and frame.capture_timestamp >= window['end']:
                self._finish_current()
        self.stats['outside_episode_frames'] += 1
        return False

    def check_muxer(self):
        # Called only by the writer thread, so an intentional per-episode close
        # cannot race with an "early exit" check in the receiver thread.
        process=self.process
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'Video muxer exited early with code {process.returncode}; see episode FFmpeg log')

    def abort(self):
        if self.current is not None:
            self.current.abort()

    def close(self):
        self.refresh()
        self._finish_current()


def save_status(directory, stats):
    temporary = directory/'recording.json.tmp'
    temporary.write_text(json.dumps(dict(stats), indent=2, allow_nan=False)+'\n')
    temporary.replace(directory/'recording.json')


def record_stream(endpoint, directory, fps=30, capacity=8, ffmpeg='ffmpeg', windows_file=None):
    import zmq  # Only the optional recorder needs this host dependency.
    if windows_file is None:
        raise ValueError('Episode window telemetry is required; whole-run recording is disabled')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    # Lower priority applies to both the receiver/writer and the muxer child.
    try:
        os.nice(10)
    except OSError:
        pass
    stats = recording_stats(fps, capacity)
    stats.update(endpoint=endpoint, started_at_unix=time.time())
    save_status(directory, stats)
    pending = queue.Queue(maxsize=capacity)
    stop = threading.Event()
    writer_failed = threading.Event()
    writer = EpisodeVideoWriter(directory, fps, ffmpeg, stats, windows_file)

    def write_frames():
        frame = None
        try:
            while frame is not None or not stop.is_set() or not pending.empty():
                if frame is None:
                    try:
                        frame = pending.get(timeout=.05)
                    except queue.Empty:
                        writer.idle()
                        stats['status']='recording' if writer.current else 'waiting_for_episode'
                        continue
                result=writer.write(frame)
                if result is None:
                    if stop.is_set():
                        stats['unconfirmed_frames_dropped'] += 1
                        frame=None
                    else:
                        stop.wait(.01)
                    continue
                stats['status']='recording' if writer.current else 'waiting_for_episode'
                frame=None
        except Exception as exc:
            stats['error'] = f'{type(exc).__name__}: {exc}'
            writer_failed.set()
        finally:
            try:
                writer.close()
            except Exception as exc:
                stats['error'] = stats['error'] or f'{type(exc).__name__}: {exc}'
                writer_failed.set()

    thread = threading.Thread(target=write_frames, name='jpeg-copy-writer', daemon=True)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.setsockopt(zmq.MAXMSGSIZE, 9*1024*1024)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.connect(endpoint)
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    thread.start()
    last_status = time.monotonic()
    try:
        while not stop.is_set() and not writer_failed.is_set():
            if socket.poll(50, zmq.POLLIN):
                parts = socket.recv_multipart()
                try:
                    frame = decode_message(parts, stats['frames_received'], time.time())
                except ValueError:
                    stats['invalid_messages'] += 1
                    continue
                stats['frames_received'] += 1
                stats['queue_dropped_frames'] += offer_latest(pending, frame)
            if time.monotonic()-last_status >= 1:
                save_status(directory, stats)
                last_status = time.monotonic()
    finally:
        stop.set()
        socket.close()
        context.term()
        thread.join(timeout=4)
        if thread.is_alive():
            stats['error'] = stats['error'] or 'Recorder flush exceeded 4 s; muxer stopped'
            writer_failed.set()
            writer.abort()
            thread.join(timeout=2)
        stats['status'] = 'failed' if writer_failed.is_set() else ('complete' if stats['video_frames'] else 'no_frames')
        stats['finished_at_unix'] = time.time()
        stats['video_duration_s'] = stats['video_frames']/fps
        save_status(directory, stats)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(stats), flush=True)
    return int(writer_failed.is_set())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--queue-size', type=int, default=8)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--episode-windows-file', type=Path, required=True,
                        help='controller goal telemetry containing confirmed episode windows')
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or not 0 < args.fps <= 120:
        parser.error('--fps must be finite and in (0, 120]')
    if not 1 <= args.queue_size <= 64:
        parser.error('--queue-size must be in [1, 64]')
    executable = shutil.which(args.ffmpeg)
    if executable is None:
        parser.error('FFmpeg executable not found; install ffmpeg to enable video recording')
    return record_stream(args.endpoint, args.output_dir, args.fps, args.queue_size, executable,
                         args.episode_windows_file)
