"""Optional telemetry: nonblocking local IPC to a separate, low-priority writer."""
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys
import time

import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported telemetry type: {type(value).__name__}')


def _writer(fd, path):
    """Only this process formats JSON or performs recording I/O during control.

    Pickle input comes exclusively from our private inherited socketpair; no
    file or externally connectable socket is accepted as a pickle source.
    """
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET, fileno=fd)
    # The bridge closes the private socket after torque cleanup. Group signals
    # must not abort the recorder before it has drained already queued samples.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    written = 0
    error = None
    try:
        os.nice(10)
        with Path(path).open('a', buffering=1024*1024) as stream:
            connection.send(b'READY')
            last_flush = time.monotonic()
            while True:
                packet = connection.recv(1024*1024)
                if not packet:
                    break
                record = pickle.loads(packet)
                stream.write(json.dumps(record, default=_json_value,
                    separators=(',', ':'), allow_nan=False)+'\n')
                written += 1
                if time.monotonic()-last_flush >= .5:
                    stream.flush()
                    last_flush = time.monotonic()
            stream.write(json.dumps(dict(event='end', timestamp_us=int(time.time()*1e6),
                written=written))+'\n')
            stream.flush()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        connection.close()
        try:
            Path(path).with_suffix('.status.json').write_text(json.dumps(
                dict(written=written, error=error, writer_finished=True))+'\n')
        except OSError:
            pass
    return int(error is not None)


class BridgeTelemetry:
    """Drop recording samples rather than wait for a recorder or its disk.

    The control process has no recording thread, JSON encoder, disk writer or
    multiprocessing.Queue feeder. A bounded, nonblocking SOCK_SEQPACKET socket
    preserves record boundaries and reports recorder backpressure explicitly.
    """

    def __init__(self, path, metadata=None, capacity=1024):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.error = None
        self.accepted = self.written = self.dropped = 0
        self.closed = False
        # Fail on an invalid output path before the bridge enables torque.
        with self.path.open('a') as stream:
            stream.write(json.dumps(dict(event='start', schema_version=2,
                timestamp_us=int(time.time()*1e6), metadata=metadata or {},
                recorder='separate_process_nonblocking_ipc'))+'\n')
        producer, consumer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.socket = producer
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            try:
                producer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                                    max(4096, min(int(capacity)*4096, 4*1024*1024)))
            except OSError:
                # Restricted hosts may forbid tuning socket buffers. The
                # kernel default is still bounded and submit() still drops.
                pass
            self.path.with_suffix('.status.json').unlink(missing_ok=True)
            self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--writer', str(consumer.fileno()), str(self.path)], pass_fds=(consumer.fileno(),),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                env=environment, close_fds=True)
            consumer.close()
            # Initialization happens before serial connection/torque enable.
            producer.settimeout(10)
            if producer.recv(64) != b'READY':
                raise RuntimeError('Telemetry recorder did not become ready')
            producer.setblocking(False)
        except BaseException:
            producer.close()
            consumer.close()
            if hasattr(self, 'process') and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1)
            raise

    def submit(self, record):
        if self.error is not None or self.closed:
            self.dropped += 1
            return
        try:
            payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
            self.socket.send(payload)  # nonblocking: never wait for disk/collector
            self.accepted += 1
        except BlockingIOError:
            self.dropped += 1
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.dropped += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Called after torque cleanup; EOF lets the collector drain queued data.
        try:
            self.socket.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        self.socket.close()
        finished = False
        try:
            self.process.wait(timeout=5)
            finished = True
        except subprocess.TimeoutExpired:
            self.error = self.error or 'Recorder did not drain within 5 s'
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        status_path = self.path.with_suffix('.status.json')
        try:
            recorded = json.loads(status_path.read_text())
            self.written = recorded.get('written', 0)
            self.error = self.error or recorded.get('error')
        except (OSError, ValueError):
            self.error = self.error or 'Recorder status unavailable'
        if self.process.returncode:
            self.error = self.error or f'Recorder exited with code {self.process.returncode}'
        status = dict(written=self.written, accepted=self.accepted, dropped=self.dropped,
            error=self.error, writer_finished=finished)
        try:
            status_path.write_text(json.dumps(status, indent=2)+'\n')
        except OSError:
            pass


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--writer':
        raise SystemExit('Private telemetry collector; launch through the bridge.')
    raise SystemExit(_writer(int(sys.argv[2]), sys.argv[3]))
