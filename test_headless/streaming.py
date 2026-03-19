"""Utilities for streaming MuJoCo models and states between processes."""

from __future__ import annotations

import pickle
import socket
import struct
from pathlib import Path
from typing import Iterable, Optional

import mujoco
import numpy as np


_HEADER = struct.Struct("!I")


def _recvall(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed before all data received")
        data.extend(chunk)
    return bytes(data)


def send_packet(conn: socket.socket, payload) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(_HEADER.pack(len(data)))
    conn.sendall(data)


def recv_packet(sock: socket.socket):
    header = _recvall(sock, _HEADER.size)
    (length,) = _HEADER.unpack(header)
    payload = _recvall(sock, length)
    return pickle.loads(payload)


def pack_mjdata(
    mjd: mujoco.MjData,
    target_pos: Optional[np.ndarray] = None,
    target_quat: Optional[np.ndarray] = None,
    target_body_id: Optional[int] = None,
):
    return {
        "time": float(mjd.time),
        "qpos": np.array(mjd.qpos, copy=True),
        "qvel": np.array(mjd.qvel, copy=True),
        "act": np.array(mjd.act, copy=True),
        "ctrl": np.array(mjd.ctrl, copy=True),
        "xfrc_applied": np.array(mjd.xfrc_applied, copy=True),
        "target_pos": None if target_pos is None else np.array(target_pos, copy=True),
        "target_quat": None if target_quat is None else np.array(target_quat, copy=True),
        "target_body_id": target_body_id,
    }


def apply_mjdata(mjd: mujoco.MjData, state) -> None:
    mjd.time = state.get("time", mjd.time)
    if "qpos" in state:
        np.copyto(mjd.qpos, state["qpos"])
    if "qvel" in state:
        np.copyto(mjd.qvel, state["qvel"])
    if "act" in state:
        np.copyto(mjd.act, state["act"])
    if "ctrl" in state:
        np.copyto(mjd.ctrl, state["ctrl"])
    if "xfrc_applied" in state:
        np.copyto(mjd.xfrc_applied, state["xfrc_applied"])
    if "target_pos" in state and "target_body_id" in state and state["target_pos"] is not None:
        np.copyto(mjd.xpos[state["target_body_id"]], state["target_pos"])
    if "target_quat" in state and "target_body_id" in state and state["target_quat"] is not None:
        np.copyto(mjd.xquat[state["target_body_id"]], state["target_quat"])

    



class StreamServer:
    """Convenience wrapper that owns the TCP listener used by test-view-remote."""

    def __init__(self, model_path: Optional[str] = None, host: str = "127.0.0.1", port: int = 0):
        self._model_path = Path(model_path) if model_path is not None else None
        self.host = host
        self.port = port
        self._listener: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None

    @property
    def enabled(self) -> bool:
        return self.port > 0

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def start(self) -> None:
        """Backward-compatible alias for wait_for_connection()."""
        self.wait_for_connection()

    def wait_for_connection(self) -> None:
        """Block until a viewer connects when streaming is enabled."""
        if not self.enabled:
            return
        if self.connected:
            return
        if self._listener is None:
            self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._listener.bind((self.host, self.port))
            self._listener.listen(1)
        print(f"Waiting for local viewer on {self.host}:{self.port}...")
        self._conn, addr = self._listener.accept()
        print(f"Viewer connected from {addr}.")
        if self._model_path is not None:
            send_packet(self._conn, {"kind": "model_path", "path": str(self._model_path)})

    def send_state(
        self,
        mjd: mujoco.MjData,
        target_pos: Optional[np.ndarray] = None,
        target_quat: Optional[np.ndarray] = None,
        target_body_id: Optional[int] = None,
    ) -> None:
        if self._conn is None:
            return
        try:
            send_packet(self._conn, {"kind": "state", "state": pack_mjdata(mjd, target_pos, target_quat, target_body_id)})
        except OSError as exc:
            print(f"Viewer disconnected: {exc}")
            self.stop_connection()

    def stop_connection(self) -> None:
        if self._conn is not None:
            try:
                send_packet(self._conn, {"kind": "close"})
            except OSError:
                pass
            self._conn.close()
            self._conn = None
        if self._listener is not None:
            self._listener.close()
            self._listener = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_connection()


def make_file_packet(path: Path, data: Optional[bytes] = None, name: Optional[str] = None):
    file_path = Path(path)
    payload = {
        "kind": "file",
        "name": name or file_path.name,
        "data": data if data is not None else file_path.read_bytes(),
    }
    return payload


def send_file(sock: socket.socket, path: Path, name: Optional[str] = None) -> None:
    send_packet(sock, make_file_packet(path, name=name))


class UploadServer:
    """Simple helper that accepts a single upload connection and saves incoming files."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6100, out_dir: Path | str = "uploads"):
        self.host = host
        self.port = port
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def serve_once(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            print(f"Upload server listening on {self.host}:{self.port} -> {self.out_dir}")
            conn, addr = server.accept()
            print(f"Client connected from {addr}")
            with conn:
                while True:
                    try:
                        packet = recv_packet(conn)
                    except (ConnectionError, OSError) as exc:
                        print(f"Connection error: {exc}")
                        break
                    kind = packet.get("kind")
                    if kind == "close" or kind is None:
                        break
                    if kind != "file":
                        continue
                    self._save_packet(packet)

    def _save_packet(self, packet):
        name = packet.get("name", "upload.bin")
        data = packet.get("data", b"")
        target = self.out_dir / name
        target.write_bytes(data)
        print(f"Stored {len(data)} bytes at {target}")


def upload_files(host: str, port: int, files: Iterable[Path | str]):
    with socket.create_connection((host, port)) as sock:
        for file_name in files:
            payload = make_file_packet(Path(file_name))
            print(f"Uploading {payload['name']} ({len(payload['data'])} bytes)")
            send_packet(sock, payload)
        send_packet(sock, {"kind": "close"})
