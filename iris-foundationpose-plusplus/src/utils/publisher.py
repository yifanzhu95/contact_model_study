import zmq
import numpy as np
import cv2

class Publisher:
    def __init__(self, port=2223, endpoint=None):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(endpoint or f"tcp://*:{port}")


    def publish(self, pose, color, timestamp):
        pose_msg = np.asarray(pose, dtype=np.float32).reshape(4, 4).tobytes()
        color_msg = np.ascontiguousarray(color)
        encoded, buffer = cv2.imencode('.jpg', color_msg)
        if not encoded:
            raise ValueError('Could not encode supervision frame')
        timestamp_msg = np.array(timestamp, dtype=np.float64).tobytes()
        try:
            self._socket.send_multipart([pose_msg, buffer.tobytes(), timestamp_msg], flags=zmq.NOBLOCK)
        except zmq.Again:
            return False  # dropping a display frame must not stall pose tracking
        return True

    def close(self):
        self._socket.close(linger=0)
        self._context.term()

