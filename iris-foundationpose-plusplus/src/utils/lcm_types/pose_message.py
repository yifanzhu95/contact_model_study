"""
LCM message type for 4x4 pose matrix
"""
import lcm
import struct
import numpy as np

class PoseMessage:
    """
    LCM message for transmitting a 4x4 transformation matrix (pose)
    """
    __slots__ = ["timestamp", "pose_matrix"]
    
    def __init__(self):
        self.timestamp = 0.0  # seconds
        self.pose_matrix = np.eye(4, dtype=np.float32)  # 4x4 identity matrix
    
    def encode(self):
        """Encode pose message to bytes"""
        buf = bytearray()
        # Pack timestamp (double, 8 bytes)
        buf.extend(struct.pack('<d', self.timestamp))
        # Pack pose matrix (16 floats, 64 bytes)
        buf.extend(self.pose_matrix.astype(np.float32).tobytes())
        return bytes(buf)
    
    def decode(self, data):
        """Decode bytes to pose message"""
        if len(data) < 72:  # 8 bytes timestamp + 64 bytes pose matrix
            raise ValueError("Invalid data length for PoseMessage")
        
        offset = 0
        self.timestamp = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8
        self.pose_matrix = np.frombuffer(data[offset:offset+64], dtype=np.float32).reshape(4, 4)
    
    @staticmethod
    def get_hash():
        return 0x12345678  # Simple hash for message type
