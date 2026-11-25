"""
LCM Publisher for sending pose data
"""
import lcm
import numpy as np
import sys
import os
import time

# Add lcm_types to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lcm_types'))
from pose_types import Pose


class LCMPublisher:
    """
    Publisher class for sending 4x4 pose matrix via LCM
    """
    
    def __init__(self, channel_name="POSE", provider=None):
        """
        Initialize LCM publisher
        
        Args:
            channel_name: LCM channel name to publish to
            provider: LCM provider URL (e.g., "udpm://239.255.76.67:7667?ttl=1")
                     If None, uses default provider
        """
        try:
            self.lcm_obj = lcm.LCM(provider) if provider else lcm.LCM()
            self.channel_name = channel_name
            self.is_connected = True
        except Exception as e:
            print(f"Failed to initialize LCM: {e}")
            self.is_connected = False
    
    def publish_pose(self, pose: np.ndarray, timestamp: float = None):
        """
        Publish a 4x4 pose matrix to the LCM channel
        
        Args:
            pose: 4x4 numpy array (transformation matrix)
            timestamp: timestamp in seconds (uses current time if None)
        """
        if not self.is_connected:
            return False
        
        try:
            # Handle torch tensors
            if hasattr(pose, 'cpu'):
                pose = pose.cpu().numpy()
            
            # Ensure it's a numpy array and properly shaped
            pose_np = np.asarray(pose, dtype=np.float64)
            
            # Handle batched tensors (e.g., [1, 4, 4])
            if pose_np.ndim == 3:
                pose_np = pose_np[0]
            
            # Ensure 4x4 shape
            pose_4x4 = pose_np.reshape(4, 4).astype(np.float32)
            
            # Verify the matrix is valid (no NaN or Inf)
            if not np.isfinite(pose_4x4).all():
                print(f"Warning: Invalid pose matrix contains NaN or Inf")
                return False
            
            # Create LCM message
            msg = Pose()
            
            # Convert timestamp to microseconds (int64)
            if timestamp is not None:
                msg.timestamp = int(timestamp * 1e6)
            else:
                msg.timestamp = int(time.time() * 1e6)
            
            # Flatten 4x4 matrix to list of 16 floats in row-major order
            msg.pose_matrix = pose_4x4.flatten().tolist()
            
            # Publish the encoded message
            self.lcm_obj.publish(self.channel_name, msg.encode())
            return True
        except Exception as e:
            print(f"Error publishing pose: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close(self):
        """Close the LCM connection"""
        if self.is_connected:
            try:
                self.lcm_obj.destroy()
                self.is_connected = False
            except:
                pass
