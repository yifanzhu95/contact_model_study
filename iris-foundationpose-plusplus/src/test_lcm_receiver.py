#!/usr/bin/env python3
"""
LCM receiver test script to verify pose data transmission
"""
import lcm
import sys
import os
import numpy as np

# Add lcm_types to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lcm_types'))
from pose_types import Pose


def pose_handler(channel, data):
    """Handler function for incoming pose messages"""
    try:
        msg = Pose.decode(data)
        
        print(f"\n{'='*60}")
        print(f"Received pose on channel: {channel}")
        print(f"Timestamp: {msg.timestamp} microseconds ({msg.timestamp / 1e6:.3f} seconds)")
        
        # Reconstruct 4x4 matrix from flattened array
        pose_matrix = np.array(msg.pose_matrix, dtype=np.float32).reshape(4, 4)
        
        print(f"Pose matrix shape: {pose_matrix.shape}")
        print(f"Pose matrix dtype: {pose_matrix.dtype}")
        print(f"\nPose matrix:\n{pose_matrix}")
        
        # Extract translation and rotation
        translation = pose_matrix[:3, 3]
        rotation = pose_matrix[:3, :3]
        
        print(f"\nTranslation (x, y, z): {translation}")
        print(f"Rotation matrix:\n{rotation}")
        
        # Check for validity
        if not np.isfinite(pose_matrix).all():
            print("\n⚠️  WARNING: Pose contains NaN or Inf values!")
        else:
            print("\n✓ Pose is valid (all finite values)")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error decoding message: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to subscribe to LCM pose channel"""
    print("Starting LCM pose receiver...")
    print("Listening on channel: POSE")
    print("Press Ctrl+C to exit\n")
    
    lc = lcm.LCM()
    subscription = lc.subscribe("POSE", pose_handler)
    
    try:
        while True:
            lc.handle()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        lc.unsubscribe(subscription)


if __name__ == "__main__":
    main()
