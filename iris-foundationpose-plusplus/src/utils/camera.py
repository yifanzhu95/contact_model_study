import os
from pathlib import Path
import sys
import pyrealsense2 as rs
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ContactModelStudy.Hardware.Camera import verify_camera_profile

# <<< EDIT THIS LINE FOR YOUR CAMERA, OR LEAVE AS None TO USE FIRST DEVICE >>>
DEFAULT_REAL_SENSE_SERIAL = "310522070297"  # or None

def discover_camera_devices():
    """Discover and list all available RealSense camera devices"""
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("No RealSense devices found!")
            return []
        
        device_list = []
        print(f"Found {len(devices)} RealSense device(s):")
        for i, device in enumerate(devices):
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            product_line = device.get_info(rs.camera_info.product_line)
            device_info = {
                'index': i,
                'serial': serial,
                'name': name,
                'product_line': product_line
            }
            device_list.append(device_info)
            print(f"  Device {i}: {name}")
            print(f"    Serial: {serial}")
            print(f"    Product: {product_line}")
            print()
        
        return device_list
    except Exception as e:
        print(f"Error discovering devices: {e}")
        return []


class Camera:

    def __init__(self, device_id=None, width=640, height=480, fps=60, calibration=None) -> None:
        self.pipeline = rs.pipeline()
        config = rs.config()

        # 1) if device_id was passed explicitly, use it
        # 2) else, use env var REALSENSE_SERIAL if set
        # 3) else, use DEFAULT_REAL_SENSE_SERIAL if not None
        # 4) else, let RealSense pick the first device
        serial = (
            device_id
            or os.environ.get("REALSENSE_SERIAL")
            or DEFAULT_REAL_SENSE_SERIAL
        )

        if serial is not None:
            print(f"[camera] Using device with serial: {serial}")
            config.enable_device(serial)
        else:
            print("[camera] No serial specified, using first available device")

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        device_serial = device.get_info(rs.camera_info.serial_number)
        print(f"[camera] Final device: serial={device_serial}, product_line={device_product_line}")

        # Streams
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = self.pipeline.start(config)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        K = [[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]]
        device = profile.get_device()
        self.calibration = dict(camera_intrinsics=K, camera_serial=device_serial,
            image_size=[intrinsics.width, intrinsics.height], camera_fps=stream.fps(),
            realsense=dict(name=device.get_info(rs.camera_info.name),
                firmware=device.get_info(rs.camera_info.firmware_version),
                color_intrinsics=K, distortion_model=str(intrinsics.model),
                distortion_coefficients=list(intrinsics.coeffs),
                depth_scale_m=device.first_depth_sensor().get_depth_scale(), color_format=str(stream.format())))
        try:
            verify_camera_profile(self.calibration if calibration is None else calibration, self.calibration)
        except Exception:
            self.pipeline.stop()
            raise

        color_sensor = self.pipeline.get_active_profile().get_device().first_color_sensor()

        if color_sensor.supports(rs.option.enable_auto_white_balance):
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

        if color_sensor.supports(rs.option.white_balance):
            color_sensor.set_option(rs.option.white_balance, 3600)

        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        if color_sensor.supports(rs.option.brightness):
            try:
                brightness = -5
                color_sensor.set_option(rs.option.brightness, brightness)
                print("Brightness set to ", brightness)
            except Exception as e:
                print(f"Failed to set brightness: {e}")

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        clipping_distance_in_meters = 1  # 1 meter
        clipping_distance = clipping_distance_in_meters / self.depth_scale
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def getFrame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = depth_image.astype(np.float32) * self.depth_scale

        H, W = color_image.shape[:2]
        color = cv2.resize(color_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W, H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.1) | (depth >= np.inf)] = 0

        timestamp = color_frame.get_timestamp()

        return color, depth, timestamp
    
    
def save_img(path, img):
    cv2.imwrite(path, img)
