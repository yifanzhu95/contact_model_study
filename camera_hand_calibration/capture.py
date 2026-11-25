"""RealSense RGB-D capture and opt-in PASSIVE joint-state subscription.

This module has no serial, torque, action publisher or motor-command API.
"""
from collections import deque
from datetime import datetime, timezone
import json
import queue
import shutil
import struct
import threading
import time

import numpy as np

from workspace import ROOT,local,write_json,read_json,identifier,utc,digest,environment,verify_model


class Writer:
    def __init__(self,directory):
        self.directory=directory;self.queue=queue.Queue(128);self.error=None;self.frames=0
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def submit(self,kind,record):
        if self.error:raise RuntimeError('Capture writer failed') from self.error
        try:self.queue.put_nowait((kind,record))
        except queue.Full:raise RuntimeError('Capture writer queue full; stopping rather than silently losing data')

    def run(self):
        streams={}
        try:
            while True:
                job=self.queue.get()
                if job is None:break
                kind,record=job
                if kind=='frame':
                    index,color,depth,meta=record
                    np.savez_compressed(self.directory/f'frame_{index:05d}.npz',
                        color_rgb=color,depth_z16=depth,metadata_json=np.asarray(json.dumps(meta,allow_nan=False)))
                    self.frames+=1
                    if index==0:
                        import cv2
                        cv2.imwrite(str(self.directory/'preview_rgb.png'),cv2.cvtColor(color,cv2.COLOR_RGB2BGR))
                else:
                    if kind not in streams:streams[kind]=(self.directory/f'{kind}.jsonl').open('w')
                    streams[kind].write(json.dumps(record,allow_nan=False)+'\n')
        except BaseException as exc:self.error=exc
        finally:
            for f in streams.values():f.close()

    def close(self):
        while self.thread.is_alive():
            try:self.queue.put(None,timeout=.1);break
            except queue.Full:continue
        self.thread.join()
        if self.error:raise RuntimeError('Capture writer failed') from self.error


class JointSubscriber:
    def __init__(self,config,writer):
        import lcm
        self.config,self.writer=config,writer
        self.connection=lcm.LCM(config['url']);self.samples=deque(maxlen=10000)
        self.lock=threading.Lock();self.stopping=threading.Event();self.error=None
        self.subscription=self.connection.subscribe(config['channel'],self.callback)
        self.subscription.set_queue_capacity(1000)
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def callback(self,channel,payload):
        if len(payload)!=136:raise ValueError('LEAP_STATE must be <q timestamp + 16 float32 positions + 16 float32 velocities')
        stamp=struct.unpack_from('<q',payload)[0]*1e-6
        values=np.frombuffer(payload,dtype='<f4',offset=8).astype(float)
        if stamp<=0 or not np.isfinite(values).all():raise ValueError('Invalid measured joint state')
        receive=time.time()
        record=dict(timestamp_epoch_s=stamp,received_epoch_s=receive,q=values[:16].tolist(),v=values[16:].tolist())
        with self.lock:self.samples.append(record)
        self.writer.submit('joints',record)

    def run(self):
        try:
            while not self.stopping.is_set():self.connection.handle_timeout(20)
        except BaseException as exc:self.error=exc

    def match(self,stamp,time_basis):
        if self.error:raise RuntimeError('Joint subscriber failed') from self.error
        with self.lock:rows=list(self.samples)
        return match_joints(rows,stamp,time_basis,self.config)

    def close(self):
        self.stopping.set();self.thread.join(timeout=1)
        self.connection.unsubscribe(self.subscription)


def match_joints(rows,stamp,time_basis,config):
    if not rows:return dict(usable=False,reason='No joint states received')
    nearest=min(rows,key=lambda r:abs(r['timestamp_epoch_s']-stamp))
    skew=nearest['timestamp_epoch_s']-stamp
    window=sorted({r['timestamp_epoch_s']:r for r in rows if stamp-config['settle_window_s']<=r['timestamp_epoch_s']<=stamp}.values(),key=lambda r:r['timestamp_epoch_s'])
    enough=len(window)>=config['min_settle_samples']
    span=window[-1]['timestamp_epoch_s']-window[0]['timestamp_epoch_s'] if len(window)>1 else 0.
    motion=float(np.ptp([r['q'] for r in window],axis=0).max()) if window else None
    speed=float(np.max(abs(np.asarray([r['v'] for r in window])))) if window else None
    gap=float(np.max(np.diff([r['timestamp_epoch_s'] for r in window]))) if len(window)>1 else None
    usable=(time_basis=='sdk_epoch' and abs(skew)*1000<=config['max_frame_skew_ms'] and enough
            and span>=.75*config['settle_window_s'] and motion<=config['max_motion_rad']
            and speed<=config['max_speed_rad_s'] and gap<=config.get('max_sample_gap_s',.05)+1e-8)
    return dict(usable=bool(usable),timestamp_epoch_s=nearest['timestamp_epoch_s'],
        skew_ms=skew*1000,q=nearest['q'],v=nearest['v'],settle_samples=len(window),
        settle_span_s=span,max_motion_rad=motion,max_speed_rad_s=speed,max_sample_gap_s=gap,
        reason=None if usable else 'Require epoch-synchronized, nearby joints and a sufficiently long settled window')


def intrinsics(value):
    return dict(width=value.width,height=value.height,fx=value.fx,fy=value.fy,cx=value.ppx,cy=value.ppy,
        K=[[value.fx,0,value.ppx],[0,value.fy,value.ppy],[0,0,1]],
        distortion_model=str(value.model),distortion_coefficients=list(value.coeffs))


def devices():
    import pyrealsense2 as rs
    return [dict(serial=d.get_info(rs.camera_info.serial_number),name=d.get_info(rs.camera_info.name),
        firmware=d.get_info(rs.camera_info.firmware_version)) for d in rs.context().query_devices()]


def frame_epoch(timestamp_ms,domain,received):
    epoch=timestamp_ms*.001
    if domain in ('timestamp_domain.global_time','timestamp_domain.system_time') and abs(epoch-received)<5:
        return epoch,'sdk_epoch'
    return received,'host_receipt_approximate'


def record(config_path,frames=120,joints=False,preview=False,pose_id='camera_check',split='unassigned'):
    import pyrealsense2 as rs
    import cv2
    config=read_json(config_path);verify_model()
    camera=config['camera'];directory=local('datasets')/identifier();directory.mkdir(parents=True)
    if any(not isinstance(camera[k],int) or camera[k]<=0 for k in ['width','height','fps','save_every_n_frames']):
        raise ValueError('Camera dimensions, FPS and save interval must be positive integers')
    if not isinstance(camera['warmup_frames'],int) or camera['warmup_frames']<0:raise ValueError('Invalid warmup frame count')
    writer=Writer(directory);subscriber=None;pipeline=rs.pipeline();started=False;original=[]
    manifest=dict(schema_version=1,created_at_utc=utc(),status='starting',pose_id=pose_id,split=split,
        joints_enabled=joints,config=config,config_sha256=digest(local(config_path)),environment=environment(),
        code_sha256={p.name:digest(p) for p in ROOT.glob('*.py')},
        model_manifest=read_json('model/manifest.json'),frames_requested=frames)
    write_json(directory/'session.json',manifest)
    (directory/'code').mkdir()
    for source in ROOT.glob('*.py'):shutil.copy2(source,directory/'code'/source.name)
    write_json(directory/'experiment.json',config)
    rs.log_to_file(rs.log_severity.warn,str(directory/'realsense.log'))
    try:
        if joints:subscriber=JointSubscriber(config['joints'],writer)
        selection=rs.config()
        if camera.get('serial'):selection.enable_device(camera['serial'])
        selection.enable_stream(rs.stream.color,camera['width'],camera['height'],rs.format.rgb8,camera['fps'])
        selection.enable_stream(rs.stream.depth,camera['width'],camera['height'],rs.format.z16,camera['fps'])
        profile=pipeline.start(selection);started=True;device=profile.get_device()
        for sensor in device.query_sensors():
            if camera['enable_global_time'] and sensor.supports(rs.option.global_time_enabled):
                original.append((sensor,sensor.get_option(rs.option.global_time_enabled)))
                sensor.set_option(rs.option.global_time_enabled,1)
        color=profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth=profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_intrinsics=intrinsics(color.get_intrinsics())
        serial=device.get_info(rs.camera_info.serial_number)
        scale=float(device.first_depth_sensor().get_depth_scale())
        sensor_options=[]
        for sensor in device.query_sensors():
            values={}
            for option in sensor.get_supported_options():
                try:values[str(option)]=float(sensor.get_option(option))
                except RuntimeError:pass
            sensor_options.append(dict(name=sensor.get_info(rs.camera_info.name),options=values))
        camera_profile=dict(serial=serial,name=device.get_info(rs.camera_info.name),
            firmware=device.get_info(rs.camera_info.firmware_version),fps=color.fps(),
            color=color_intrinsics,raw_depth=intrinsics(depth.get_intrinsics()),
            depth_to_color=dict(rotation=list(depth.get_extrinsics_to(color).rotation),
                rotation_storage='column-major 3x3 (RealSense SDK)',translation_m=list(depth.get_extrinsics_to(color).translation)),
            depth_scale_m=scale,sensor_options=sensor_options,
            depth_storage='uint16 aligned to color; multiply by depth_scale_m for meters',
            pinhole_supported=bool(np.allclose(color_intrinsics['distortion_coefficients'],0,atol=1e-10)))
        write_json(directory/'camera_profile.json',camera_profile)
        manifest.update(status='capturing',camera_profile=camera_profile);write_json(directory/'session.json',manifest)
        align=rs.align(rs.stream.color)
        for _ in range(camera['warmup_frames']):pipeline.wait_for_frames(5000)
        previous_number=None;saved=0;all_times=[];skews=[];valid_depth=[];drops=0;clock_bases=set()
        for i in range(frames):
            request=time.time();raw=pipeline.wait_for_frames(5000);received=time.time()
            aligned=align.process(raw);c=aligned.get_color_frame();d=aligned.get_depth_frame()
            if not c or not d:raise RuntimeError('Missing aligned RGB/depth frame')
            stamp,basis=frame_epoch(c.get_timestamp(),str(c.get_frame_timestamp_domain()),received)
            clock_bases.add(basis);number=int(c.get_frame_number())
            skipped=0 if previous_number is None else max(0,number-previous_number-1)
            previous_number=number;drops+=skipped
            all_times.append(float(c.get_timestamp())*.001)
            skew=float(c.get_timestamp()-d.get_timestamp());skews.append(skew)
            depth_data=np.asanyarray(d.get_data());sample=depth_data[::8,::8]*scale
            valid_depth.append(float(np.mean((sample>=camera['depth_min_m'])&(sample<=camera['depth_max_m']))))
            timing=dict(index=i,request_epoch_s=request,received_epoch_s=received,
                color_timestamp_ms=float(c.get_timestamp()),color_timestamp_domain=str(c.get_frame_timestamp_domain()),
                depth_timestamp_ms=float(d.get_timestamp()),depth_timestamp_domain=str(d.get_frame_timestamp_domain()),
                color_frame_number=number,depth_frame_number=int(d.get_frame_number()),
                color_depth_skew_ms=skew,source_frames_skipped=skipped,
                timestamp_epoch_s=stamp,time_basis=basis)
            writer.submit('timing',timing)
            if subscriber and subscriber.error:raise RuntimeError('Joint subscription failed') from subscriber.error
            if i%camera['save_every_n_frames']==0:
                rgb=np.asanyarray(c.get_data()).copy();depth_copy=depth_data.copy()
                if rgb.shape!=(camera['height'],camera['width'],3) or depth_copy.shape!=(camera['height'],camera['width']):
                    raise RuntimeError('Actual frame dimensions differ from requested profile')
                meta=dict(**timing,depth_scale_m=scale,camera_serial=serial,
                    camera_intrinsics=color_intrinsics['K'],distortion_coefficients=color_intrinsics['distortion_coefficients'],
                    pose_id=pose_id,split=split,
                    joints=None if subscriber is None else subscriber.match(stamp,basis))
                writer.submit('frame',(saved,rgb,depth_copy,meta));saved+=1
            if preview:
                cv2.imshow('Camera calibration capture — Q/Esc stops',cv2.cvtColor(np.asanyarray(c.get_data()),cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1)&255 in (27,ord('q')):break
        intervals=np.diff(all_times)
        manifest.update(status='complete',frames_received=len(all_times),frames_saved=saved,
            source_frames_skipped=drops,median_frame_interval_ms=float(np.median(intervals)*1000) if len(intervals) else None,
            measured_fps=float(1/np.median(intervals)) if len(intervals) else None,
            mean_valid_depth_fraction=float(np.mean(valid_depth)),
            median_abs_color_depth_skew_ms=float(np.median(abs(np.asarray(skews)))),clock_bases=sorted(clock_bases))
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        cleanup=[]
        if subscriber:
            try:subscriber.close()
            except Exception as exc:cleanup.append(str(exc))
        for sensor,value in original:
            try:sensor.set_option(rs.option.global_time_enabled,value)
            except RuntimeError as exc:cleanup.append(str(exc))
        if started:
            try:pipeline.stop()
            except RuntimeError as exc:cleanup.append(str(exc))
        if preview:cv2.destroyAllWindows()
        try:writer.close()
        except BaseException as exc:
            manifest.update(status='failed',writer_error=str(exc));raise
        finally:
            manifest.update(finished_at_utc=utc(),writer_frames=writer.frames,cleanup_warnings=cleanup)
            write_json(directory/'session.json',manifest)
    return dict(directory=str(directory.relative_to(ROOT)),**{k:manifest[k] for k in ['status','frames_received','frames_saved','measured_fps','source_frames_skipped','mean_valid_depth_fraction','clock_bases']})


def load_frame(path):
    with np.load(local(path),allow_pickle=False) as f:
        rgb=f['color_rgb'];depth=f['depth_z16'];meta=json.loads(str(f['metadata_json']))
    if rgb.dtype!=np.uint8 or depth.dtype!=np.uint16 or rgb.shape[:2]!=depth.shape:raise ValueError('Corrupt RGB-D frame')
    if not np.allclose(meta['distortion_coefficients'],0,atol=1e-10):
        raise ValueError('Fitting currently requires rectified/pinhole frames; nonzero distortion must be rectified explicitly')
    return rgb,depth.astype(np.float32)*meta['depth_scale_m'],meta
