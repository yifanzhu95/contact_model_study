# Preserve the prepared FoundationPose/SAM/CUDA stack; add the tracking LCM codec.
FROM irislab_fppp:online_tracking_snapshot
RUN python -m pip install --no-cache-dir lcm==1.5.2
