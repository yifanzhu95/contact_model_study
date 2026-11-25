Per-object hardware joint-cost references

Capture cube: python capture_leap_joint_reference.py --object cube
Other supported objects: ball, duck.

The viewer begins with every motor torque-off. Select 1=index, 2=middle,
3=ring, 4=thumb. Move the selected finger by hand, hold still and press Enter.
Only that finger is enabled, with its measured pose preloaded as the target.

Select any locked finger and press Backspace to release only that finger.
Reposition it and press Enter again. The other fingers remain held.

After all four fingers are locked and stable for 0.75 seconds, the mean measured
pose is written to cube.json, ball.json or duck.json. Targets and stability
statistics are also recorded. Earlier versions are copied into history/.
The hand stays held for inspection. Q, Escape, Space, window close or Ctrl-C
ends capture and disables torque on all motors.

The hardware workflow automatically loads the selected object's file at the
next launch, using joint_reference for the w_joint cost. init_qpos, init_ctrl
and episode preparation robot_qref are unchanged. Each object has one reference
shared across its contact models and geometry accuracies.

Use --no-object-joint-reference to restore the legacy joint-cost home, or
--joint-reference-file PATH to choose a specific saved file. The existing
--measured-reference option explicitly overrides the cost reference per episode.
Standard run_episodes.py does not automatically load hardware capture files.
