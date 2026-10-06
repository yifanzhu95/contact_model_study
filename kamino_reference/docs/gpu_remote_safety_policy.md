# GPU/AnyDesk remote-work safety policy

## Observed host topology

Read-only inspection on 2026-10-05 found:

- one NVIDIA RTX 5070 Ti with 16,303 MiB VRAM;
- NVIDIA `display_active=Enabled`;
- Xorg and GNOME Shell are NVIDIA graphics clients;
- the connected 2560×1440 display runs at 144 Hz on NVIDIA HDMI;
- AnyDesk captures the same X11 desktop;
- an AMD integrated GPU is present and initialized, but has no connected
  display and is only an offload provider;
- AnyDesk is active, while SSH is inactive and not installed as a service;
- 30 GiB system RAM and 8 GiB swap are healthy at inspection time;
- no NVIDIA `NVRM: Xid`, GPU reset, or OOM was present in the retained current
  and three previous kernel boots;
- recent AnyDesk traces include network timeouts, X11 `Could not find a primary
  monitor`, and IPC timeout events.

The evidence does not prove one unique historical failure mechanism. It does
prove a single point of failure: desktop rendering, X11 capture, AnyDesk, and
CUDA share the RTX. CUDA starvation, VRAM pressure, or a driver hang can make
the only remote-control path unavailable.

## Mandatory policy for assistant-run experiments

1. No heavy CUDA command may be run directly. Use
   `scripts/run_gpu_guarded.py run`.
2. `artifacts/safety/REMOTE_WORK_LOCK.json` remains present whenever remote
   continuity is important. The guard refuses work under this lock unless the
   exact one-shot recovery-window token is supplied after explicit user
   confirmation.
3. Because NVIDIA currently drives the desktop, even an authorized run must
   explicitly pass `--allow-display-gpu`.
4. Start with at most 4 worlds, H≤5, and a 30 s watchdog. Increase one resource
   at a time only after the desktop remains responsive.
5. Keep at least 6 GiB VRAM and 6 GiB available system RAM. Abort above 75 C,
   if AnyDesk stops, if X11 stops responding, or at the wall-time limit.
6. Long candidate sets must be processed in bounded chunks with a health pause
   between chunks. Chunk-size invariance must be audited before scientific
   results from different chunk sizes are combined.
7. The guard log is an experiment artifact. A killed or disconnected run is
   invalid, not a partial success.

The wrapper lowers CPU scheduling priority, but Linux `nice` does not limit
CUDA occupancy. The watchdog also cannot recover a fully wedged NVIDIA driver;
it only stops the workload while the host can still schedule the monitor.

## Read-only status command

```bash
.venv/bin/python scripts/run_gpu_guarded.py status
```

## Example authorized micro-smoke

Only after the user explicitly confirms a recovery window:

```bash
.venv/bin/python scripts/run_gpu_guarded.py run \
  --allow-display-gpu \
  --remote-window-token USER_CONFIRMED_RECOVERY_WINDOW \
  --maximum-runtime-s 30 \
  --minimum-free-vram-mib 6144 \
  --maximum-temperature-c 75 \
  --minimum-available-ram-mib 6144 \
  --log artifacts/safety/example_micro_smoke.jsonl \
  -- <CUDA command and arguments>
```

The token is not standing permission. It records that one invocation was
launched only after an explicitly confirmed safe window.

## High-assurance remediation

Software guards reduce risk; they cannot guarantee AnyDesk continuity while
the RTX drives both X11 and CUDA. The robust topology is:

```text
AMD integrated GPU -> physical display -> Xorg/GNOME -> AnyDesk
NVIDIA RTX 5070 Ti -> headless CUDA only
```

This requires a scheduled local/recovery window: connect the monitor to the
motherboard output, make the AMD iGPU the primary display in firmware/Xorg,
reboot, and verify that `nvidia-smi` reports no Xorg/GNOME graphics clients and
`display_active` is disabled. Do not attempt this while AnyDesk is the only
connection.

A second control path is also required before unattended heavy runs. Enable
and test SSH on the LAN or a managed VPN such as Tailscale, including an actual
reconnect and process-kill drill. Installation/firewall changes require a
separate authorized maintenance window.

Reducing the RTX power limit from 300 W to its 250 W minimum may reduce power
and thermal transients, but it does not reserve GPU scheduling time for Xorg
and is not a substitute for GPU separation. NVIDIA MPS/thread-percentage
limits could alter Warp/Kamino behavior and are not adopted without a separate
validation.
