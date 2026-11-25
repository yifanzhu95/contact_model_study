#!/usr/bin/env bash
# Interactive by default; --detach and an explicit command are used by the launcher.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$TRACKER_ROOT/.." && pwd)"
IMAGE="${LEAP_VISION_IMAGE:-irislab_fppp:online_tracking_snapshot}"
CONTAINER_NAME="fppp_dev"
DOCKER_BIN="docker"
DETACH=false
MOUNTS=()
while (($#)); do
  case "$1" in
    --detach) DETACH=true; shift ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --name) CONTAINER_NAME="$2"; shift 2 ;;
    --docker-executable) DOCKER_BIN="$2"; shift 2 ;;
    --mount-path) MOUNTS+=("-v" "$2:$2"); shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
TRACKER_ROOT="$REPO_ROOT/iris-foundationpose-plusplus"
COMMAND=("$@")
if ((${#COMMAND[@]} == 0)); then COMMAND=(bash); fi
OPTIONS=(run --gpus all --network=host --ipc=host --name "$CONTAINER_NAME"
  --security-opt seccomp=unconfined --rm --privileged
  -w "$TRACKER_ROOT" -v "$REPO_ROOT:$REPO_ROOT" -v /dev:/dev
  -e "DISPLAY=${DISPLAY:-}" -e PYTHONUNBUFFERED=1)
if $DETACH; then OPTIONS+=(-d); else OPTIONS+=(-it); fi
exec "$DOCKER_BIN" "${OPTIONS[@]}" "${MOUNTS[@]}" "$IMAGE" "${COMMAND[@]}"
