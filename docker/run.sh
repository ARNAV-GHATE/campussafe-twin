#!/usr/bin/env bash
# Start the container with the Gazebo GUI forwarded to your X11 display (Linux).
# Extra arguments are passed to `ros2 launch`, e.g.:
#   ./docker/run.sh policy:=ncm attack:=false_hazard defense:=full
set -e
xhost +local:root >/dev/null 2>&1 || true
docker run --rm -it --net=host \
  -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/results:/root/campussafe_results" \
  campussafe ros2 launch campussafe_twin campussafe.launch.py "$@"
