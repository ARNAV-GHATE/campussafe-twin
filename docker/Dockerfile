# CampusSafe Twin: ROS 2 Jazzy + Gazebo Harmonic + CPU PyTorch, ready to launch.
#
#   docker build -t campussafe -f docker/Dockerfile .          (from the repo root)
#   ./docker/run.sh                                            (Gazebo GUI on Linux/X11)
#   docker run --rm -it campussafe python3 run_experiments.py --quick   (headless)
FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz python3-pip python3-pytest python3-numpy python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*
# CPU wheel keeps the image small; a GPU is not needed for these models.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /campussafe
COPY . .
RUN . /opt/ros/jazzy/setup.sh && cd ros2_ws && colcon build --symlink-install

RUN echo 'source /opt/ros/jazzy/setup.bash && source /campussafe/ros2_ws/install/setup.bash' >> /root/.bashrc
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && source /campussafe/ros2_ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]
