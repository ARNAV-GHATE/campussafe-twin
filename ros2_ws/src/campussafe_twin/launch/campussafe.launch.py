"""Launch Gazebo with the campus digital twin plus all CampusSafe nodes.

    ros2 launch campussafe_twin campussafe.launch.py
    ros2 launch campussafe_twin campussafe.launch.py scenario:=blocked_exit policy:=rule
    ros2 launch campussafe_twin campussafe.launch.py policy:=slm model:=qwen3:4b
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

PKG = "campussafe_twin"


def generate_launch_description():
    share = FindPackageShare(PKG)
    world = PathJoinSubstitution([share, "worlds", "campus_building.sdf"])
    params = PathJoinSubstitution([share, "config", "params.yaml"])

    args = [
        DeclareLaunchArgument("scenario", default_value="fire_north",
                              description="fire_north | blocked_exit | sensor_loss | multi_hazard"),
        DeclareLaunchArgument("policy", default_value="slm", description="slm | rule | mock | none"),
        DeclareLaunchArgument("model", default_value="llama3.2:3b", description="Ollama model tag"),
        DeclareLaunchArgument("seed", default_value="0"),
    ]
    overrides = {
        "scenario": LaunchConfiguration("scenario"),
        "policy": LaunchConfiguration("policy"),
        "model": LaunchConfiguration("model"),
        "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
        "use_sim_time": True,
    }

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])),
        launch_arguments={"gz_args": ["-r ", world]}.items(),
    )
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge", output="screen",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
    )

    def node(executable):
        return Node(package=PKG, executable=executable, name=executable, output="screen",
                    parameters=[params, overrides])

    return LaunchDescription(args + [
        gazebo, bridge,
        node("scenario_manager"), node("state_builder"), node("decision_node"),
        node("safety_validator"), node("robot_controller"), node("metrics_logger"),
    ])
