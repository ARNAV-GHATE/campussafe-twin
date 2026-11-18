"""Task 2 in Gazebo: hazard-aware navigation with a world model + NCM planner, under attack.

    ros2 launch campussafe_twin nav_task.launch.py
    ros2 launch campussafe_twin nav_task.launch.py route:=smoke_reroute nav_policy:=ncm_wm
    ros2 launch campussafe_twin nav_task.launch.py nav_attack:=odom_spoof nav_defense:=full
    ros2 launch campussafe_twin nav_task.launch.py nav_attack:=cmd_injection nav_defense:=none nav_policy:=classical

Arguments
    route        corridor_west | room_to_main | lobby_to_n4 | long_haul | blocked_east |
                 smoke_reroute | fire_behind | west_room          (nav/routes.py)
    nav_policy   classical | ncm | ncm_wm
    nav_attack   none | odom_spoof | lidar_phantom | cmd_injection | sensor_replay | dos | coordinated
    nav_defense  none | crypto | wm | full

Data flow
    Gazebo --(bridge)--> /odom_gz, /scan_gz --> nav_attack_injector --signed, tampered--> nav_controller
    nav_controller --signed /campussafe/nav/cmd--> nav_attack_injector (+ forged) --> nav_motor_driver --> /cmd_vel
    nav_metrics scores the run from ground truth and writes ~/campussafe_results/nav_*.csv
"""
import math

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PKG = "campussafe_twin"


def _nodes(context):
    from campussafe_twin.core.building import lamar_cs_floor
    from campussafe_twin.nav.routes import TEST_ROUTES

    route = LaunchConfiguration("route").perform(context)
    if route not in TEST_ROUTES:
        raise RuntimeError(f"unknown route '{route}', choose from {list(TEST_ROUTES)}")
    start = TEST_ROUTES[route].start_pose(lamar_cs_floor())
    common = {
        "route": route,
        "nav_policy": LaunchConfiguration("nav_policy").perform(context),
        "nav_attack": LaunchConfiguration("nav_attack").perform(context),
        "nav_defense": LaunchConfiguration("nav_defense").perform(context),
        "seed": int(LaunchConfiguration("seed").perform(context)),
        "spawn_x": float(start.x), "spawn_y": float(start.y), "spawn_yaw": float(start.th),
        "use_sim_time": True,
    }

    def node(executable):
        return Node(package=PKG, executable=executable, name=executable, output="screen", parameters=[common])

    return [node("nav_attack_injector"), node("nav_controller"), node("nav_motor_driver"), node("nav_metrics")]


def generate_launch_description():
    share = FindPackageShare(PKG)
    world = PathJoinSubstitution([share, "worlds", "campus_building.sdf"])
    args = [
        DeclareLaunchArgument("route", default_value="corridor_west"),
        DeclareLaunchArgument("nav_policy", default_value="ncm_wm", description="classical | ncm | ncm_wm"),
        DeclareLaunchArgument("nav_attack", default_value="none",
                              description="none | odom_spoof | lidar_phantom | cmd_injection | sensor_replay | dos | "
                                          "coordinated"),
        DeclareLaunchArgument("nav_defense", default_value="full", description="none | crypto | wm | full"),
        DeclareLaunchArgument("seed", default_value="0"),
    ]
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])),
        launch_arguments={"gz_args": ["-r ", world]}.items(),
    )
    # Gazebo truth goes to *_gz topics; the robot only ever sees what crosses the injector.
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge", output="screen",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        remappings=[("/odom", "/odom_gz"), ("/scan", "/scan_gz")],
    )
    return LaunchDescription(args + [gazebo, bridge, OpaqueFunction(function=_nodes)])
