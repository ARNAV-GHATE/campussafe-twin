#!/usr/bin/env python3
"""Generates worlds/campus_building.sdf from the building model in core/building.py.

Re-run this whenever you change the floor plan, so Gazebo and the planner agree:
    python3 tools/generate_world.py
"""
from __future__ import annotations

import math
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from campussafe_twin.core.building import lamar_cs_floor, wall_segments  # noqa: E402
from campussafe_twin.core.scenarios import SCENARIOS  # noqa: E402

WALL_T, WALL_H = 0.2, 2.5
SPAWN = (20.0, 0.0)


def wall_links() -> str:
    out = []
    for i, (x1, y1, x2, y2) in enumerate(wall_segments()):
        length = math.hypot(x2 - x1, y2 - y1) + WALL_T
        cx, cy, yaw = (x1 + x2) / 2, (y1 + y2) / 2, math.atan2(y2 - y1, x2 - x1)
        geom = f"<geometry><box><size>{length:.2f} {WALL_T} {WALL_H}</size></box></geometry>"
        out.append(f"""      <link name="wall_{i}">
        <pose>{cx:.2f} {cy:.2f} {WALL_H / 2} 0 0 {yaw:.4f}</pose>
        <collision name="c">{geom}</collision>
        <visual name="v">{geom}<material><ambient>0.85 0.83 0.78 1</ambient><diffuse>0.85 0.83 0.78 1</diffuse></material></visual>
      </link>""")
    return "\n".join(out)


def exit_markers(b) -> str:
    out = []
    for e in b.exits:
        x, y = b.xy(e)
        out.append(f"""    <model name="marker_{e}">
      <static>true</static>
      <pose>{x} {y} 0.01 0 0 0</pose>
      <link name="l"><visual name="v"><geometry><box><size>1.2 1.2 0.02</size></box></geometry>
        <material><ambient>0 0.8 0.2 1</ambient><diffuse>0 0.8 0.2 1</diffuse><emissive>0 0.4 0.1 1</emissive></material></visual></link>
    </model>""")
    return "\n".join(out)


def hazard_markers(b) -> str:
    """One hidden marker per possible hazard location; scenario_manager raises it when active."""
    nodes = sorted({n for s in SCENARIOS.values() for n in s.hazard_nodes})
    out = []
    for n in nodes:
        x, y = b.xy(n)
        out.append(f"""    <model name="hazard_{n}">
      <pose>{x} {y} -5 0 0 0</pose>
      <link name="l">
        <gravity>false</gravity>
        <inertial><mass>1</mass></inertial>
        <visual name="v"><geometry><cylinder><radius>1.5</radius><length>2.0</length></cylinder></geometry>
          <transparency>0.4</transparency>
          <material><ambient>1 0.2 0 1</ambient><diffuse>1 0.2 0 1</diffuse><emissive>0.8 0.1 0 1</emissive></material></visual>
      </link>
    </model>""")
    return "\n".join(out)


ROBOT = f"""    <!-- Placeholder humanoid: torso + head on a differential-drive base.
         Swap for a full humanoid description (e.g. Unitree H1/G1) once locomotion is in scope. -->
    <model name="campus_robot">
      <pose>{SPAWN[0]} {SPAWN[1]} 0 0 0 0</pose>
      <link name="chassis">
        <pose>0 0 0.15 0 0 0</pose>
        <inertial><mass>8.0</mass><inertia><ixx>0.15</ixx><iyy>0.2</iyy><izz>0.25</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><box><size>0.5 0.4 0.15</size></box></geometry></collision>
        <visual name="base"><geometry><box><size>0.5 0.4 0.15</size></box></geometry>
          <material><ambient>0.2 0.2 0.25 1</ambient><diffuse>0.2 0.2 0.25 1</diffuse></material></visual>
        <visual name="torso"><pose>0 0 0.45 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>0.6</length></cylinder></geometry>
          <material><ambient>0.1 0.35 0.8 1</ambient><diffuse>0.1 0.35 0.8 1</diffuse></material></visual>
        <visual name="head"><pose>0.02 0 0.88 0 0 0</pose><geometry><sphere><radius>0.12</radius></sphere></geometry>
          <material><ambient>0.9 0.9 0.9 1</ambient><diffuse>0.9 0.9 0.9 1</diffuse></material></visual>
        <visual name="beacon"><pose>0.02 0 1.03 0 0 0</pose><geometry><cylinder><radius>0.04</radius><length>0.06</length></cylinder></geometry>
          <material><ambient>1 0.6 0 1</ambient><diffuse>1 0.6 0 1</diffuse><emissive>1 0.5 0 1</emissive></material></visual>
      </link>
      <link name="left_wheel">
        <pose>0.08 0.225 0.1 -1.5707 0 0</pose>
        <inertial><mass>0.5</mass><inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.0025</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></visual>
      </link>
      <link name="right_wheel">
        <pose>0.08 -0.225 0.1 -1.5707 0 0</pose>
        <inertial><mass>0.5</mass><inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.0025</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></visual>
      </link>
      <link name="caster">
        <pose>-0.18 0 0.05 0 0 0</pose>
        <inertial><mass>0.2</mass><inertia><ixx>0.0002</ixx><iyy>0.0002</iyy><izz>0.0002</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><sphere><radius>0.05</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><sphere><radius>0.05</radius></sphere></geometry></visual>
      </link>
      <joint name="left_wheel_joint" type="revolute">
        <parent>chassis</parent><child>left_wheel</child>
        <axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1.79769e+308</lower><upper>1.79769e+308</upper></limit></axis>
      </joint>
      <joint name="right_wheel_joint" type="revolute">
        <parent>chassis</parent><child>right_wheel</child>
        <axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1.79769e+308</lower><upper>1.79769e+308</upper></limit></axis>
      </joint>
      <joint name="caster_joint" type="ball"><parent>chassis</parent><child>caster</child></joint>
      <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.45</wheel_separation>
        <wheel_radius>0.1</wheel_radius>
        <odom_publish_frequency>20</odom_publish_frequency>
        <max_linear_acceleration>1.0</max_linear_acceleration>
        <topic>cmd_vel</topic>
        <odom_topic>odom</odom_topic>
      </plugin>
    </model>"""


def build_sdf() -> str:
    b = lamar_cs_floor()
    return f"""<?xml version="1.0"?>
<!-- GENERATED by tools/generate_world.py from core/building.py. Do not edit by hand. -->
<sdf version="1.9">
  <world name="campus">
    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <gui fullscreen="0"><camera name="user_camera"><pose>20 -28 30 0 0.85 1.5707</pose></camera></gui>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows><pose>0 0 30 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.4 0.2 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>120 120</size></plane></geometry></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>120 120</size></plane></geometry>
          <material><ambient>0.6 0.6 0.6 1</ambient><diffuse>0.6 0.6 0.6 1</diffuse></material></visual>
      </link>
    </model>
    <model name="building">
      <static>true</static>
{wall_links()}
    </model>
{exit_markers(b)}
{hazard_markers(b)}
{ROBOT}
  </world>
</sdf>
"""


if __name__ == "__main__":
    path = os.path.join(PKG, "worlds", "campus_building.sdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(build_sdf())
    print(f"wrote {path}")
