# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        choices=["true", "false"],
        description="Use simulation time",
    ),
    DeclareLaunchArgument(
        "image_topic",
        default_value="/top_camera/rgb/preview/image_raw",
        description="Downward-facing camera image topic",
    ),
    DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="Velocity command topic",
    ),
]


def generate_launch_description():
    line_follower = Node(
        package="dis_tutorial7",
        executable="line_follower.py",
        name="line_follower",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"image_topic": LaunchConfiguration("image_topic")},
            {"cmd_vel_topic": LaunchConfiguration("cmd_vel_topic")},
        ],
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(line_follower)
    return ld
