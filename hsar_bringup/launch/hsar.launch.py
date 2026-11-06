#!/usr/bin/env python3
"""
The whole HSAR pipeline in one namespace: MediaPipe 3D pose, posture and arm state.

    ros2 launch hsar_bringup hsar.launch.py image_topic:=/camera/color/image_raw

With the default namespace the topics are /hsar/human_pose_3d,
/hsar/mediapipe_pose_debug, /hsar/posture and /hsar/arm_state. The two
classifiers read the pose from /<namespace>/human_pose_3d. Every argument of
the three per-node launch files (models, thresholds, device, ...) can be set
here too.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _include(launch_file, namespace, extra_arguments=(), condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("hsar_bringup"), "launch", launch_file])
        ),
        launch_arguments=[("namespace", namespace), *extra_arguments],
        condition=condition,
    )


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    launch_posture = LaunchConfiguration("launch_posture")
    launch_arm_state = LaunchConfiguration("launch_arm_state")

    # The classifiers read the pose published in the same namespace
    pose_topic = ("mp_topic", ["/", namespace, "/human_pose_3d"])

    return LaunchDescription([
        DeclareLaunchArgument(
            "namespace",
            default_value="hsar",
            description="Namespace of the three nodes and their topics"
        ),
        DeclareLaunchArgument(
            "launch_posture",
            default_value="True",
            description="Start the posture classifier"
        ),
        DeclareLaunchArgument(
            "launch_arm_state",
            default_value="True",
            description="Start the arm-state classifier"
        ),

        _include("mediapipe_pose_node.launch.py", namespace),
        _include(
            "detect_posture_node.launch.py",
            namespace,
            [pose_topic],
            condition=IfCondition(launch_posture),
        ),
        _include(
            "detect_arm_state_node.launch.py",
            namespace,
            [pose_topic],
            condition=IfCondition(launch_arm_state),
        ),
    ])
