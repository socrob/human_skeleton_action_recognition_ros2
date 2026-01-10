#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    image_topic = LaunchConfiguration("image_topic")
    image_reliability = LaunchConfiguration("image_reliability")
    live_visualization = LaunchConfiguration("live_visualization")

    return LaunchDescription([
        DeclareLaunchArgument(
            "namespace",
            default_value="mediapipe_pose_node",
            description="Namespace for the Mediapipe node"
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/k4a/rgb/image_raw",
            description="Image topic for RGB input"
        ),
        DeclareLaunchArgument(
            "image_reliability",
            default_value="2",  # 1 = RELIABLE, 2 = BEST_EFFORT
            description="QoS reliability setting"
        ),
        DeclareLaunchArgument(
            "live_visualization",
            default_value="False",
            description="Mediapipe Visualization visualization"
        ),
        LifecycleNode(
            package="hsar",
            executable="mediapipe_pose_node",
            name="mediapipe_pose_node",
            namespace=namespace,
            parameters=[{
                "image_topic": image_topic,
                "image_reliability": image_reliability,
                "live_visualization": live_visualization,
            }],
            output="screen"
        )
    ])
