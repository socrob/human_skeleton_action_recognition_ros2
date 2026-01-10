#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    mp_topic = LaunchConfiguration("mp_topic")
    model_path = LaunchConfiguration("model_path")
    scaler_path = LaunchConfiguration("scaler_path")
    use_geom_norm = LaunchConfiguration("use_geom_norm")
    device = LaunchConfiguration("device")
    smooth_window = LaunchConfiguration("smooth_window")
    conf_threshold = LaunchConfiguration("conf_threshold")

    return LaunchDescription([
        DeclareLaunchArgument(
            "namespace",
            default_value="detect_posture_node",
            description="Namespace for posture detection node"
        ),
        DeclareLaunchArgument(
            "mp_topic",
            default_value="/mediapipe_pose_node/human_pose_3d",
            description="Mediapipe pose topic"
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value="posture_mlp_merged_norm.pt",
            description="Posture model path"
        ),
        DeclareLaunchArgument(
            "scaler_path",
            default_value="posture_scaler_mereged_norm.pkl",
            description="Posture model scaler path"
        ),
        DeclareLaunchArgument(
            "use_geom_norm",
            default_value="True",
            description="Whether the model was trained in geometric normalized data or not"
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="GPU or CPU"
        ),
        DeclareLaunchArgument(
            "smooth_window",
            default_value="15",
            description="Prediction windown"
        ),
        DeclareLaunchArgument(
            "conf_threshold",
            default_value="0.6",
            description="Prediction confidence"
        ),
        LifecycleNode(
            package="hsar",
            executable="detect_posture_node",
            name="detect_posture_node",
            namespace=namespace,
            parameters=[{
                "mp_topic": mp_topic,
                "model_path": model_path,
                "scaler_path": scaler_path,
                "use_geom_norm": use_geom_norm,
                "device": device,
                "smooth_window": smooth_window,
                "conf_threshold": conf_threshold,
            }],
            output="screen"
        )
    ])
