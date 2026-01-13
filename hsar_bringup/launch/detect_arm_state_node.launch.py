#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    mp_topic = LaunchConfiguration("mp_topic")

    # Models
    arm_state_model_name = LaunchConfiguration("arm_state_model_name")
    arm_state_scaler_name = LaunchConfiguration("arm_state_scaler_name")

    pointing_dir_model_name = LaunchConfiguration("pointing_dir_model_name")
    pointing_dir_scaler_name = LaunchConfiguration("pointing_dir_scaler_name")

    arm_raised_dir_model_name = LaunchConfiguration("arm_raised_dir_model_name")
    arm_raised_dir_scaler_name = LaunchConfiguration("arm_raised_dir_scaler_name")

    # Runtime params
    device = LaunchConfiguration("device")
    smooth_window = LaunchConfiguration("smooth_window")
    state_conf_threshold = LaunchConfiguration("state_conf_threshold")
    dir_conf_threshold = LaunchConfiguration("dir_conf_threshold")
    wrist_hip_min_distance = LaunchConfiguration("wrist_hip_min_distance")
    wrist_shoulder_min_dy = LaunchConfiguration("wrist_shoulder_min_dy")

    return LaunchDescription([

        # --------------------------------------------------
        # Namespace / input
        # --------------------------------------------------
        DeclareLaunchArgument(
            "namespace",
            default_value="detect_arm_state_node",
            description="Namespace for full arm pipeline node"
        ),
        DeclareLaunchArgument(
            "mp_topic",
            default_value="/mediapipe_pose_node/human_pose_3d",
            description="Mediapipe pose topic"
        ),

        # --------------------------------------------------
        # Arm state model
        # --------------------------------------------------
        DeclareLaunchArgument(
            "arm_state_model_name",
            default_value="arm_state_mlp.pt",
            description="Arm-state MLP model filename"
        ),
        DeclareLaunchArgument(
            "arm_state_scaler_name",
            default_value="arm_state_scaler.pkl",
            description="Arm-state scaler filename"
        ),

        # --------------------------------------------------
        # Pointing direction model
        # --------------------------------------------------
        DeclareLaunchArgument(
            "pointing_dir_model_name",
            default_value="pointing_mlp_direction.pt",
            description="Pointing direction MLP model filename"
        ),
        DeclareLaunchArgument(
            "pointing_dir_scaler_name",
            default_value="pointing_scaler.pkl",
            description="Pointing direction scaler filename"
        ),

        # --------------------------------------------------
        # Arm-raised direction model
        # --------------------------------------------------
        DeclareLaunchArgument(
            "arm_raised_dir_model_name",
            default_value="arm_raised_direction_mlp.pt",
            description="Arm-raised direction MLP model filename"
        ),
        DeclareLaunchArgument(
            "arm_raised_dir_scaler_name",
            default_value="arm_raised_direction_scaler.pkl",
            description="Arm-raised direction scaler filename"
        ),

        # --------------------------------------------------
        # Runtime parameters
        # --------------------------------------------------
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="Device to run inference on (cuda / cpu)"
        ),
        DeclareLaunchArgument(
            "smooth_window",
            default_value="15",
            description="Temporal smoothing window"
        ),
        DeclareLaunchArgument(
            "state_conf_threshold",
            default_value="0.6",
            description="Confidence threshold for arm-state prediction"
        ),
        DeclareLaunchArgument(
            "dir_conf_threshold",
            default_value="0.6",
            description="Confidence threshold for direction prediction"
        ),
        DeclareLaunchArgument(
            "wrist_hip_min_distance",
            default_value="0.95",
            description="Geometric gate: wrist_hip minimum distance"
        ),
        DeclareLaunchArgument(
            "wrist_shoulder_min_dy",
            default_value="0.35",
            description="Geometric gate: wrist_shoulder vertical offset"
        ),

        # --------------------------------------------------
        # Lifecycle node
        # --------------------------------------------------
        LifecycleNode(
            package="hsar",
            executable="detect_arm_state_node",
            name="detect_arm_state_node",
            namespace=namespace,
            parameters=[{
                "mp_topic": mp_topic,

                "arm_state_model_name": arm_state_model_name,
                "arm_state_scaler_name": arm_state_scaler_name,

                "pointing_dir_model_name": pointing_dir_model_name,
                "pointing_dir_scaler_name": pointing_dir_scaler_name,

                "arm_raised_dir_model_name": arm_raised_dir_model_name,
                "arm_raised_dir_scaler_name": arm_raised_dir_scaler_name,

                "device": device,
                "smooth_window": smooth_window,
                "state_conf_threshold": state_conf_threshold,
                "dir_conf_threshold": dir_conf_threshold,
                "wrist_hip_min_distance": wrist_hip_min_distance,
                "wrist_shoulder_min_dy": wrist_shoulder_min_dy,
            }],
            output="screen"
        )
    ])
