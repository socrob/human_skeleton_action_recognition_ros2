# Human Skeleton Action Recognition ROS2

A ROS 2 repository for real-time human pose, posture, and arm state recognition.
This project combines MediaPipe pose detection with lightweight MLP models for posture classification and arm action detection.

## Overview

- `hsar_msgs`: custom message definitions for `HumanPose3D` and `HumanPosture`.
- `hsar`: ROS 2 lifecycle node implementations.
- `hsar_bringup`: launch files and shell wrappers for the nodes.
- `hsar_assets`: pretrained model files and normalization assets.

## Available Nodes and Functionality

### `mediapipe_pose_node`
- Reads an RGB image topic.
- Runs MediaPipe pose estimation.
- Publishes `HumanPose3D` on `human_pose_3d`.
- Can optionally publish a visualization image when `live_visualization=True`.

### `detect_posture_node`
- Subscribes to `/mediapipe_pose_node/human_pose_3d` by default.
- Publishes posture output on `posture`.
- Outputs one of `standing`, `sitting`, or `lying`.

### `detect_arm_state_node`
- Subscribes to `/mediapipe_pose_node/human_pose_3d` by default.
- Publishes arm state output on `arm_state`.
- Produces labels such as `arm_down`, `pointing`, and `arm_raised`.
- When the arm is classified as pointing or raised, it can refine the label to `pointing_left`, `pointing_right`, `arm_raised_left`, or `arm_raised_right`.
- If pose input is invalid or confidence is low, the output may be `unknown`.

## Supported Messages

### `hsar_msgs/msg/HumanPose3D.msg`
- `std_msgs/Header header`
- `float32[] landmarks` (33 × 3 = 99 values)
- `bool valid`

### `hsar_msgs/msg/HumanPosture.msg`
- `std_msgs/Header header`
- `string posture`
- `float32 confidence`
- `float32[] probabilities`

## Model Notes and Accuracy Warning

The provided pretrained models were trained on a very small dataset.
This means they are prototype-level models and may not be highly accurate in general use.
Expect limited generalization and verify results carefully before relying on them in production.

## Recommended Setup

This repository is typically used inside a Python virtual environment to isolate dependencies.

### Recommended workflow

1. Create and install dependencies in a virtual environment:

```bash
python3 -m venv ~/hsar_venv
source ~/hsar_venv/bin/activate
pip install -r requirements.txt
deactivate
```

2. Build the workspace without activating the virtual environment:

```bash
source /opt/ros/<ros2-distro>/setup.bash
cd /home/socrob/ros2_ws
colcon build
```

> Do not use `colcon build --symlink-install`.
> Do not activate the Python virtual environment while building or compiling.

3. Launch nodes with the virtual environment activated:

```bash
source /opt/ros/<ros2-distro>/setup.bash
source /home/socrob/ros2_ws/install/setup.bash
source ~/hsar_venv/bin/activate
ros2 launch hsar_bringup mediapipe_pose_node.launch.py
```

## Example Launch Commands

Launch the MediaPipe pose node:

```bash
ros2 launch hsar_bringup mediapipe_pose_node.launch.py
```

Launch posture detection:

```bash
ros2 launch hsar_bringup detect_posture_node.launch.py
```

Launch full arm state pipeline:

```bash
ros2 launch hsar_bringup detect_arm_state_node.launch.py
```

## Notes on Launch Files

The `hsar_bringup` package includes these launch definitions:
- `mediapipe_pose_node.launch.py`
- `detect_posture_node.launch.py`
- `detect_arm_state_node.launch.py`

It also includes shell wrappers under `hsar_bringup/launch/` for easier invocation.

## Package Structure

- `hsar/`: node implementations and models package.
- `hsar_bringup/`: ROS 2 launch files and wrappers.
- `hsar_msgs/`: custom ROS 2 message definitions.
- `hsar_assets/`: pretrained model weights and normalization files.

## Important

This repository is intended to be used as a research/prototype system.
The models available are not guaranteed to be accurate on arbitrary input and should be evaluated carefully.
