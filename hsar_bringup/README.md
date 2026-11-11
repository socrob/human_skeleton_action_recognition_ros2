# hsar_bringup Package

The `hsar_bringup` package provides ROS 2 launch files and shell wrappers for the HSAR nodes.

## Launch Files

- `hsar.launch.py`: the three nodes in one namespace (`hsar` by default), the classifiers reading `/<namespace>/human_pose_3d`; `launch_posture` / `launch_arm_state` switch the classifiers
- `mediapipe_pose_node.launch.py`
- `detect_posture_node.launch.py`
- `detect_arm_state_node.launch.py`

## Usage

After building the workspace, use `ros2 launch hsar_bringup <launch-file>`.
See the top-level `README.md` for complete setup and virtual environment guidance.
