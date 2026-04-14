# hsar Package

The `hsar` package contains the ROS 2 node implementations for human skeleton action recognition.

## Nodes

- `mediapipe_pose_node`: converts image input into `HumanPose3D` messages.
- `detect_posture_node`: reads pose data and publishes posture classification.
- `detect_arm_state_node`: reads pose data and publishes arm state and direction classification.

## Models

Model assets are stored in `hsar_assets/models/`.

## Usage

See the top-level `README.md` for build and launch instructions.
