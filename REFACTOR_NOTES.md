# Perception refactor notes: human_skeleton_action_recognition_ros2

Working notes for the `perception-refactor` branch, kept until it is merged into `main`.

- Base: `main` at `0d60b19`

## Changes

### One launch file for the whole pipeline

- New `hsar_bringup/launch/hsar.launch.py`: includes the three per-node launch files with one `namespace` (default `hsar`), so the nodes are `/hsar/mediapipe_pose_node`, `/hsar/detect_posture_node` and `/hsar/detect_arm_state_node`, and the topics `/hsar/human_pose_3d`, `/hsar/mediapipe_pose_debug`, `/hsar/posture` and `/hsar/arm_state`.
  - The classifiers' `mp_topic` is set to `/<namespace>/human_pose_3d`, so they read the pose node of the same launch.
  - `launch_posture` and `launch_arm_state` (default `True`) leave a classifier out.
  - Every argument of the per-node launch files can be passed to it (they are included, not copied), and their defaults stay in one place.
- The per-node launch files and shell wrappers are unchanged.
- README and `hsar_bringup/README.md` describe it.
- Checked by starting it on a laptop without the HSAR environment: the nodes got namespace `/hsar` and the posture node `mp_topic: /hsar/human_pose_3d`; with `launch_arm_state:=False` the arm-state node wasn't started. The nodes then stopped on the missing `mediapipe`/`torch` imports, so they haven't run.
