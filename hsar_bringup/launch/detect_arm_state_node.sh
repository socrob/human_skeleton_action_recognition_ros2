#!/bin/bash
set -e

# ==========================================================
# ACTIVATE VIRTUAL ENV
# ==========================================================
VENV_PATH=~/venvs/my_hpc/bin/activate
if [ -f "$VENV_PATH" ]; then
    echo "✅ Activating virtual environment from $VENV_PATH"
    source "$VENV_PATH"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi


# ==========================================================
# RUN FULL ARM PIPELINE NODE
# ==========================================================
echo "🚀 Launching Full Arm State + Direction pipeline..."

ros2 launch hsar_bringup detect_arm_state_node.launch.py \
    namespace:=arm_pipeline_node \
    mp_topic:=/mediapipe_pose_node/human_pose_3d \
    \
    arm_state_model_name:=arm_state_mlp.pt \
    arm_state_scaler_name:=arm_state_scaler.pkl \
    \
    pointing_dir_model_name:=pointing_mlp_direction.pt \
    pointing_dir_scaler_name:=pointing_scaler.pkl \
    \
    arm_raised_dir_model_name:=arm_raised_direction_mlp.pt \
    arm_raised_dir_scaler_name:=arm_raised_direction_scaler.pkl \
    \
    device:=cuda \
    smooth_window:=15 \
    state_conf_threshold:=0.6 \
    dir_conf_threshold:=0.6 \
    wrist_hip_min_distance:=0.95 \
    wrist_shoulder_min_dy:=0.35


# ==========================================================
# USAGE
# ==========================================================
# chmod +x detect_arm_pipeline_node.sh
# ./detect_arm_pipeline_node.sh
