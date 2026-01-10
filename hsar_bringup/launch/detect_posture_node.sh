#!/bin/bash
set -e

# Activate venv
VENV_PATH=~/venvs/my_hpc/bin/activate
if [ -f "$VENV_PATH" ]; then
    echo "✅ Activating virtual environment from $VENV_PATH"
    source "$VENV_PATH"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi


# === RUN NODE ===
echo "🚀 Launching Posture detection node..."
ros2 launch hsar_bringup detect_posture_node.launch.py \
    namespace:=detect_posture_node \
    mp_topic:=/mediapipe_pose_node/human_pose_3d \
    model_name:=posture_mlp_merged_norm.pt \
    model_scaler_name:=posture_scaler_mereged_norm.pkl \
    use_geom_norm:=True \
    device:=cuda \
    smooth_window:=15 \
    conf_threshold:=0.6



# USAGE EXAMPLE
# ./detect_posture_node.sh
# ./detect_posture_node.sh
