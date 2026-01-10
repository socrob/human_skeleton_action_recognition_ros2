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
    model_path:=posture_mlp_merged_norm.pt \
    scaler_path:=posture_scaler_mereged_norm.pkl \
    use_geom_norm:=2 \
    use_geom_norm:=True \
    smooth_window:=15 \
    conf_threshold:=0.6



# USAGE EXAMPLE
# ./detect_posture_node.sh
# ./detect_posture_node.sh
