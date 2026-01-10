#!/bin/bash
set -e

# Default to RealSense
CAMERA_TYPE="realsense"

# Parse CLI args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --camera)
            CAMERA_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Activate venv
VENV_PATH=~/venvs/my_hpc/bin/activate
if [ -f "$VENV_PATH" ]; then
    echo "✅ Activating virtual environment from $VENV_PATH"
    source "$VENV_PATH"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi


# Set image topics based on camera type
if [ "$CAMERA_TYPE" == "azure" ]; then
    IMAGE_TOPIC="/k4a/rgb/image_raw"
elif [ "$CAMERA_TYPE" == "realsense" ]; then
    IMAGE_TOPIC="/camera/camera/color/image_raw"
elif [ "$CAMERA_TYPE" == "sim_head" ]; then
    IMAGE_TOPIC="/head_front_camera/rgb/image_raw"
elif [ "$CAMERA_TYPE" == "sim_wrist" ]; then
    IMAGE_TOPIC="/realsense_d435/color/image_raw"
else
    echo "❌ Unknown camera type: $CAMERA_TYPE (use 'realsense' or 'azure')"
    exit 1
fi

# === RUN NODE ===
echo "🚀 Launching Mediapipe node using $CAMERA_TYPE topics..."
ros2 launch hsar_bringup mediapipe_pose_node.launch.py \
    namespace:=mediapipe_pose_node \
    image_topic:=${IMAGE_TOPIC} \
    image_reliability:=2 \
    live_visualization:=True



# USAGE EXAMPLE
# ./mediapipe_pose_node.sh --camera azure
# ./mediapipe_pose_node.sh --camera realsense
