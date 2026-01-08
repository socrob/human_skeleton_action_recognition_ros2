# Version 1
## Dataset info
```
Saved: posture_dataset.pkl
Samples: 127
Skipped bad filename: 0
Skipped by label:     276
Skipped no pose:      0
X shape: (127, 99)  (N, 99)
Label distribution:
  0 (standing): 58
  1 (sitting): 44
  2 (lying): 25
```

- Trained under my dataset for sitting, standing, and layed down.
- No normalization was applied.

## Procedure

I ran extract_mediapipe_3d_from_video.py to generate the pickle file, ran the train_mlp.py to generate the model posture_mlp.pt and the scalar posture_scaler.pkl, and final ran demo_realsense.py to test inference on the model.

In the beginning, I also used test_mediapipe_3d_average_pose.py to plot the 3D poses as outputs of Mediapipe.