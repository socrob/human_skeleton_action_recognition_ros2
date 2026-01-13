#!/usr/bin/env python3

import traceback
import rclpy
from rclpy.clock import Clock
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
import numpy as np
import torch, pickle
from collections import deque
from hsar_msgs.msg import HumanPosture
from hsar.hsar_models import PostureMLP, ArmStateMLP, PointingMLP, ArmRaisedDirectionMLP
from hsar_msgs.msg import HumanPose3D
from ament_index_python.packages import get_package_share_directory
import os


class DetectArmStateNode(LifecycleNode):

    def __init__(self):
        super().__init__('detect_arm_state_node')

        # Declare parameters
        self.declare_parameter('mp_topic', '/mediapipe_pose_node/human_pose_3d')
        
        # Models
        self.declare_parameter('arm_state_model_name', 'arm_state_mlp.pt')
        self.declare_parameter('arm_state_scaler_name', 'arm_state_scaler.pkl')
        
        self.declare_parameter('pointing_dir_model_name', 'pointing_mlp_direction.pt')
        self.declare_parameter('pointing_dir_scaler_name', 'pointing_scaler.pkl')
        
        self.declare_parameter('arm_raised_dir_model_name', 'arm_raised_direction_mlp.pt')
        self.declare_parameter('arm_raised_dir_scaler_name', 'arm_raised_direction_scaler.pkl')
        
        # Thresholds / Buffers
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('smooth_window', 15)
        self.declare_parameter('state_conf_threshold', 0.6)
        self.declare_parameter('dir_conf_threshold', 0.6)
        self.declare_parameter('wrist_hip_min_distance', 0.95)
        self.declare_parameter('wrist_shoulder_min_dy', 0.35)

        # Labels
        self.arm_state_labels = ["arm_down", "pointing", "arm_raised"]
        self.dir_labels = ["left", "right"]

        # MediaPipe landmark subset indices
        self.landmark_idxs = [11, 12, 13, 14, 15, 16, 23, 24]
        
        self.idx_ls, self.idx_rs = 0, 1
        self.idx_le, self.idx_re = 2, 3
        self.idx_lw, self.idx_rw = 4, 5
        self.idx_lh, self.idx_rh = 6, 7

        # Runtime state
        self.last_label = "None"
        
        # Pkg where models exist
        self.assets_dir = get_package_share_directory("hsar_assets")


    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_arm_state_node] Configuring...')

            # Load parameters
            self.mp_topic = self.get_parameter('mp_topic').get_parameter_value().string_value
            
            self.arm_state_model_name = self.get_parameter('arm_state_model_name').get_parameter_value().string_value
            self.arm_state_scaler_name = self.get_parameter('arm_state_scaler_name').get_parameter_value().string_value
            
            self.pointing_dir_model_name = self.get_parameter('pointing_dir_model_name').get_parameter_value().string_value
            self.pointing_dir_scaler_name = self.get_parameter('pointing_dir_scaler_name').get_parameter_value().string_value
            
            self.arm_raised_dir_model_name = self.get_parameter('arm_raised_dir_model_name').get_parameter_value().string_value
            self.arm_raised_dir_scaler_name = self.get_parameter('arm_raised_dir_scaler_name').get_parameter_value().string_value
            
            self.device = self.get_parameter('device').get_parameter_value().string_value
            self.smooth_window = self.get_parameter('smooth_window').get_parameter_value().integer_value
            self.state_conf_threshold = self.get_parameter('state_conf_threshold').get_parameter_value().bool_value
            self.dir_conf_threshold = self.get_parameter('dir_conf_threshold').get_parameter_value().double_value
            self.wrist_hip_min_distance = self.get_parameter('wrist_hip_min_distance').get_parameter_value().double_value
            self.wrist_shoulder_min_dy = self.get_parameter('wrist_shoulder_min_dy').get_parameter_value().double_value

            
            # Get models
            self.arm_state_model_path = os.path.join(
                self.assets_dir,
                "models/arm_state/version_1/" + self.arm_state_model_name
            )
            
            self.arm_state_scaler_path = os.path.join(
                self.assets_dir,
                "models/arm_state/version_1/" + self.arm_state_scaler_name
            )
            
            self.pointing_dir_model_path = os.path.join(
                self.assets_dir,
                "models/pointing_direction/version_2/" + self.pointing_dir_model_name
            )
            
            self.pointing_dir_scaler_path = os.path.join(
                self.assets_dir,
                "models/pointing_direction/version_2/" + self.pointing_dir_scaler_name
            )
            
            self.arm_raised_dir_model_path = os.path.join(
                self.assets_dir,
                "models/arm_raised_direction/version_1/" + self.arm_raised_dir_model_name
            )
            
            self.arm_raised_dir_scaler_path = os.path.join(
                self.assets_dir,
                "models/arm_raised_direction/version_1/" + self.arm_raised_dir_scaler_name
            )

            # Setup Torch device
            if self.device == "cuda":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Lifecycle Publisher
            self._pub = self.create_lifecycle_publisher(HumanPosture, "arm_state", 10)

            super().on_configure(state)
            self.get_logger().info('[detect_arm_state_node] Configured')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[detect_arm_state_node] Exception during configuration: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_arm_state_node] Activating...')

            # ============================================================
            # LOAD MODEL + SCALER
            # ============================================================
            # Arm state Model
            try:
                self.arm_state_model = ArmStateMLP().to(self.device)
                self.arm_state_model.load_state_dict(torch.load(self.arm_state_model_path, map_location=self.device))
                self.arm_state_model.eval()
                self.get_logger().info('[detect_arm_state_node] Arm state model loaded')
            except FileNotFoundError:
                self.get_logger().error(f"Model file '{self.arm_state_model_path}' does not exists")
                return TransitionCallbackReturn.ERROR

            try:
                with open(self.arm_state_scaler_path, "rb") as f:
                    self.arm_state_scaler = pickle.load(f)
                self.get_logger().info('[detect_arm_state_node] Arm state scaler loaded')
            except FileNotFoundError:
                self.get_logger().error(f"Scaler file '{self.arm_state_scaler_path}' does not exists")
                return TransitionCallbackReturn.ERROR


            # Pointing Direction Model
            try:
                self.pointing_dir_model = PointingMLP().to(self.device)
                self.pointing_dir_model.load_state_dict(torch.load(self.pointing_dir_model_path, map_location=self.device))
                self.pointing_dir_model.eval()
                self.get_logger().info('[detect_arm_state_node] Pointing direction model loaded')
            except FileNotFoundError:
                self.get_logger().error(f"Model file '{self.pointing_dir_model_path}' does not exists")
                return TransitionCallbackReturn.ERROR

            try:
                with open(self.pointing_dir_scaler_path, "rb") as f:
                    self.pointing_dir_scaler = pickle.load(f)
                self.get_logger().info('[detect_arm_state_node] Pointing direction scaler loaded')
            except FileNotFoundError:
                self.get_logger().error(f"Scaler file '{self.pointing_dir_scaler_path}' does not exists")
                return TransitionCallbackReturn.ERROR
            
            
            # Arm Raised Direction Model
            try:
                self.arm_raised_dir_model = ArmRaisedDirectionMLP().to(self.device)
                self.arm_raised_dir_model.load_state_dict(torch.load(self.arm_raised_dir_model_path, map_location=self.device))
                self.arm_raised_dir_model.eval()
                self.get_logger().info('[detect_arm_state_node] Arm raised direction model loaded')
            except FileNotFoundError:
                self.get_logger().error(f"Model file '{self.arm_raised_dir_model_path}' does not exists")
                return TransitionCallbackReturn.ERROR

            try:
                with open(self.arm_raised_dir_scaler_path, "rb") as f:
                    self.arm_raised_dir_scaler = pickle.load(f)
                self.get_logger().info('[detect_arm_state_node] Arm raised direction scaler loaded')
            except FileNotFoundError:
                self.get_logger().error(f"Scaler file '{self.arm_raised_dir_scaler_path}' does not exists")
                return TransitionCallbackReturn.ERROR
            
            
            # Mediapipe subscription
            self.mp_sub = self.create_subscription(HumanPose3D, self.mp_topic, self.mp_cb, 10)
            
            
            # ============================================================
            # VARIABLES
            # ============================================================
            self.state_buffer = deque(maxlen=self.smooth_window)
            self.last_label = "None"

            super().on_activate(state)
            self.get_logger().info('[detect_arm_state_node] Subscriptions activated')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[detect_arm_state_node] Exception during activation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_arm_state_node] Deactivating...')

            if hasattr(self, "arm_state_model"):
                del self.arm_state_model
                self.get_logger().info("Clearing CUDA cache")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if hasattr(self, "pointing_dir_model"):
                del self.pointing_dir_model
                self.get_logger().info("Clearing CUDA cache")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if hasattr(self, "arm_raised_dir_model"):
                del self.pointing_dir_model
                self.get_logger().info("Clearing CUDA cache")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if hasattr(self, "state_buffer"):
                del self.state_buffer

            if hasattr(self, "mp_sub"):
                self.destroy_subscription(self.mp_sub)

            super().on_deactivate(state)
            self.get_logger().info('[detect_arm_state_node] Deactivated')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[detect_arm_state_node] Exception during deactivation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_arm_state_node] Cleaning up...')

            if hasattr(self, "_pub"):
                self.destroy_publisher(self._pub)

            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[detect_arm_state_node] Exception during cleanup: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_arm_state_node] Shutting down...')
            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[detect_arm_state_node] Exception during shutdown: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def normalize_arm_pose(self, x24):
        k = x24.reshape(8, 3).astype(np.float32)

        center = 0.5 * (k[self.idx_ls] + k[self.idx_rs])
        k -= center

        v = k[self.idx_rs] - k[self.idx_ls]
        yaw = np.arctan2(v[2], v[0])
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        k = (R @ k.T).T

        scale = np.linalg.norm(k[self.idx_rs] - k[self.idx_ls])
        if scale < 1e-6 or not np.isfinite(scale):
            return None

        return (k / scale).reshape(-1)
    
    def arm_extended(self, x):
        k = x.reshape(8, 3)
        dl = np.linalg.norm(k[self.idx_lw] - k[self.idx_lh])
        dr = np.linalg.norm(k[self.idx_rw] - k[self.idx_rh])
        return max(dl, dr) > self.wrist_hip_min_distance


    def arm_raised(self, x):
        k = x.reshape(8, 3)
        dy_l = abs(k[self.idx_lw][1] - k[self.idx_ls][1])
        dy_r = abs(k[self.idx_rw][1] - k[self.idx_rs][1])
        return max(dy_l, dy_r) > self.wrist_shoulder_min_dy
    

    def predict_arm_state(self, x_norm):
        xs = self.arm_state_scaler.transform(x_norm.reshape(1, -1))
        x_tensor = torch.tensor(xs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.arm_state_model(x_tensor), dim=1)[0].cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        return pred, conf, probs
    
    
    def predict_direction(self, x_norm, model, scaler):
        xs = scaler.transform(x_norm.reshape(1, -1))
        x_tensor = torch.tensor(xs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(model(x_tensor), dim=1)[0].cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        return pred, conf, probs


    def extract_arm_subset(self, landmarks_99):
        pts = []
        for idx in self.landmark_idxs:
            base = 3 * idx
            pts.extend(landmarks_99[base:base + 3])
        return np.array(pts, dtype=np.float32)


    def mp_cb(self, msg: HumanPose3D):
        probs = []
        if not msg.valid:
            self.state_buffer.clear()
            self.last_label = "unknown"
            final_label = "unknown"
            confidence = 0.0
            probs = []
        else:
            pose_vec_24 = self.extract_arm_subset(msg.landmarks)
            x_norm = self.normalize_arm_pose(pose_vec_24)

            if x_norm is None:
                final_label = "unknown"
                confidence = 0.0
                probs = []
            else:
                # ----------------------------------------------
                # 1) Arm state prediction
                # ----------------------------------------------
                pred, conf, _ = self.predict_arm_state(x_norm)

                if conf >= self.state_conf_threshold:
                    self.state_buffer.append((pred, conf))

                if len(self.state_buffer) == 0:
                    final_label = "unknown"
                    confidence = 0.0
                    probs = []
                else:
                    preds = [p for p, _ in self.state_buffer]
                    majority = max(set(preds), key=preds.count)
                    self.last_label = self.arm_state_labels[majority]

                    confs = [c for p, c in self.state_buffer if p == majority]
                    confidence = float(np.mean(confs))
                    final_label = self.last_label

                    # ----------------------------------------------
                    # 2) Pointing refinement
                    # ----------------------------------------------
                    if final_label == "pointing":
                        if self.arm_extended(x_norm):
                            d, d_conf, d_probs = self.predict_direction(
                                x_norm,
                                self.pointing_dir_model,
                                self.pointing_dir_scaler
                            )
                            if d_conf >= self.dir_conf_threshold:
                                final_label = f"pointing_{self.dir_labels[d]}"
                            else:
                                final_label = "unknown"
                        else:
                            final_label = "arm_down"

                    # ----------------------------------------------
                    # 3) Arm-raised refinement
                    # ----------------------------------------------
                    elif final_label == "arm_raised":
                        if self.arm_raised(x_norm):
                            d, d_conf, d_probs = self.predict_direction(
                                x_norm,
                                self.arm_raised_dir_model,
                                self.arm_raised_dir_scaler
                            )
                            if d_conf >= self.dir_conf_threshold:
                                final_label = f"arm_raised_{self.dir_labels[d]}"
                            else:
                                final_label = "unknown"
                        else:
                            final_label = "unknown"

        # --------------------------------------------------
        # Publish result
        # --------------------------------------------------
        out = HumanPosture()
        out.header.stamp = Clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id
        out.posture = final_label
        out.confidence = confidence
        out.probabilities = probs

        self._pub.publish(out)




def main():
    rclpy.init()
    node = DetectArmStateNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
