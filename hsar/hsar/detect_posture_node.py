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
from hsar.hsar_models import PostureMLP
from hsar_msgs.msg import HumanPose3D
from ament_index_python.packages import get_package_share_directory
import os


class DetectPostureNode(LifecycleNode):

    def __init__(self):
        super().__init__('detect_posture_node')

        # Declare parameters
        self.declare_parameter('mp_topic', '/mediapipe_pose_node/human_pose_3d')
        self.declare_parameter('model_name', 'posture_mlp_merged_norm.pt')
        self.declare_parameter('model_scaler_name', 'posture_scaler_mereged_norm.pkl')
        self.declare_parameter('use_geom_norm', True)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('smooth_window', 15)
        self.declare_parameter('conf_threshold', 0.6)

        self.labels = ["standing", "sitting", "lying"]

        # MediaPipe landmark indices
        self.left_shoulder = 11
        self.right_shoulder = 12
        self.left_hip = 23
        self.right_hip = 24

        # Runtime state
        self.last_label = "None"
        
        # Pkg where models exist
        self.assets_dir = get_package_share_directory("hsar_assets")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_posture_node] Configuring...')

            # Load parameters (typed)
            self.mp_topic = self.get_parameter('mp_topic').get_parameter_value().string_value
            self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
            self.model_scaler_name = self.get_parameter('model_scaler_name').get_parameter_value().string_value
            self.use_geom_norm = self.get_parameter('use_geom_norm').get_parameter_value().bool_value
            self.device = self.get_parameter('device').get_parameter_value().string_value
            self.smooth_window = self.get_parameter('smooth_window').get_parameter_value().integer_value
            self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value

            self.get_logger().info(f'[detect_posture_node] Using model: {self.model_name}')
            
            # Get models
            self.model_path = os.path.join(
                self.assets_dir,
                "models/posture/version_3/" + self.model_name
            )
            
            self.scaler_path = os.path.join(
                self.assets_dir,
                "models/posture/version_3/" + self.model_scaler_name
            )

            # Setup Torch device
            if self.device == "cuda":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Lifecycle Publisher
            self._pub = self.create_lifecycle_publisher(HumanPosture, "posture", 10)

            super().on_configure(state)
            self.get_logger().info('[detect_posture_node] Configured')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[detect_posture_node] Exception during configuration: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_posture_node] Activating...')

            # ============================================================
            # LOAD MODEL + SCALER
            # ============================================================
            try:
                self.model = PostureMLP().to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                self.get_logger().info('[detect_posture_node] Posture model loaded')
            except FileNotFoundError:
                self.get_logger().error(f"Model file '{self.model_path}' does not exists")
                return TransitionCallbackReturn.ERROR

            try:
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.get_logger().info('[detect_posture_node] Posture scaler loaded')
            except FileNotFoundError:
                self.get_logger().error(f"Scaler file '{self.scaler_path}' does not exists")
                return TransitionCallbackReturn.ERROR

            self.mp_sub = self.create_subscription(HumanPose3D, self.mp_topic, self.mp_cb, 10)

            # ============================================================
            # VARIABLES
            # ============================================================
            self.pred_buffer = deque(maxlen=self.smooth_window)
            self.last_label = "None"

            super().on_activate(state)
            self.get_logger().info('[detect_posture_node] Subscriptions activated')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[detect_posture_node] Exception during activation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_posture_node] Deactivating...')

            if hasattr(self, "model"):
                del self.model
                self.get_logger().info("Clearing CUDA cache")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if hasattr(self, "pred_buffer"):
                del self.pred_buffer

            if hasattr(self, "mp_sub"):
                self.destroy_subscription(self.mp_sub)

            super().on_deactivate(state)
            self.get_logger().info('[detect_posture_node] Deactivated')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[detect_posture_node] Exception during deactivation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_posture_node] Cleaning up...')

            if hasattr(self, "_pub"):
                self.destroy_publisher(self._pub)

            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[detect_posture_node] Exception during cleanup: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_posture_node] Shutting down...')
            super().on_shutdown(state)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[detect_posture_node] Exception during shutdown: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def normalize_pose_world(self, pose_99):
        pts = pose_99.reshape(33, 3)

        pelvis = (pts[self.left_hip] + pts[self.right_hip]) / 2.0
        pts = pts - pelvis

        shoulder_center = (pts[self.left_shoulder] + pts[self.right_shoulder]) / 2.0
        torso_len = np.linalg.norm(shoulder_center - np.zeros(3))

        if torso_len < 1e-6:
            return None

        pts = pts / torso_len
        return pts.reshape(-1)

    def predict_posture(self, x_99, use_geom_norm):
        if use_geom_norm:
            x_99 = self.normalize_pose_world(x_99)
            if x_99 is None:
                return None, 0.0, None

        x_scaled = self.scaler.transform(x_99.reshape(1, -1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred])

        return pred, conf, probs

    def mp_cb(self, msg: HumanPose3D):
        if msg.valid:
            last_label = self.last_label
            pose_vec = np.array(msg.landmarks, dtype=np.float32)

            pred, conf, probs = self.predict_posture(pose_vec, self.use_geom_norm)

            if pred is not None and conf >= self.conf_threshold:
                self.pred_buffer.append((pred, conf))

            if len(self.pred_buffer) > 0:
                preds = [p for p, _ in self.pred_buffer]
                majority = max(set(preds), key=preds.count)
                last_label = self.labels[majority]
                self.last_label = last_label

                # confidence = mean confidence of majority class
                confs = [c for p, c in self.pred_buffer if p == majority]
                confidence = float(np.mean(confs))
                probabilities = [float(p) for p in probs]

            else:
                confidence = 0.0
                probabilities = [0.0, 0.0, 0.0]

        else:
            self.pred_buffer.clear()
            self.last_label = "None"
            last_label = "None"
            confidence = 0.0
            probabilities = [0.0, 0.0, 0.0]

        posture_msg = HumanPosture()
        posture_msg.header.stamp = Clock().now().to_msg()
        posture_msg.header.frame_id = msg.header.frame_id
        posture_msg.posture = last_label
        posture_msg.confidence = confidence
        posture_msg.probabilities = probabilities

        self._pub.publish(posture_msg)



def main():
    rclpy.init()
    node = DetectPostureNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
