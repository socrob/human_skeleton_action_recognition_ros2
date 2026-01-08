#!/usr/bin/env python3

import traceback
import rclpy
from rclpy.clock import Clock
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hsar_msgs.msg import HumanPosture
import cv2
import numpy as np
import torch, pickle
import mediapipe as mp
from collections import deque
from hsar_models import PostureMLP


class DetectPostureNode(LifecycleNode):

    def __init__(self):
        super().__init__('detect_posture_node')

        # Declare parameters
        self.declare_parameter('node_rate', 10)
        self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('model_path', '../models/posture/version_3/posture_mlp_merged_norm.pt')
        self.declare_parameter('scaler_path', '../models/posture/version_3/posture_scaler_mereged_norm.pkl')
        self.declare_parameter('use_geom_norm', True)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('smooth_window', 15)
        self.declare_parameter('conf_threshold', 0.6)
        self.declare_parameter('image_reliability', QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter('live_visualization', False)

        self.labels = ["standing", "sitting", "lying"]

        # MediaPipe landmark indices
        self.left_shoulder = 11
        self.right_shoulder = 12
        self.left_hip = 23
        self.right_hip = 24

        # Runtime state
        self.received_img = False
        self.frame = None
        self.cam_header = None
        self.last_label = "None"

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_posture_node] Configuring...')

            # Load parameters (typed)
            self.node_rate = self.get_parameter('node_rate').get_parameter_value().integer_value
            self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
            self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
            self.scaler_path = self.get_parameter('scaler_path').get_parameter_value().string_value
            self.use_geom_norm = self.get_parameter('use_geom_norm').get_parameter_value().bool_value
            self.device = self.get_parameter('device').get_parameter_value().string_value
            self.smooth_window = self.get_parameter('smooth_window').get_parameter_value().integer_value
            self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
            self.reliability = self.get_parameter('image_reliability').get_parameter_value().integer_value
            self.live_visualization = self.get_parameter('live_visualization').get_parameter_value().bool_value

            self.get_logger().info(f'[detect_posture_node] Using model: {self.model_path}')

            # Setup QoS
            self.qos = QoSProfile(
                reliability=self.reliability,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1
            )

            self.cv_bridge = CvBridge()

            # Setup Torch device
            if self.device == "cuda":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Lifecycle Publisher
            self._pub = self.create_lifecycle_publisher(HumanPosture, "posture", 10)
            self._img_pub = self.create_lifecycle_publisher(Image, "posture_debug", 10)

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

            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, self.qos)

            # ============================================================
            # MEDIAPIPE SETUP
            # ============================================================
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils

            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # ============================================================
            # NODE RATE
            # ============================================================
            self.timer = self.create_timer(1.0 / float(self.node_rate), self.run)

            # ============================================================
            # VARIABLES
            # ============================================================
            self.pred_buffer = deque(maxlen=self.smooth_window)
            self.last_label = "None"
            self.received_img = False
            self.frame = None
            self.cam_header = None

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

            if hasattr(self, "pose"):
                self.pose.close()
                del self.pose

            if hasattr(self, "mp_drawing"):
                del self.mp_drawing
            if hasattr(self, "mp_pose"):
                del self.mp_pose

            if hasattr(self, "pred_buffer"):
                del self.pred_buffer

            if hasattr(self, "image_sub"):
                self.destroy_subscription(self.image_sub)

            if hasattr(self, 'timer'):
                self.timer.cancel()
                del self.timer

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
            if hasattr(self, "_img_pub"):
                self.destroy_publisher(self._img_pub)

            if hasattr(self, "qos"):
                del self.qos

            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[detect_posture_node] Exception during cleanup: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[detect_posture_node] Shutting down...')
            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[detect_posture_node] Exception during shutdown: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def extract_3d_pose(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_world_landmarks:
            return None, None
        pts = []
        for lm in results.pose_world_landmarks.landmark[:33]:
            pts.extend([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float32), results

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

    def image_cb(self, msg: Image):
        self.frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.cam_header = msg.header
        self.received_img = True

    def run(self):
        if not self.received_img:
            return
        
        if self.frame is None or self.cam_header is None:
            self.received_img = False
            return

        now = Clock().now().to_msg()  # always defined for debug publishing
        last_label = self.last_label  # default to previous (prevents undefined variable)

        pose_vec, results = self.extract_3d_pose(self.frame)

        probs = None
        conf = 0.0

        if pose_vec is not None:
            pred, conf, probs = self.predict_posture(pose_vec, self.use_geom_norm)

            if pred is not None and conf >= self.conf_threshold:
                self.pred_buffer.append(pred)

            if len(self.pred_buffer) > 0:
                majority = max(set(self.pred_buffer), key=self.pred_buffer.count)
                last_label = self.labels[majority]
                self.last_label = last_label

                posture_msg = HumanPosture()
                posture_msg.header.stamp = now
                posture_msg.header.frame_id = self.cam_header.frame_id
                posture_msg.id = 1
                posture_msg.posture = last_label
                posture_msg.confidence = conf
                posture_msg.probabilities = [float(probs[0]), float(probs[1]), float(probs[2])]
                self._pub.publish(posture_msg)
        else:
            # If no pose, avoid “sticky” buffered posture
            self.pred_buffer.clear()

        if self.live_visualization:
            debug_img = self.frame.copy()

            cv2.putText(
                debug_img,
                f"Posture: {last_label}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2
            )

            if probs is not None:
                y0 = 80
                for i, lbl in enumerate(self.labels):
                    txt = f"{lbl}: {probs[i]:.2f}"
                    cv2.putText(
                        debug_img,
                        txt,
                        (30, y0 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

            if results and results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    debug_img,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
                )

            debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header.stamp = now
            debug_msg.header.frame_id = self.cam_header.frame_id
            self._img_pub.publish(debug_msg)

        self.received_img = False


def main():
    rclpy.init()
    node = DetectPostureNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
