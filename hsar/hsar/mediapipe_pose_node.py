#!/usr/bin/env python3

import traceback
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.clock import Clock

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import mediapipe as mp

from hsar_msgs.msg import HumanPose3D


class MediaPipePoseNode(LifecycleNode):

    def __init__(self):
        super().__init__('mediapipe_pose_node')

        # Parameters
        # self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('image_reliability', QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter('live_visualization', True)

        self.bridge = None
        self.pose = None
        self.cam_header = None

    # ==========================================================
    # CONFIGURE
    # ==========================================================
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[mediapipe_pose_node] Configuring...')

            self.image_topic = self.get_parameter('image_topic').value
            self.reliability = self.get_parameter('image_reliability').value
            self.live_visualization = self.get_parameter('live_visualization').value

            self.qos = QoSProfile(
                reliability=self.reliability,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )

            self.bridge = CvBridge()

            self.pose_pub = self.create_lifecycle_publisher(HumanPose3D, 'human_pose_3d', 10)

            self.debug_pub = self.create_lifecycle_publisher(Image, 'mediapipe_pose_debug', 10)

            super().on_configure(state)
            self.get_logger().info('[mediapipe_pose_node] Configured!')
            
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[mediapipe_pose_node] Exception during configuration: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    # ==========================================================
    # ACTIVATE
    # ==========================================================
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[mediapipe_pose_node] Activating...')

            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils

            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, self.qos)
            
            super().on_activate(state)
            self.get_logger().info('[mediapipe_pose_node] Activated!')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[mediapipe_pose_node] Exception during activation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    # ==========================================================
    # DEACTIVATE
    # ==========================================================
    def on_deactivate(self, state: LifecycleState)-> TransitionCallbackReturn:
        try:
            self.get_logger().info('[mediapipe_pose_node] Deactivating...')
            
            if hasattr(self, 'image_sub'):
                self.destroy_subscription(self.image_sub)

            if hasattr(self, 'pose'):
                self.pose.close()
                del self.pose

            super().on_deactivate(state)
            self.get_logger().info('[mediapipe_pose_node] Deactivated!')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"[mediapipe_pose_node] Exception during deactivation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE
        
    
    # ==========================================================
    # CLEAN UP
    # ==========================================================
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[mediapipe_pose_node] Cleaning up...')
            
            # Destroy Publisher
            self.destroy_publisher(self.pose_pub)
            self.destroy_publisher(self.debug_pub)
            
            del self.qos
            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS
        
        except Exception as e:
            self.get_logger().error(f"[mediapipe_pose_node] Exception during cleanup: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE
        
    # ==========================================================
    # SHUT DOWN
    # ==========================================================
    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[mediapipe_pose_node] Shutting down...')
            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[mediapipe_pose_node] Exception during shutdown: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    # ==========================================================
    # IMAGE CALLBACK (MAIN LOGIC)
    # ==========================================================
    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.cam_header = msg.header
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        pose_msg = HumanPose3D()
        pose_msg.header.stamp = Clock().now().to_msg()
        pose_msg.header.frame_id = msg.header.frame_id

        if results.pose_world_landmarks:
            pts = []
            for lm in results.pose_world_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])
            pose_msg.landmarks = pts
            pose_msg.valid = True
        else:
            pose_msg.landmarks = []
            pose_msg.valid = False
        self.pose_pub.publish(pose_msg)

        if self.live_visualization:
            dbg = frame.copy()
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    dbg,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
            
            dbg_msg = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
            dbg_msg.header = pose_msg.header
            self.debug_pub.publish(dbg_msg)


def main():
    rclpy.init()
    node = MediaPipePoseNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
