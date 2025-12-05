#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1B: Bad Fruit Detection and TF Publishing
- Detects greyish-white (bad/spoiled) fruits on conveyor belt
- Publishes TF transforms: <TEAM_ID>_bad_fruit_<id>
- Visualizes with contours, bounding boxes, and labels
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf_transformations import quaternion_matrix, quaternion_from_euler

# ==================== USER CONFIGURATION ====================
TEAM_ID = 3251  # ⚠️ CHANGE THIS to your actual team ID (mandatory)
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False

# Detection parameters
DEPTH_MEDIAN_KERNEL = 5     # Median filter size for depth smoothing
MIN_CONTOUR_AREA = 600      # Minimum area to filter noise
MAX_DEPTH = 2.0             # Maximum depth in meters (ignore far objects)
MIN_DEPTH = 0.01            # Minimum valid depth

# Camera intrinsics (from boilerplate - verify these match your setup)
sizeCamX = 1280
sizeCamY = 720
centerCamX = 642.724365234375
centerCamY = 361.9780578613281
focalX = 915.3003540039062
focalY = 914.0320434570312
# ===========================================================


class FruitsTF(Node):
    """
    ROS2 Node for detecting bad fruits and publishing TF transforms.
    """
    
    def __init__(self):
        super().__init__('fruits_tf_node')
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        
        # Callback group handling
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()
        
        # Subscribe to multiple topic names for robustness
        self.create_subscription(
            Image, '/camera/color/image_raw', 
            self.colorimagecb, 10, callback_group=self.cb_group
        )
        self.create_subscription(
            Image, '/camera/image_raw', 
            self.colorimagecb, 10, callback_group=self.cb_group
        )
        self.create_subscription(
            Image, '/camera/depth/image_rect_raw', 
            self.depthimagecb, 10, callback_group=self.cb_group
        )
        self.create_subscription(
            Image, '/camera/depth/image_raw', 
            self.depthimagecb, 10, callback_group=self.cb_group
        )
        
        # TF2 broadcaster and buffer
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Timer for processing at 5 Hz
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)
        
        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)
        
        self.get_logger().info(f"FruitsTF node started (Team ID: {TEAM_ID})")

    # ==================== CALLBACKS ====================
    
    def depthimagecb(self, data: Image):
        """
        Callback for depth image topic.
        Converts ROS Image to numpy array.
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warning(f"Depth conversion failed: {e}")
            self.depth_image = None

    def colorimagecb(self, data: Image):
        """
        Callback for color image topic.
        Converts ROS Image to BGR format.
        """
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            
            # Uncomment if camera orientation needs adjustment
            # self.cv_image = cv2.flip(self.cv_image, 0)  # Vertical flip
            # self.cv_image = cv2.flip(self.cv_image, 1)  # Horizontal flip
            # self.cv_image = cv2.rotate(self.cv_image, cv2.ROTATE_180)
            
        except Exception as e:
            self.get_logger().warning(f"Color conversion failed: {e}")
            self.cv_image = None

    # ==================== DETECTION ====================
    
    def bad_fruit_detection(self, rgb_image):
        """
        Detects bad (greyish-white) fruits in the image.
        
        Args:
            rgb_image: BGR image from camera
            
        Returns:
            list: List of dictionaries with fruit information:
                  - 'center': (x, y) pixel coordinates
                  - 'distance': depth in meters (filled later)
                  - 'angle': orientation angle
                  - 'width': bounding box width
                  - 'id': sequential fruit ID
                  - 'contour': OpenCV contour
                  - 'bbox': (x, y, w, h) bounding rectangle
        """
        bad_fruits = []
        
        if rgb_image is None:
            return bad_fruits
        
        # Convert to HSV for color-based segmentation
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Define HSV range for greyish-white (bad) fruits
        # Bad fruits: low saturation, medium-high value
        lower_bad = np.array([0, 0, 120])      # Any hue, very low saturation
        upper_bad = np.array([179, 80, 255])   # Up to S=80 for grey-white
        
        # Create mask for bad fruits
        mask_bad = cv2.inRange(hsv, lower_bad, upper_bad)
        
        # Remove reddish-pink (good) fruits from mask
        # Red wraps around in HSV (0-10 and 160-179)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([179, 255, 255])
        
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        
        # Subtract red regions from bad fruit mask
        mask_bad = cv2.bitwise_and(mask_bad, cv2.bitwise_not(mask_red))
        
        # Morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask_bad = cv2.morphologyEx(mask_bad, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_bad = cv2.morphologyEx(mask_bad, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask_bad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and process each
        fruit_id = 1
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < MIN_CONTOUR_AREA:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center using moments (more accurate than bbox center)
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            
            # Calculate orientation angle (optional)
            angle = 0.0
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]
                except:
                    pass
            
            # Store fruit information
            fruit_info = {
                'center': (cX, cY),
                'distance': None,  # Will be filled in process_image
                'angle': angle,
                'width': w,
                'id': fruit_id,
                'contour': contour,
                'bbox': (x, y, w, h)
            }
            
            bad_fruits.append(fruit_info)
            fruit_id += 1
        
        return bad_fruits

    # ==================== DEPTH READING ====================
    
    def get_depth_at_point(self, x, y):
        """
        Get depth value at point (x, y) using median filtering.
        
        Args:
            x, y: Pixel coordinates
            
        Returns:
            float: Depth in meters, or None if invalid
        """
        if self.depth_image is None:
            return None
        
        try:
            h, w = self.depth_image.shape[:2]
            
            # Clamp coordinates to image bounds
            px = np.clip(x, 0, w - 1)
            py = np.clip(y, 0, h - 1)
            
            # Extract patch around point for median filtering
            k = DEPTH_MEDIAN_KERNEL // 2
            x0 = max(0, px - k)
            x1 = min(w - 1, px + k)
            y0 = max(0, py - k)
            y1 = min(h - 1, py + k)
            
            patch = self.depth_image[y0:y1 + 1, x0:x1 + 1]
            
            if patch.size == 0:
                return None
            
            # Convert to meters based on dtype
            if patch.dtype == np.uint16:
                # Depth in millimeters (common for RealSense)
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0] = np.nan
                depth_m = float(np.nanmedian(patch_f)) / 1000.0
            elif patch.dtype in [np.float32, np.float64]:
                # Already in meters
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0.0] = np.nan
                depth_m = float(np.nanmedian(patch_f))
            else:
                # Fallback
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0] = np.nan
                depth_m = float(np.nanmedian(patch_f))
            
            # Validate depth
            if math.isnan(depth_m) or depth_m < MIN_DEPTH or depth_m > MAX_DEPTH:
                return None
            
            return depth_m
            
        except Exception as e:
            self.get_logger().warning(f"Depth read error at ({x},{y}): {e}")
            return None

    # ==================== PROCESSING & TF PUBLISHING ====================
    
    def process_image(self):
        """
        Main processing loop: detect fruits, compute poses, publish TFs.
        """
        if self.cv_image is None:
            return
        
        # Create visualization image
        display_img = self.cv_image.copy()
        
        # Detect bad fruits
        bad_fruits = self.bad_fruit_detection(self.cv_image)
        
        # Process each detected fruit
        for fruit in bad_fruits:
            cX, cY = fruit['center']
            fruit_id = fruit['id']
            contour = fruit['contour']
            bbox = fruit['bbox']
            x, y, w, h = bbox
            
            # Get depth at fruit center
            depth_m = self.get_depth_at_point(cX, cY)
            
            if depth_m is None:
                # Still visualize but don't publish TF
                cv2.drawContours(display_img, [contour], -1, (0, 165, 255), 2)  # Orange
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(display_img, "bad_fruit (no depth)", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                continue
            
            fruit['distance'] = depth_m
            
            # Compute 3D position in camera frame using camera intrinsics
            x_cam = depth_m * (sizeCamX - cX - centerCamX) / focalX
            y_cam = depth_m * (sizeCamY - cY - centerCamY) / focalY
            z_cam = depth_m
            
            # Draw visualization (green contour, red center, blue bbox)
            cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)  # Green contour
            cv2.circle(display_img, (cX, cY), 5, (0, 0, 255), -1)  # Red center
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bbox
            cv2.putText(display_img, "bad_fruit", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(display_img, f"ID:{fruit_id} D:{depth_m:.2f}m", (cX - 30, cY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Publish intermediate TF: camera_link -> cam_<id>
            cam_child = f"cam_{fruit_id}"
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = 'camera_link'
            t_cam.child_frame_id = cam_child
            t_cam.transform.translation.x = x_cam
            t_cam.transform.translation.y = y_cam
            t_cam.transform.translation.z = z_cam
            
            # Identity quaternion (no rotation)
            q_identity = quaternion_from_euler(0, 0, 0)
            t_cam.transform.rotation.x = q_identity[0]
            t_cam.transform.rotation.y = q_identity[1]
            t_cam.transform.rotation.z = q_identity[2]
            t_cam.transform.rotation.w = q_identity[3]
            
            try:
                self.tf_broadcaster.sendTransform(t_cam)
            except Exception as e:
                self.get_logger().warning(f"Failed to broadcast camera->cam_{fruit_id}: {e}")
                continue
            
            # Lookup transform: base_link -> camera_link
            try:
                trans = self.tf_buffer.lookup_transform(
                    target_frame='base_link',
                    source_frame='camera_link',
                    time=rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                
                # Extract rotation quaternion and translation
                q_base_cam = (
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w
                )
                t_base_cam = np.array([
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z
                ])
                
                # Build 4x4 transformation matrix: base_link <- camera_link
                T_base_cam = quaternion_matrix(q_base_cam)
                T_base_cam[0:3, 3] = t_base_cam
                
                # Transform point from camera frame to base frame
                p_cam_homogeneous = np.array([x_cam, y_cam, z_cam, 1.0])
                p_base_homogeneous = T_base_cam.dot(p_cam_homogeneous)
                p_base = p_base_homogeneous[:3]
                
                # Publish final TF: base_link -> <TEAM_ID>_bad_fruit_<id>
                child_name = f"{TEAM_ID}_bad_fruit_{fruit_id}"
                t_base_obj = TransformStamped()
                t_base_obj.header.stamp = self.get_clock().now().to_msg()
                t_base_obj.header.frame_id = 'base_link'
                t_base_obj.child_frame_id = child_name
                t_base_obj.transform.translation.x = float(p_base[0])
                t_base_obj.transform.translation.y = float(p_base[1])
                t_base_obj.transform.translation.z = float(p_base[2])
                t_base_obj.transform.rotation.x = q_base_cam[0]
                t_base_obj.transform.rotation.y = q_base_cam[1]
                t_base_obj.transform.rotation.z = q_base_cam[2]
                t_base_obj.transform.rotation.w = q_base_cam[3]
                
                self.tf_broadcaster.sendTransform(t_base_obj)
                
                self.get_logger().info(
                    f"Published {child_name}: base({p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f})"
                )
                
            except Exception as e:
                self.get_logger().warning(
                    f"Could not compute base_link transform for fruit {fruit_id}: {e}"
                )
        
        # Display the visualization
        if SHOW_IMAGE:
            try:
                cv2.imshow('fruits_tf_view', display_img)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warning(f"OpenCV display error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF node")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
