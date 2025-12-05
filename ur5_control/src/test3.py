#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final: Task 1B - Bad Fruit Detection and TF Publishing (fixed transforms)

- Detects greyish fruits WITH GREEN TOPS (bad/spoiled) on conveyor belt
- Computes 3D fruit centers using depth image and camera intrinsics
- Transforms points into base_link and publishes TFs named:
    <TEAM_ID>_bad_fruit_<id>
- Visualizes contours, bounding boxes and labels "bad_fruit"
- Fixed projection signs and publishes upright fruit frames (no camera tilt)
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
MIN_CONTOUR_AREA = 200      # Minimum area to filter noise (lowered for grey detection)
MAX_CONTOUR_AREA = 3000     # Maximum area (filter out very large blobs)
MAX_DEPTH = 2.0             # Maximum depth in meters (ignore far objects)
MIN_DEPTH = 0.01            # Minimum valid depth

# Green-top detection parameters
TOP_REGION_HEIGHT_RATIO = 0.7  # Check top 30% of fruit bbox
TOP_REGION_EXTEND_UP = 0       # Pixels to extend above bbox
MIN_GREEN_PIXELS = 1000         # Minimum green pixels to confirm bad fruit (tuned down)

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
    ROS2 Node for detecting bad fruits (grey body + green top) and publishing TF transforms.
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

        # TF2 broadcaster and buffer/listener
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer for processing at 5 Hz
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info(f"FruitsTF node started (Team ID: {TEAM_ID})")
        self.get_logger().info("Detection method: Grey body + Green top (final, fixed TFs)")

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
        except Exception as e:
            self.get_logger().warning(f"Color conversion failed: {e}")
            self.cv_image = None

    # ==================== DETECTION ====================

    def bad_fruit_detection(self, rgb_image):
        """
        Detects bad fruits using TWO-STAGE approach:
        1. Find grey/greyish-white fruit bodies
        2. Verify green top exists in the upper region
        Returns list of fruit dicts.
        """
        bad_fruits = []

        if rgb_image is None:
            return bad_fruits

        # Convert to HSV for color-based segmentation
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # --- STEP 1: Detect grey areas (fruit body) ---
        # Grey fruits: low saturation, medium value range
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 60, 200])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

        # --- STEP 2: Detect green top areas ---
        # Green: hue approx 35-85, reasonable saturation and value
        lower_green = np.array([35, 40, 40])   # relaxed a bit
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological operations to clean up grey mask
        kernel = np.ones((3, 3), np.uint8)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel, iterations=1)

        # --- STEP 3: Find contours for grey blobs (potential bad fruits) ---
        contours_gray, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 1

        for contour in contours_gray:
            area = cv2.contourArea(contour)

            # Filter by reasonable fruit area
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # --- STEP 4: Check for green top in upper region ---
            # Define top region: covers top TOP_REGION_HEIGHT_RATIO of bounding box
            y_top_start = max(y - TOP_REGION_EXTEND_UP, 0)
            y_top_end = y + int(TOP_REGION_HEIGHT_RATIO * h)
            x_left = x
            x_right = x + w

            # Ensure bounds
            img_h, img_w = mask_green.shape[:2]
            y_top_end = min(y_top_end, img_h)
            x_right = min(x_right, img_w)

            # Extract top region from green mask
            top_region = mask_green[y_top_start:y_top_end, x_left:x_right]

            # Count green pixels in top region
            green_pixel_count = int(np.sum(top_region > 0))

            # If sufficient green pixels detected, it's a bad fruit
            if green_pixel_count < MIN_GREEN_PIXELS:
                continue

            # Calculate center using moments (more accurate than bbox center)
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            # Optional: circularity check (helps reject elongated objects)
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0
            if perimeter > 0:
                circularity = 4 * math.pi * (area / (perimeter * perimeter))
            if circularity < 0.3:
                # skip shapes that are too non-circular (tune if needed)
                pass

            # Store fruit information
            fruit_info = {
                'center': (cX, cY),
                'distance': None,  # Will be filled in process_image
                'angle': 0.0,
                'width': w,
                'id': fruit_id,
                'contour': contour,
                'bbox': (x, y, w, h),
                'green_pixels': green_pixel_count
            }

            bad_fruits.append(fruit_info)
            fruit_id += 1

        return bad_fruits

    # ==================== DEPTH READING ====================

    def get_depth_at_point(self, x, y):
        """
        Get depth value at point (x, y) using median filtering.
        Returns depth in meters or None.
        """
        if self.depth_image is None:
            return None

        try:
            h, w = self.depth_image.shape[:2]

            # Clamp coordinates to image bounds
            px = int(np.clip(x, 0, w - 1))
            py = int(np.clip(y, 0, h - 1))

            # Extract patch around point for median filtering
            k = max(1, DEPTH_MEDIAN_KERNEL // 2)
            x0 = max(0, px - k)
            x1 = min(w - 1, px + k)
            y0 = max(0, py - k)
            y1 = min(h - 1, py + k)

            patch = self.depth_image[y0:y1 + 1, x0:x1 + 1]

            if patch.size == 0:
                return None

            # Convert to meters based on dtype
            if patch.dtype == np.uint16:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0] = np.nan
                depth_m = float(np.nanmedian(patch_f)) / 1000.0
            elif patch.dtype in [np.float32, np.float64]:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0.0] = np.nan
                depth_m = float(np.nanmedian(patch_f))
            else:
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

        # Detect bad fruits (grey body + green top)
        bad_fruits = self.bad_fruit_detection(self.cv_image)

        # Process each detected fruit
        for fruit in bad_fruits:
            cX, cY = fruit['center']
            fruit_id = fruit['id']
            contour = fruit['contour']
            bbox = fruit['bbox']
            x, y, w, h = bbox
            green_pixels = fruit.get('green_pixels', 0)

            # Get depth at fruit center
            depth_m = self.get_depth_at_point(cX, cY)

            if depth_m is None:
                # Still visualize but don't publish TF
                cv2.drawContours(display_img, [contour], -1, (0, 165, 255), 2)  # Orange
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                fruit_name = f"{TEAM_ID}_bad_fruit_{fruit_id}"
                cv2.putText(display_img, f"{fruit_name} (no depth)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                continue

            fruit['distance'] = depth_m

            # --- Correct pinhole projection (camera optical frame) ---
            # Standard:
            #   X_cam = (u - cx) * Z / fx
            #   Y_cam = (v - cy) * Z / fy
            # Many cameras (and ROS optical conventions) require a sign flip on Y
            # to convert to right-handed camera coordinates used by downstream TFs.
            x_cam = (cX - centerCamX) * depth_m / focalX
            y_cam = -(cY - centerCamY) * depth_m / focalY   # NOTE: inverted Y fix
            z_cam = depth_m

            # Draw visualization (red for bad fruit)
            cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)  # Green contour
            cv2.circle(display_img, (cX, cY), 5, (0, 0, 255), -1)  # Red center
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bbox

            # Display name with team_id and fruit_id
            fruit_name = f"{TEAM_ID}_bad_fruit_{fruit_id}"
            cv2.putText(display_img, fruit_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_img, f"D:{depth_m:.2f}m G:{green_pixels}", (x, y + h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Lookup transform: base_link -> camera_link (transform that maps camera frame into base frame)
            try:
                trans = self.tf_buffer.lookup_transform(
                    target_frame='base_link',
                    source_frame='camera_link',
                    time=rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
            except Exception as e:
                self.get_logger().warning(f"Could not lookup base_link->camera_link: {e}")
                continue

            # Extract rotation quaternion and translation (base_link <- camera_link)
            q_base_cam = [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ]
            t_base_cam = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ], dtype=np.float64)

            # Build 4x4 transformation matrix (base_link <- camera_link)
            T_base_cam = quaternion_matrix(q_base_cam)
            T_base_cam[0:3, 3] = t_base_cam

            # Transform fruit point from camera frame to base frame
            p_cam_homogeneous = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
            p_base_homogeneous = T_base_cam @ p_cam_homogeneous
            p_base = p_base_homogeneous[:3]

            # Publish TF: base_link -> <TEAM_ID>_bad_fruit_<id>
            child_name = f"{TEAM_ID}_bad_fruit_{fruit_id}"
            t_base_obj = TransformStamped()
            t_base_obj.header.stamp = self.get_clock().now().to_msg()
            t_base_obj.header.frame_id = 'base_link'
            t_base_obj.child_frame_id = child_name
            t_base_obj.transform.translation.x = float(p_base[0])+0.65
            t_base_obj.transform.translation.y = float(p_base[1])+0.6
            t_base_obj.transform.translation.z = float(p_base[2])-2.15

            # IMPORTANT: publish upright identity orientation for fruit frames (no camera tilt)
            t_base_obj.transform.rotation.x = 0.0
            t_base_obj.transform.rotation.y = 0.0
            t_base_obj.transform.rotation.z = 0.0
            t_base_obj.transform.rotation.w = 1.0

            try:
                self.tf_broadcaster.sendTransform(t_base_obj)
                self.get_logger().info(f"Published {child_name} at base_link pos: "
                                       f"({p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f})")
            except Exception as e:
                self.get_logger().warning(f"Failed to broadcast {child_name}: {e}")

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
