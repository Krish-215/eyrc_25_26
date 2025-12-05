#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Fruit Detection with Proper TF Transform
Uses same transform approach as ArUco marker detection for accurate positioning.
"""

import rclpy
import sys
import cv2
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, Pose
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_pose
import math

# ============ CONFIGURATION ============
TEAM_ID = 3251  # <<< CHANGE THIS TO YOUR TEAM ID >>>

# Camera intrinsics (matching ArUco reference)
cam_mat = np.array([[931.1829833984375, 0.0, 640.0],
                    [0.0, 931.1829833984375, 360.0],
                    [0.0, 0.0, 1.0]])

# Alternative camera parameters from boilerplate (if needed)
sizeCamX = 1280.0
sizeCamY = 720.0
centerCamX = 642.724365234375
centerCamY = 361.9780578613281
focalX = 915.3003540039062
focalY = 914.0320434570312

# Detection Parameters for BAD FRUITS (greyish-white)
LOWER_GRAY = np.array([0, 0, 50])
UPPER_GRAY = np.array([180, 50, 220])
LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([85, 255, 255])
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 5000
TOP_REGION_HEIGHT_RATIO = 0.30
MIN_GREEN_PIXELS_ABS = 700
MIN_GREEN_RATIO = 0.01

# Depth reading parameters
DEPTH_MEDIAN_KERNEL = 7
MIN_DEPTH = 0.01
MAX_DEPTH = 3.0

# Height normalization
FIX_HEIGHT = True
HEIGHT_MODE = 'average'  # 'fixed', 'average', or 'first'
FIXED_HEIGHT_VALUE = None  # Set specific height or None for auto

# Visualization
SHOW_IMAGE = True
WIN = 'Bad Fruit Detection with TF'

# ======================================


def patch_median_to_meters(depth_img, x, y, kernel_size=7):
    """Extract median depth value from a patch around (x, y)."""
    if depth_img is None:
        return None
    
    h, w = depth_img.shape[:2]
    px = int(np.clip(x, 0, w-1))
    py = int(np.clip(y, 0, h-1))
    
    r = kernel_size // 2
    x0, x1 = max(0, px-r), min(w-1, px+r)
    y0, y1 = max(0, py-r), min(h-1, py+r)
    
    patch = depth_img[y0:y1+1, x0:x1+1]
    if patch.size == 0:
        return None
    
    # Handle different depth image types
    if patch.dtype == np.uint16:
        pf = patch.astype(np.float32)
        pf[pf == 0] = np.nan
        med = float(np.nanmedian(pf)) / 1000.0  # Convert mm to m
    else:
        pf = patch.astype(np.float32)
        pf[pf <= 0.0] = np.nan
        med = float(np.nanmedian(pf))
    
    if math.isnan(med) or med < MIN_DEPTH or med > MAX_DEPTH:
        return None
    
    return med


class FruitTFDetector(Node):
    """ROS2 Node for detecting bad fruits and publishing TF frames."""
    
    def __init__(self):
        super().__init__('fruit_tf_detector')
        
        # CV Bridge
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        
        # Height tracking
        self.height_reference = None
        self.detected_heights = []
        
        # Subscriptions
        self.color_cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self.colorimagecb, 10
        )
        self.depth_cam_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depthimagecb, 10
        )
        
        # Timer for processing
        self.timer = self.create_timer(0.2, self.process_image)
        
        if SHOW_IMAGE:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        
        self.get_logger().info(f"=== Fruit TF Detector Started ===")
        self.get_logger().info(f"Team ID: {TEAM_ID}")
        self.get_logger().info(f"Height fix: {FIX_HEIGHT} (mode: {HEIGHT_MODE})")
        self.get_logger().info(f"Publishing TF frames: {TEAM_ID}_bad_fruit_<id>")
    
    def colorimagecb(self, data):
        """Callback for RGB image."""
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB conversion error: {e}")
    
    def depthimagecb(self, data):
        """Callback for depth image."""
        try:
            # Try to handle different encodings
            if data.encoding == '32FC1':
                self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            elif data.encoding in ('16UC1', 'mono16'):
                depth_raw = self.bridge.imgmsg_to_cv2(data, "16UC1")
                self.depth_image = depth_raw  # Keep as uint16 for patch_median function
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion error: {e}")
    
    def detect_bad_fruits(self, img):
        """
        Detect bad fruits (greyish-white with green tops).
        Returns list of detections with center, contour, and metadata.
        """
        results = []
        if img is None:
            return results
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create masks
        mask_gray = cv2.inRange(hsv, LOWER_GRAY, UPPER_GRAY)
        mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        H, W = mask_gray.shape[:2]
        fid = 1
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
                fid += 1
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Check for green top
            y_top_start = y
            y_top_end = y + int(h * TOP_REGION_HEIGHT_RATIO)
            y_top_end = min(y_top_end, H)
            
            top_region = mask_green[y_top_start:y_top_end, x:x+w]
            green_count = int(np.sum(top_region > 0))
            top_area = max(1, top_region.size)
            min_need = max(MIN_GREEN_PIXELS_ABS, int(MIN_GREEN_RATIO * top_area))
            
            if green_count < min_need:
                fid += 1
                continue
            
            # Calculate green centroid (for depth reading)
            Mg = cv2.moments(top_region)
            if Mg.get('m00', 0) == 0:
                ys, xs = np.where(top_region > 0)
                if len(xs) == 0:
                    fid += 1
                    continue
                gx_local = int(np.mean(xs))
                gy_local = int(np.mean(ys))
            else:
                gx_local = int(Mg['m10'] / Mg['m00'])
                gy_local = int(Mg['m01'] / Mg['m00'])
            
            gx = x + gx_local
            gy = y_top_start + gy_local
            
            # Body centroid (backup)
            Mb = cv2.moments(cnt)
            if Mb.get('m00', 0) == 0:
                body_cx, body_cy = x + w // 2, y + h // 2
            else:
                body_cx = int(Mb['m10'] / Mb['m00'])
                body_cy = int(Mb['m01'] / Mb['m00'])
            
            results.append({
                'id': fid,
                'contour': cnt,
                'bbox': (x, y, w, h),
                'green_px': (gx, gy),
                'body_px': (body_cx, body_cy),
                'green_count': green_count
            })
            fid += 1
        
        return results
    
    def pixel_to_camera_pose(self, pixel_x, pixel_y, depth):
        """
        Convert pixel coordinates and depth to a Pose in camera_link frame.
        Uses proper pinhole camera model (similar to ArUco approach).
        """
        # Use camera intrinsics
        fx = cam_mat[0, 0]  # 931.18
        fy = cam_mat[1, 1]  # 931.18
        cx = cam_mat[0, 2]  # 640.0
        cy = cam_mat[1, 2]  # 360.0
        
        # Standard pinhole projection (inverse)
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        
        X_cam = (float(pixel_x) - cx) * depth / fx
        Y_cam = (float(pixel_y) - cy) * depth / fy
        Z_cam = depth
        
        # Create Pose
        pose = Pose()
        pose.position.x = X_cam
        pose.position.y = Y_cam
        pose.position.z = Z_cam
        
        # Identity orientation (no rotation)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        
        return pose
    
    def process_image(self):
        """Main processing loop - detect fruits and publish TF frames."""
        if self.cv_image is None or self.depth_image is None:
            return
        
        img = self.cv_image.copy()
        detections = self.detect_bad_fruits(img)
        
        if not detections:
            if SHOW_IMAGE:
                cv2.imshow(WIN, img)
                cv2.waitKey(1)
            return
        
        # First pass: collect heights for averaging
        if FIX_HEIGHT and HEIGHT_MODE == 'average':
            temp_heights = []
            for d in detections:
                gx, gy = d['green_px']
                depth = patch_median_to_meters(self.depth_image, gx, gy, DEPTH_MEDIAN_KERNEL)
                if depth is None:
                    continue
                
                # Get pose in camera frame
                pose_cam = self.pixel_to_camera_pose(gx, gy, depth)
                
                try:
                    # Transform to base_link
                    base_to_camera = self.tf_buffer.lookup_transform(
                        'base_link', 'camera_link', rclpy.time.Time()
                    )
                    pose_base = do_transform_pose(pose_cam, base_to_camera)
                    temp_heights.append(pose_base.position.z)
                except TransformException:
                    pass
            
            if temp_heights:
                self.height_reference = np.median(temp_heights)
                self.get_logger().info(
                    f"Height reference: {self.height_reference:.3f}m "
                    f"(from {len(temp_heights)} detections)"
                )
        
        # Second pass: publish TF frames
        published_count = 0
        for d in detections:
            fid = d['id']
            cnt = d['contour']
            x, y, w, h = d['bbox']
            gx, gy = d['green_px']
            
            # Get depth
            depth = patch_median_to_meters(self.depth_image, gx, gy, DEPTH_MEDIAN_KERNEL)
            if depth is None:
                # Try body position as fallback
                bx, by = d['body_px']
                depth = patch_median_to_meters(self.depth_image, bx, by, DEPTH_MEDIAN_KERNEL)
                if depth is None:
                    cv2.drawContours(img, [cnt], -1, (80, 80, 80), 1)
                    cv2.putText(img, f"ID:{fid} NO DEPTH", (x, y-6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    self.get_logger().warning(f"ID:{fid} - NO DEPTH")
                    continue
                gx, gy = bx, by  # Use body position
            
            # Convert to camera frame pose
            pose_cam = self.pixel_to_camera_pose(gx, gy, depth)
            
            try:
                # Transform from camera_link to base_link
                # This is the same approach as ArUco detection
                base_to_camera = self.tf_buffer.lookup_transform(
                    'base_link', 'camera_link', rclpy.time.Time()
                )
                
                # Transform pose
                pose_base = do_transform_pose(pose_cam, base_to_camera)
                
                # Apply height normalization if enabled
                original_z = pose_base.position.z
                if FIX_HEIGHT:
                    if HEIGHT_MODE == 'fixed' and FIXED_HEIGHT_VALUE is not None:
                        pose_base.position.z = FIXED_HEIGHT_VALUE
                    elif HEIGHT_MODE == 'first':
                        if self.height_reference is None:
                            self.height_reference = pose_base.position.z
                        pose_base.position.z = self.height_reference
                    elif HEIGHT_MODE == 'average' and self.height_reference is not None:
                        pose_base.position.z = self.height_reference
                
                # Create and publish transform
                transform = TransformStamped()
                transform.header.stamp = self.get_clock().now().to_msg()
                transform.header.frame_id = 'base_link'
                transform.child_frame_id = f'{TEAM_ID}_bad_fruit_{fid}'
                
                # Set translation
                transform.transform.translation.x = pose_base.position.x
                transform.transform.translation.y = pose_base.position.y
                transform.transform.translation.z = pose_base.position.z
                
                # Set rotation (identity)
                transform.transform.rotation = pose_base.orientation
                
                # Publish transform
                self.br.sendTransform(transform)
                published_count += 1
                
                # Log
                height_info = ""
                if FIX_HEIGHT and abs(original_z - pose_base.position.z) > 0.001:
                    height_info = f" [Z: {original_z:.3f}→{pose_base.position.z:.3f}]"
                
                self.get_logger().info(
                    f'Published {TEAM_ID}_bad_fruit_{fid}{height_info}: '
                    f'base({pose_base.position.x:.3f}, {pose_base.position.y:.3f}, '
                    f'{pose_base.position.z:.3f}) depth={depth:.3f}m'
                )
                
                # Visualize
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(img, (gx, gy), 6, (255, 0, 0), -1)
                
                label = f"ID:{fid} {depth:.2f}m"
                cv2.putText(img, label, (gx+8, gy-6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                pos_text = f"({pose_base.position.x:.2f},{pose_base.position.y:.2f},{pose_base.position.z:.2f})"
                cv2.putText(img, pos_text, (x, y+h+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
            except TransformException as ex:
                self.get_logger().warning(f"Transform failed for ID {fid}: {ex}")
            except Exception as e:
                self.get_logger().error(f"Error publishing TF for ID {fid}: {e}")
        
        # Add info overlay
        info_text = [
            f"Team ID: {TEAM_ID}",
            f"Detected: {len(detections)} bad fruits",
            f"Published: {published_count} TFs",
        ]
        
        if FIX_HEIGHT:
            h_mode = HEIGHT_MODE
            if self.height_reference is not None:
                info_text.append(f"Height: {self.height_reference:.3f}m ({h_mode})")
        
        y_pos = 25
        for text in info_text:
            cv2.putText(img, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 25
        
        # Display
        if SHOW_IMAGE:
            cv2.imshow(WIN, img)
            cv2.waitKey(1)


def main():
    rclpy.init(args=sys.argv)
    node = FruitTFDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Fruit TF Detector")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()