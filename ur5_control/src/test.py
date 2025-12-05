#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fruits detection and TF publisher for Task 1B (single-file implementation).
- Subscribes to color and depth images.
- Detects BAD (greyish-white) fruits (ignores good reddish-pink).
- Publishes transforms named: <TEAM_ID>_bad_fruit_<id>
- Visualizes contours and centers in OpenCV window.
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

# --------- USER CONFIG ---------
TEAM_ID = 5  # <-- CHANGE THIS to your team id before submission (mandatory naming)
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False
DEPTH_MEDIAN_KERNEL = 5     # use kxk median around centroid to reduce noise
MIN_CONTOUR_AREA = 600      # tune as necessary for filtering small blobs
# Camera intrinsics (as provided in boilerplate)
sizeCamX = 1280
sizeCamY = 720
centerCamX = 642.724365234375
centerCamY = 361.9780578613281
focalX = 915.3003540039062
focalY = 914.0320434570312
# -------------------------------


class FruitsTF(Node):
    def __init__(self):
        super().__init__('fruits_tf_node')

        # cv bridge
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.depth_encoding = None

        # callback group
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions: include multiple topic names to be robust to different setups
        self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depthimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        # TF broadcaster & buffer/listener (for computing transforms w.r.t base_link)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Processing timer (5 Hz)
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info("FruitsTF node started.")

    # ---------------- Callbacks ----------------
    def depthimagecb(self, data: Image):
        """Convert incoming depth Image to a numpy array and keep encoding."""
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            self.depth_image = depth_cv
            self.depth_encoding = data.encoding if hasattr(data, 'encoding') else None
        except Exception as e:
            self.get_logger().warning(f"Depth conversion failed: {e}")
            self.depth_image = None
            self.depth_encoding = None

    def colorimagecb(self, data: Image):
        """Convert incoming color Image to BGR numpy image."""
        try:
            # Convert to BGR (OpenCV default)
            color_cv = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            # If camera orientation differs, flip/rotate here (uncomment if needed)
            # color_cv = cv2.rotate(color_cv, cv2.ROTATE_180)
            self.cv_image = color_cv
        except Exception as e:
            self.get_logger().warning(f"Color conversion failed: {e}")
            self.cv_image = None

    # ---------------- Detection ----------------
    def bad_fruit_detection(self, rgb_image):
        """
        Detect greyish-white (bad) fruits. returns list of dicts:
         {'center': (cX,cY), 'distance': dist_m, 'angle': deg, 'width': w_px, 'id': id}
        """
        bad_fruits = []

        if rgb_image is None:
            return bad_fruits

        # Convert to HSV for color segmentation
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Grey/white: low saturation, mid-high value.
        # These thresholds are conservative; tune if needed.
        lower_bad = np.array([0, 0, 100])     # H ignored (0), S low, V > 100
        upper_bad = np.array([179, 70, 255])  # S up to 70 to catch greyish-white

        mask_bad = cv2.inRange(hsv, lower_bad, upper_bad)

        # Remove red-ish (good fruits) using an inverse-red mask to avoid misclassifying pinks
        # Red/pink HSV ranges (two ranges for red wrap-around)
        lower_red1 = np.array([0, 60, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 60, 50])
        upper_red2 = np.array([179, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # Subtract red areas from bad mask
        mask_bad = cv2.bitwise_and(mask_bad, cv2.bitwise_not(mask_red))

        # Morphological cleanups
        kernel = np.ones((5, 5), np.uint8)
        mask_bad = cv2.morphologyEx(mask_bad, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_bad = cv2.morphologyEx(mask_bad, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Contours
        contours, _ = cv2.findContours(mask_bad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # filter by area and bounding box size
        contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

        # assign sequential ids starting at 1 each frame (allowed per problem statement)
        fruit_id = 1
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            # Estimate distance using depth image (will be read in process_image)
            # Here we only return center & width; depth will be added by process_image
            fruit_info = {
                'center': (cX, cY),
                'distance': None,  # filled later
                'angle': None,     # optional, filled later if needed
                'width': w,
                'id': fruit_id,
                'contour': cnt  # include contour for drawing
            }
            bad_fruits.append(fruit_info)
            fruit_id += 1

        return bad_fruits

    # ---------------- Process & TF publishing ----------------
    def process_image(self):
        """Timer-driven processing: detect bad fruits, get depth, compute pose, and publish TFs."""
        if self.cv_image is None:
            return

        rgb = self.cv_image.copy()
        bads = self.bad_fruit_detection(rgb)

        # For each detection: read depth, compute x,y,z in camera frame, publish camera->cam_<id> and base_link->TEAM_bad_fruit_id
        for det in bads:
            cX, cY = det['center']
            w_px = det['width']
            fid = det['id']

            # draw contour and center on visualization image
            cnt = det.get('contour', None)
            if cnt is not None:
                cv2.drawContours(rgb, [cnt], -1, (0, 0, 255), 2)
            cv2.circle(rgb, (cX, cY), 4, (0, 255, 0), -1)
            cv2.putText(rgb, f"id:{fid}", (cX + 6, cY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # get depth (median in small patch)
            z_m = None
            if self.depth_image is not None:
                try:
                    h, w = self.depth_image.shape[:2]
                    # clamp coordinates
                    px = np.clip(cX, 0, w - 1)
                    py = np.clip(cY, 0, h - 1)
                    k = DEPTH_MEDIAN_KERNEL // 2
                    x0, x1 = max(0, px - k), min(w - 1, px + k)
                    y0, y1 = max(0, py - k), min(h - 1, py + k)
                    patch = self.depth_image[y0:y1 + 1, x0:x1 + 1]

                    if patch.size == 0:
                        z_m = None
                    else:
                        # convert to meters depending on dtype
                        if patch.dtype == np.uint16:
                            # typically depth in uint16 is in millimeters
                            patch = patch.astype(np.float32)
                            patch[patch == 0] = np.nan
                            z_vals = patch / 1000.0
                            z_m = float(np.nanmedian(z_vals))
                        elif patch.dtype == np.float32 or patch.dtype == np.float64:
                            patch = patch.astype(np.float32)
                            patch[patch == 0.0] = np.nan
                            z_vals = patch
                            z_m = float(np.nanmedian(z_vals))
                        else:
                            # fallback: try to convert
                            patchf = patch.astype(np.float32)
                            patchf[patchf == 0] = np.nan
                            z_m = float(np.nanmedian(patchf))
                except Exception as e:
                    self.get_logger().warning(f"Depth read error at ({cX},{cY}): {e}")
                    z_m = None

            # If we couldn't read depth, skip publishing 3D TF (but still show detection)
            if z_m is None or math.isnan(z_m) or z_m <= 0.001:
                det['distance'] = None
                continue

            det['distance'] = z_m

            # compute camera frame coordinates using provided formula:
            # x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
            # y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
            # z = distance_from_rgb
            x_cam = z_m * (sizeCamX - cX - centerCamX) / focalX
            y_cam = z_m * (sizeCamY - cY - centerCamY) / focalY
            z_cam = z_m

            # Publish camera frame -> cam_<fid> transform
            cam_child = f"cam_{fid}"
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = 'camera_link'  # as per instructions
            t_cam.child_frame_id = cam_child
            t_cam.transform.translation.x = x_cam
            t_cam.transform.translation.y = y_cam
            t_cam.transform.translation.z = z_cam
            # object's top orientation: leave identity (no rotation)
            q_identity = quaternion_from_euler(0, 0, 0)
            t_cam.transform.rotation.x = q_identity[0]
            t_cam.transform.rotation.y = q_identity[1]
            t_cam.transform.rotation.z = q_identity[2]
            t_cam.transform.rotation.w = q_identity[3]

            # broadcast cam->child
            try:
                self.tf_broadcaster.sendTransform(t_cam)
            except Exception as e:
                self.get_logger().warning(f"Failed to broadcast camera->cam_{fid}: {e}")

            # Now lookup transform base_link <- camera_link to compute position in base_link
            try:
                # Lookup transform from base_link to camera_link (so we can transform camera-frame point to base frame)
                # target_frame: 'base_link', source_frame: 'camera_link'
                trans = self.tf_buffer.lookup_transform(
                    target_frame='base_link',
                    source_frame='camera_link',
                    time=rclpy.time.Time(),  # latest available
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                # trans.transform.rotation is quaternion of camera frame in base frame
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
                ], dtype=np.float64)

                # Build 4x4 matrix for base <- camera
                T_base_cam = quaternion_matrix(q_base_cam)
                T_base_cam[0:3, 3] = t_base_cam

                # camera point homogeneous
                p_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
                p_base_h = T_base_cam.dot(p_cam)
                p_base = p_base_h[:3]

                # quaternion for base->child is q_base_cam * q_cam_child (q_cam_child is identity)
                q_base_child = q_base_cam  # identity multiplication

                # Publish base_link -> <TEAM>_bad_fruit_<fid>
                child_name = f"{TEAM_ID}_bad_fruit_{fid}"
                t_base_obj = TransformStamped()
                t_base_obj.header.stamp = self.get_clock().now().to_msg()
                t_base_obj.header.frame_id = 'base_link'
                t_base_obj.child_frame_id = child_name
                t_base_obj.transform.translation.x = float(p_base[0])
                t_base_obj.transform.translation.y = float(p_base[1])
                t_base_obj.transform.translation.z = float(p_base[2])
                t_base_obj.transform.rotation.x = q_base_child[0]
                t_base_obj.transform.rotation.y = q_base_child[1]
                t_base_obj.transform.rotation.z = q_base_child[2]
                t_base_obj.transform.rotation.w = q_base_child[3]

                self.tf_broadcaster.sendTransform(t_base_obj)

            except Exception as e:
                # Could not lookup transform or compute base pose
                self.get_logger().warning(f"Could not compute base-frame transform for fruit {fid}: {e}")
                # continue (we already published camera->cam_<fid> which may be used later)

        # Display window
        if SHOW_IMAGE:
            try:
                cv2.imshow('fruits_tf_view', rgb)
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
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

