#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fruits_tf_flip.py

Detect bad fruits (grey body + green top), place TF at green-top centroid in base_link,
and optionally flip/reflect/offset final TF positions before publishing.

Usage:
 - Edit TEAM_ID and flipping parameters below as needed.
 - Run this as a ROS2 node in your workspace.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from geometry_msgs.msg import TransformStamped, PointStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_point

# ==================== USER CONFIG ====================
TEAM_ID = 3251                # change to your team ID (int)
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False

# If you know the camera optical frame name, set it. Otherwise leave None to auto-detect.
CAMERA_FRAME = None

# Depth / detection params
DEPTH_MEDIAN_KERNEL = 5
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 3000
MAX_DEPTH = 2.0
MIN_DEPTH = 0.01

# Green-top detection
TOP_REGION_HEIGHT_RATIO = 0.30
TOP_REGION_EXTEND_UP = 0
MIN_GREEN_PIXELS_ABS = 800
MIN_GREEN_RATIO = 0.01

# Camera intrinsics (from boilerplate) - verify if needed
sizeCamX = 1280
sizeCamY = 720
centerCamX = 642.724365234375  # cx
centerCamY = 361.9780578613281  # cy
focalX = 915.3003540039062     # fx
focalY = 914.0320434570312     # fy

# ---------- FLIPPING / REFLECTION / OFFSET CONFIG ----------
# Sign flip per-axis: negate coordinate if True
FLIP_SIGN = {'x': True, 'y': False, 'z': False}

# Reflect across plane axis = value. Set axis to None to disable for that axis.
# Example: REFLECT_PLANE = {'x': 0.5, 'y': None, 'z': None}
REFLECT_PLANE = {'x': 0.1, 'y': 0.0, 'z': 1.3}

# Extra translation to apply after flip/reflect (in base_link frame), meters
EXTRA_TRANSLATION = {'x': -0.7, 'y': 0.4, 'z': 0.0}
# ===========================================================


class FruitsTFFlip(Node):
    def __init__(self):
        super().__init__('fruits_tf_flip_node')

        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions (robust to several topic names)
        self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depthimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        # TF2
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # candidate camera frames to auto-detect
        self.camera_candidates = [
            CAMERA_FRAME,
            'camera_color_optical_frame',
            'camera_color_frame',
            'camera_depth_optical_frame',
            'camera_depth_frame',
            'camera_optical_frame',
            'camera_link',
            'camera'
        ]
        seen = set()
        self.camera_candidates = [c for c in self.camera_candidates if c and not (c in seen or seen.add(c))]
        self.chosen_camera_frame = None

        # Timer
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info(f"FruitsTFFlip node started (Team ID: {TEAM_ID})")

    # ---------- Callbacks ----------
    def depthimagecb(self, data: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warning(f"Depth conversion failed: {e}")
            self.depth_image = None

    def colorimagecb(self, data: Image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except Exception as e:
            try:
                rgb = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
                self.cv_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as e2:
                self.get_logger().warning(f"Color conversion failed: {e} | fallback: {e2}")
                self.cv_image = None

    # ---------- Detection ----------
    def bad_fruit_detection(self, rgb_image):
        bad_fruits = []
        if rgb_image is None:
            return bad_fruits

        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        # grey mask (low saturation)
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 220])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        # green mask
        lower_green = np.array([35, 60, 60])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # morphological
        kernel = np.ones((3, 3), np.uint8)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours_gray, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fruit_id = 1
        img_h, img_w = mask_gray.shape[:2]

        for contour in contours_gray:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            # top region
            y_top_start = max(y - TOP_REGION_EXTEND_UP, 0)
            y_top_end = y + int(TOP_REGION_HEIGHT_RATIO * h)
            x_left = x
            x_right = x + w
            y_top_end = min(y_top_end, img_h)
            x_right = min(x_right, img_w)
            if y_top_end <= y_top_start or x_right <= x_left:
                continue

            top_region_green = (mask_green[y_top_start:y_top_end, x_left:x_right] > 0).astype(np.uint8)
            green_pixel_count = int(np.sum(top_region_green))

            bbox_top_area = max(1, (x_right - x_left) * (y_top_end - y_top_start))
            min_green_needed = max(MIN_GREEN_PIXELS_ABS, int(MIN_GREEN_RATIO * bbox_top_area))
            if green_pixel_count < min_green_needed:
                continue

            # green centroid in top region
            Mg = cv2.moments(top_region_green)
            if Mg.get('m00', 0) == 0:
                ys, xs = np.where(top_region_green > 0)
                if len(xs) == 0:
                    continue
                gx_local = int(np.mean(xs))
                gy_local = int(np.mean(ys))
            else:
                gx_local = int(Mg['m10'] / Mg['m00'])
                gy_local = int(Mg['m01'] / Mg['m00'])

            green_cx = x_left + gx_local
            green_cy = y_top_start + gy_local

            # body centroid for visualization
            M = cv2.moments(contour)
            if M.get('m00', 0) == 0:
                body_cx, body_cy = (x + w // 2, y + h // 2)
            else:
                body_cx = int(M['m10'] / M['m00'])
                body_cy = int(M['m01'] / M['m00'])

            # optional ellipse angle
            angle = 0.0
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    angle = float(ellipse[2])
                except Exception:
                    angle = 0.0

            fruit_info = {
                'id': fruit_id,
                'contour': contour,
                'bbox': (x, y, w, h),
                'body_center': (body_cx, body_cy),
                'green_center_px': (green_cx, green_cy),
                'green_pixels': green_pixel_count,
                'angle': angle
            }

            bad_fruits.append(fruit_info)
            fruit_id += 1

        return bad_fruits

    # ---------- Depth reading ----------
    def get_depth_at_point(self, x, y):
        if self.depth_image is None:
            return None
        try:
            h, w = self.depth_image.shape[:2]
            px = int(np.clip(x, 0, w - 1))
            py = int(np.clip(y, 0, h - 1))
            k = max(1, DEPTH_MEDIAN_KERNEL // 2)
            x0 = max(0, px - k)
            x1 = min(w - 1, px + k)
            y0 = max(0, py - k)
            y1 = min(h - 1, py + k)
            patch = self.depth_image[y0:y1 + 1, x0:x1 + 1]
            if patch.size == 0:
                return None
            if patch.dtype == np.uint16:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0] = np.nan
                depth_m = float(np.nanmedian(patch_f)) / 1000.0
            elif patch.dtype in [np.float32, np.float64]:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f <= 0.0] = np.nan
                depth_m = float(np.nanmedian(patch_f))
            else:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0] = np.nan
                depth_m = float(np.nanmedian(patch_f))
            if math.isnan(depth_m) or depth_m < MIN_DEPTH or depth_m > MAX_DEPTH:
                return None
            return depth_m
        except Exception as e:
            self.get_logger().warning(f"Depth read error at ({x},{y}): {e}")
            return None

    # ---------- camera frame pick ----------
    def pick_camera_frame(self):
        if self.chosen_camera_frame:
            return self.chosen_camera_frame
        for cand in self.camera_candidates:
            try:
                self.tf_buffer.lookup_transform('base_link', cand, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.2))
                self.chosen_camera_frame = cand
                self.get_logger().info(f"Selected camera frame: '{cand}'")
                return cand
            except Exception:
                continue
        self.get_logger().warning("No camera frame auto-detected. Set CAMERA_FRAME to correct frame name.")
        return None

    # ---------- apply flip/reflect/offset ----------
    def apply_flips_and_reflects(self, px, py, pz):
        # sign flips
        if FLIP_SIGN.get('x', False):
            px = -px
        if FLIP_SIGN.get('y', False):
            py = -py
        if FLIP_SIGN.get('z', False):
            pz = -pz

        # plane reflections: x' = 2*c - x
        if REFLECT_PLANE.get('x') is not None:
            c = float(REFLECT_PLANE['x'])
            px = 2.0 * c - px
        if REFLECT_PLANE.get('y') is not None:
            c = float(REFLECT_PLANE['y'])
            py = 2.0 * c - py
        if REFLECT_PLANE.get('z') is not None:
            c = float(REFLECT_PLANE['z'])
            pz = 2.0 * c - pz

        # extra translation
        px += float(EXTRA_TRANSLATION.get('x', 0.0))
        py += float(EXTRA_TRANSLATION.get('y', 0.0))
        pz += float(EXTRA_TRANSLATION.get('z', 0.0))

        return px, py, pz

    # ---------- Main processing ----------
    def process_image(self):
        if self.cv_image is None:
            return

        display_img = self.cv_image.copy()
        bad_fruits = self.bad_fruit_detection(self.cv_image)
        cam_frame = self.pick_camera_frame()
        if cam_frame is None:
            # visualize but don't publish
            for f in bad_fruits:
                x, y, w, h = f['bbox']
                cxg, cyg = f['green_center_px']
                cv2.drawContours(display_img, [f['contour']], -1, (0, 165, 255), 2)
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.circle(display_img, (cxg, cyg), 4, (255, 0, 0), -1)
                cv2.putText(display_img, f"ID:{f['id']} (no cam frame)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
            if SHOW_IMAGE:
                cv2.imshow('fruits_tf_view', display_img)
                cv2.waitKey(1)
            return

        for f in bad_fruits:
            fid = f['id']
            contour = f['contour']
            x, y, w, h = f['bbox']
            gx, gy = f['green_center_px']
            green_px = f['green_pixels']

            depth_m = self.get_depth_at_point(gx, gy)
            if depth_m is None:
                cv2.drawContours(display_img, [contour], -1, (0, 165, 255), 2)
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.circle(display_img, (gx, gy), 4, (255, 0, 0), -1)
                cv2.putText(display_img, f"ID:{fid} no depth", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
                continue

            # correct projection
            X_cam = (float(gx) - float(centerCamX)) / float(focalX) * depth_m
            Y_cam = (float(gy) - float(centerCamY)) / float(focalY) * depth_m
            Z_cam = depth_m

            # visualize
            cv2.drawContours(display_img, [contour], -1, (0, 0, 255), 2)
            y_top_start = max(y - TOP_REGION_EXTEND_UP, 0)
            y_top_end = y + int(TOP_REGION_HEIGHT_RATIO * h)
            x_left = x
            x_right = min(x + w, display_img.shape[1])
            y_top_end = min(y_top_end, display_img.shape[0])
            cv2.rectangle(display_img, (x_left, y_top_start), (x_right, y_top_end), (255, 255, 0), 1)
            cv2.circle(display_img, (gx, gy), 6, (255, 0, 0), -1)  # blue centroid
            bx, by = f['body_center']
            cv2.circle(display_img, (bx, by), 3, (255, 255, 255), -1)
            cv2.putText(display_img, f"ID:{fid} D:{depth_m:.2f} Gpx:{green_px}", (gx - 40, gy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

            # stamp point in camera frame
            pt_cam = PointStamped()
            pt_cam.header.stamp = self.get_clock().now().to_msg()
            pt_cam.header.frame_id = cam_frame
            pt_cam.point.x = float(X_cam)
            pt_cam.point.y = float(Y_cam)
            pt_cam.point.z = float(Z_cam)

            # lookup transform and use do_transform_point
            try:
                transform_cam_to_base = self.tf_buffer.lookup_transform('base_link', cam_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.5))
                pt_base = do_transform_point(pt_cam, transform_cam_to_base)

                px = float(pt_base.point.x)
                py = float(pt_base.point.y)
                pz = float(pt_base.point.z)

                # apply flips/reflections/offset in base_link frame
                px, py, pz = self.apply_flips_and_reflects(px, py, pz)

                # publish final TF
                child_name = f"{TEAM_ID}_bad_fruit_{fid}"
                t_base = TransformStamped()
                t_base.header.stamp = self.get_clock().now().to_msg()
                t_base.header.frame_id = 'base_link'
                t_base.child_frame_id = child_name
                t_base.transform.translation.x = px
                t_base.transform.translation.y = py
                t_base.transform.translation.z = pz
                # identity rotation
                t_base.transform.rotation.x = 0.0
                t_base.transform.rotation.y = 0.0
                t_base.transform.rotation.z = 0.0
                t_base.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t_base)

                self.get_logger().info(f"Published {child_name} at base ({px:.3f}, {py:.3f}, {pz:.3f})")

            except Exception as e:
                self.get_logger().warning(f"Failed to transform or publish for ID {fid}: {e}")

        if SHOW_IMAGE:
            try:
                cv2.imshow('fruits_tf_view', display_img)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warning(f"OpenCV display error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTFFlip()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTFFlip node")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
