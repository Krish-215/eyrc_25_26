#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRECTED VERSION: Fixed projection formula
The original boilerplate formula had inverted signs. This version corrects them.

Original (WRONG):
    x = distance * (sizeCamX - cX - centerCamX) / focalX
    y = distance * (sizeCamY - cY - centerCamY) / focalY

Corrected (STANDARD):
    x = distance * (cX - centerCamX) / focalX
    y = distance * (cY - centerCamY) / focalY
    z = distance
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, PointStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import cv2
import numpy as np
import math

# ---------------- User config ----------------
TEAM_ID = 3251            # <-- set your team ID
SHOW_IMAGE = True

RGB_TOPIC = '/camera/image_raw'
DEPTH_TOPIC = '/camera/depth/image_raw'
CAMERA_INFO_TOPICS = ['/camera/camera_info', '/camera/depth/camera_info']

# detection params
LOWER_GRAY = np.array([0, 0, 50])
UPPER_GRAY = np.array([180, 50, 220])
LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([85, 255, 255])
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 5000
TOP_REGION_HEIGHT_RATIO = 0.30
MIN_GREEN_PIXELS_ABS = 700
MIN_GREEN_RATIO = 0.01

# depth reading params
DEPTH_MEDIAN_KERNEL = 7
DEPTH_MEDIAN_KERNEL_LARGE = 11
VERTICAL_OFFSETS = [0, 3, 6, 9]
MAX_NEAREST_SEARCH_RADIUS = 40
MIN_DEPTH = 0.01
MAX_DEPTH = 3.0

# frames
OPTICAL_FRAME = 'camera_optical_frame'
CAMERA_LINK_FRAME = 'camera_link'
BASE_FRAME = 'base_link'

# Optional corrections (try these if still misaligned)
FLIP_X = True
FLIP_Y = False
FLIP_Z = False
EXTRA_TX = 0.0
EXTRA_TY = 0.0
EXTRA_TZ = 0.0

# Use corrected formula instead of boilerplate?
USE_CORRECTED_FORMULA = True  # Set to False to use original boilerplate formula

# ROTATION FIX for coordinate frame alignment
# If TFs appear rotated, enable this to apply 90° clockwise rotation
APPLY_ROTATION_FIX = True  # Rotates coordinates 90° clockwise around Z-axis

# HEIGHT NORMALIZATION - Fix all fruits at same height
FIX_HEIGHT = True          # Enable to normalize all Z coordinates
FIXED_HEIGHT_VALUE = None  # Set to specific height (e.g., 0.15), or None to use average/first detection
HEIGHT_MODE = 'average'    # 'fixed': use FIXED_HEIGHT_VALUE, 'average': average of all detections, 'first': use first detection's height

WIN = 'Fruits TF: CORRECTED Projection + Rotation Fix'

# Camera intrinsics from boilerplate
sizeCamX = 1280.0
sizeCamY = 720.0
centerCamX = 642.724365234375
centerCamY = 361.9780578613281
focalX = 915.3003540039062
focalY = 914.0320434570312

# ---------------- helpers ----------------
def patch_median_to_meters(depth_img, x, y, k):
    if depth_img is None:
        return None
    h, w = depth_img.shape[:2]
    px = int(np.clip(x, 0, w-1)); py = int(np.clip(y, 0, h-1))
    r = k // 2
    x0, x1 = max(0, px-r), min(w-1, px+r)
    y0, y1 = max(0, py-r), min(h-1, py+r)
    patch = depth_img[y0:y1+1, x0:x1+1]
    if patch.size == 0:
        return None
    if patch.dtype == np.uint16:
        pf = patch.astype(np.float32); pf[pf == 0] = np.nan
        med = float(np.nanmedian(pf)) / 1000.0
    else:
        pf = patch.astype(np.float32); pf[pf <= 0.0] = np.nan
        med = float(np.nanmedian(pf))
    if math.isnan(med) or med < MIN_DEPTH or med > MAX_DEPTH:
        return None
    return med

def find_nearest_valid(depth_img, x, y, max_r=MAX_NEAREST_SEARCH_RADIUS):
    if depth_img is None:
        return (None, None, None)
    h, w = depth_img.shape[:2]
    cx = int(np.clip(x, 0, w-1)); cy = int(np.clip(y, 0, h-1))
    for r in range(1, max_r+1):
        x0, x1 = max(0, cx-r), min(w-1, cx+r)
        y0, y1 = max(0, cy-r), min(h-1, cy+r)
        coords = []
        for xi in range(x0, x1+1):
            coords.append((xi, y0)); coords.append((xi, y1))
        for yi in range(y0+1, y1):
            coords.append((x0, yi)); coords.append((x1, yi))
        for (px, py) in coords:
            try:
                val = depth_img[py, px]
            except Exception:
                continue
            if depth_img.dtype == np.uint16:
                if int(val) == 0:
                    continue
                return (float(int(val))/1000.0, px, py)
            else:
                fv = float(val)
                if fv <= 0.0 or math.isnan(fv):
                    continue
                return (fv, px, py)
    return (None, None, None)

# ---------------- Node ----------------
class FruitsTFCorrected(Node):
    def __init__(self):
        super().__init__('fruits_tf_corrected')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        self.fx = None; self.fy = None; self.cx = None; self.cy = None
        self.camera_info_received = False

        # Height normalization tracking
        self.height_reference = None  # Store reference height for normalization
        self.detected_heights = []    # Store all detected heights for averaging

        # TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # subscribe
        self.create_subscription(Image, RGB_TOPIC, self.rgb_cb, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, 10)
        for t in CAMERA_INFO_TOPICS:
            try:
                self.create_subscription(CameraInfo, t, self.camera_info_cb, 10)
            except Exception:
                pass

        if SHOW_IMAGE:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        self.create_timer(0.12, self.tick)
        
        formula_type = "CORRECTED (standard)" if USE_CORRECTED_FORMULA else "ORIGINAL (boilerplate)"
        rotation_status = "WITH 90° CW rotation" if APPLY_ROTATION_FIX else "NO rotation"
        height_status = f"HEIGHT FIX: {HEIGHT_MODE}" if FIX_HEIGHT else "NO height fix"
        self.get_logger().info(f"Node started with {formula_type} projection formula {rotation_status}")
        self.get_logger().info(f"Height normalization: {height_status}")
        self.get_logger().info("Publishing: camera_link->cam_<id> and base_link->TEAM_bad_fruit_<id>")

    def rgb_cb(self, msg: Image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f"RGB conversion failed: {e}")
            self.cv_image = None

    def depth_cb(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warning(f"Depth conversion failed: {e}")
            self.depth_image = None

    def camera_info_cb(self, msg):
        try:
            K = msg.k
            self.fx = float(K[0]); self.fy = float(K[4]); self.cx = float(K[2]); self.cy = float(K[5])
            self.camera_info_received = True
            self.get_logger().info(f"CameraInfo: fx={self.fx:.3f} fy={self.fy:.3f} cx={self.cx:.3f} cy={self.cy:.3f}")
        except Exception as e:
            self.get_logger().warning(f"camera_info_cb error: {e}")

    def detect_bad_fruits(self, img):
        res = []
        if img is None:
            return res
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_gray = cv2.inRange(hsv, LOWER_GRAY, UPPER_GRAY)
        mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        kernel = np.ones((3,3), np.uint8)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = mask_gray.shape[:2]
        fid = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
                fid += 1
                continue
            x,y,w,h = cv2.boundingRect(cnt)
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
            Mg = cv2.moments(top_region)
            if Mg.get('m00', 0) == 0:
                ys, xs = np.where(top_region > 0)
                if len(xs) == 0:
                    fid += 1
                    continue
                gx_local = int(np.mean(xs)); gy_local = int(np.mean(ys))
            else:
                gx_local = int(Mg['m10']/Mg['m00']); gy_local = int(Mg['m01']/Mg['m00'])
            gx = x + gx_local; gy = y_top_start + gy_local
            Mb = cv2.moments(cnt)
            if Mb.get('m00', 0) == 0:
                body_cx, body_cy = x + w//2, y + h//2
            else:
                body_cx = int(Mb['m10']/Mb['m00']); body_cy = int(Mb['m01']/Mb['m00'])
            res.append({'id': fid, 'contour': cnt, 'bbox': (x,y,w,h),
                        'green_px': (gx, gy), 'body_px': (body_cx, body_cy), 'green_count': green_count})
            fid += 1
        return res

    def get_depth_for_point(self, gx, gy, body_px):
        # 1) patch median
        d = patch_median_to_meters(self.depth_image, gx, gy, DEPTH_MEDIAN_KERNEL)
        if d is not None:
            return d, gx, gy, 'patch'
        # 2) vertical offsets
        for off in VERTICAL_OFFSETS:
            if off == 0:
                continue
            y_try = gy + off
            d = patch_median_to_meters(self.depth_image, gx, y_try, DEPTH_MEDIAN_KERNEL)
            if d is not None:
                return d, gx, y_try, f'offset_{off}'
        # 3) body patch
        bx, by = body_px
        d = patch_median_to_meters(self.depth_image, bx, by, DEPTH_MEDIAN_KERNEL)
        if d is not None:
            return d, bx, by, 'body'
        # 4) nearest valid
        val, px, py = find_nearest_valid(self.depth_image, gx, gy, max_r=MAX_NEAREST_SEARCH_RADIUS)
        if val is not None:
            dref = patch_median_to_meters(self.depth_image, px, py, DEPTH_MEDIAN_KERNEL)
            if dref is not None:
                return dref, px, py, 'nearest_patch'
            return val, px, py, 'nearest_raw'
        # 5) larger patch
        d = patch_median_to_meters(self.depth_image, gx, gy, DEPTH_MEDIAN_KERNEL_LARGE)
        if d is not None:
            return d, gx, gy, 'large_patch'
        return None, None, None, 'no_depth'

    def pixel_to_camera_optical(self, u, v, distance):
        """
        Two projection formulas available:
        
        CORRECTED (STANDARD - RECOMMENDED):
            x = distance * (cX - centerCamX) / focalX
            y = distance * (cY - centerCamY) / focalY
            z = distance
        
        ORIGINAL BOILERPLATE (NON-STANDARD):
            x = distance * (sizeCamX - cX - centerCamX) / focalX
            y = distance * (sizeCamY - cY - centerCamY) / focalY
            z = distance
        
        The boilerplate formula appears to have inverted X and Y coordinates.
        """
        if USE_CORRECTED_FORMULA:
            # CORRECTED: Standard pinhole projection
            x = distance * (float(u) - centerCamX) / focalX
            y = distance * (float(v) - centerCamY) / focalY
            z = float(distance)
        else:
            # ORIGINAL: Boilerplate formula (likely incorrect)
            x = distance * (sizeCamX - float(u) - centerCamX) / focalX
            y = distance * (sizeCamY - float(v) - centerCamY) / focalY
            z = float(distance)
        
        return x, y, z

    def tick(self):
        if self.cv_image is None:
            return

        img = self.cv_image.copy()
        detections = self.detect_bad_fruits(img)

        # First pass: collect all heights if using average mode
        if FIX_HEIGHT and HEIGHT_MODE == 'average':
            temp_heights = []
            for d in detections:
                fid = d['id']
                gx, gy = d['green_px']
                bx, by = d['body_px']
                
                depth_m, used_x, used_y, method = self.get_depth_for_point(gx, gy, (bx, by))
                if depth_m is None:
                    continue
                
                # Quick compute just to get height
                Xc_opt, Yc_opt, Zc_opt = self.pixel_to_camera_optical(gx, gy, depth_m)
                pt_opt = PointStamped()
                pt_opt.header.stamp = self.get_clock().now().to_msg()
                pt_opt.header.frame_id = OPTICAL_FRAME
                pt_opt.point.x = float(Xc_opt)
                pt_opt.point.y = float(Yc_opt)
                pt_opt.point.z = float(Zc_opt)
                
                try:
                    trans_camlink = self.tf_buffer.lookup_transform(
                        CAMERA_LINK_FRAME, OPTICAL_FRAME,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.5)
                    )
                    pt_camlink = do_transform_point(pt_opt, trans_camlink)
                    
                    trans_base_camlink = self.tf_buffer.lookup_transform(
                        BASE_FRAME, CAMERA_LINK_FRAME,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.5)
                    )
                    pt_cl = PointStamped()
                    pt_cl.header.stamp = self.get_clock().now().to_msg()
                    pt_cl.header.frame_id = CAMERA_LINK_FRAME
                    pt_cl.point.x = float(pt_camlink.point.x)
                    pt_cl.point.y = float(pt_camlink.point.y)
                    pt_cl.point.z = float(pt_camlink.point.z)
                    pt_base = do_transform_point(pt_cl, trans_base_camlink)
                    
                    base_z = float(pt_base.point.z)
                    temp_heights.append(base_z)
                except Exception:
                    pass
            
            if temp_heights:
                self.height_reference = np.median(temp_heights)
                self.get_logger().info(f"Height reference (median): {self.height_reference:.3f}m from {len(temp_heights)} detections")

        # Second pass: process and publish all detections
        for d in detections:
            fid = d['id']
            cnt = d['contour']; x,y,w,h = d['bbox']
            gx, gy = d['green_px']; bx, by = d['body_px']

            # get depth
            depth_m, used_x, used_y, method = self.get_depth_for_point(gx, gy, (bx, by))
            if depth_m is None:
                cv2.drawContours(img, [cnt], -1, (80,80,80), 1)
                cv2.putText(img, f"ID:{fid} NO DEPTH", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                self.get_logger().warning(f"ID:{fid} - NO DEPTH at ({gx},{gy})")
                continue

            # compute camera-optical coords
            Xc_opt, Yc_opt, Zc_opt = self.pixel_to_camera_optical(gx, gy, depth_m)

            # create PointStamped in optical frame
            pt_opt = PointStamped()
            pt_opt.header.stamp = self.get_clock().now().to_msg()
            pt_opt.header.frame_id = OPTICAL_FRAME
            pt_opt.point.x = float(Xc_opt)
            pt_opt.point.y = float(Yc_opt)
            pt_opt.point.z = float(Zc_opt)

            # Transform point optical -> camera_link
            try:
                trans_camlink = self.tf_buffer.lookup_transform(
                    CAMERA_LINK_FRAME, OPTICAL_FRAME, 
                    rclpy.time.Time(), 
                    rclpy.duration.Duration(seconds=0.5)
                )
                pt_camlink = do_transform_point(pt_opt, trans_camlink)
                cam_x = float(pt_camlink.point.x)
                cam_y = float(pt_camlink.point.y)
                cam_z = float(pt_camlink.point.z)
            except Exception as e:
                self.get_logger().warning(f"Failed to transform optical->camera_link for ID {fid}: {e}")
                # fallback: use optical coords directly
                cam_x, cam_y, cam_z = Xc_opt, Yc_opt, Zc_opt

            # Publish intermediate TF: camera_link -> cam_<fid>
            cam_child = f"cam_{fid}"
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = CAMERA_LINK_FRAME
            t_cam.child_frame_id = cam_child
            t_cam.transform.translation.x = cam_x
            t_cam.transform.translation.y = cam_y
            t_cam.transform.translation.z = cam_z
            t_cam.transform.rotation.x = 0.0
            t_cam.transform.rotation.y = 0.0
            t_cam.transform.rotation.z = 0.0
            t_cam.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(t_cam)

            # Transform camera_link -> base_link
            try:
                trans_base_camlink = self.tf_buffer.lookup_transform(
                    BASE_FRAME, CAMERA_LINK_FRAME,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.5)
                )
                pt_cl = PointStamped()
                pt_cl.header.stamp = self.get_clock().now().to_msg()
                pt_cl.header.frame_id = CAMERA_LINK_FRAME
                pt_cl.point.x = cam_x
                pt_cl.point.y = cam_y
                pt_cl.point.z = cam_z
                pt_base = do_transform_point(pt_cl, trans_base_camlink)
                base_x = float(pt_base.point.x)
                base_y = float(pt_base.point.y)
                base_z = float(pt_base.point.z)
            except Exception as e:
                self.get_logger().warning(f"Failed to transform camera_link->base_link for ID {fid}: {e}")
                # fallback: try optical->base directly
                try:
                    trans_base_opt = self.tf_buffer.lookup_transform(
                        BASE_FRAME, OPTICAL_FRAME,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.5)
                    )
                    pt_base = do_transform_point(pt_opt, trans_base_opt)
                    base_x = float(pt_base.point.x)
                    base_y = float(pt_base.point.y)
                    base_z = float(pt_base.point.z)
                except Exception as e2:
                    self.get_logger().warning(f"Failed fallback optical->base for ID {fid}: {e2}")
                    continue

            # ROTATION FIX: Rotate 90 degrees clockwise around Z-axis (base_link center)
            if APPLY_ROTATION_FIX:
                # Clockwise 90° rotation matrix around Z:
                # [x']   [ cos(-90°)  -sin(-90°)  0 ] [x]   [ 0   1  0 ] [x]
                # [y'] = [ sin(-90°)   cos(-90°)  0 ] [y] = [-1   0  0 ] [y]
                # [z']   [    0           0       1 ] [z]   [ 0   0  1 ] [z]
                #
                # Result: x' = y,  y' = -x,  z' = z
                base_x_original = base_x
                base_y_original = base_y
                
                base_x = base_y_original
                base_y = -base_x_original
                # base_z remains unchanged
            
            # HEIGHT NORMALIZATION: Fix all fruits at same height
            original_base_z = base_z  # Store original for logging
            if FIX_HEIGHT:
                if HEIGHT_MODE == 'fixed' and FIXED_HEIGHT_VALUE is not None:
                    # Use user-specified fixed height
                    base_z = FIXED_HEIGHT_VALUE
                    
                elif HEIGHT_MODE == 'first':
                    # Use first detection's height as reference
                    if self.height_reference is None:
                        self.height_reference = base_z
                        self.get_logger().info(f"Height reference set to first detection: {self.height_reference:.3f}m")
                    base_z = self.height_reference
                    
                elif HEIGHT_MODE == 'average':
                    # Use pre-computed average/median height
                    if self.height_reference is not None:
                        base_z = self.height_reference
                    # else: keep original height (fallback)
            
            # apply optional corrections (after rotation and height fix)
            if FLIP_X:
                base_x = -base_x
            if FLIP_Y:
                base_y = -base_y
            if FLIP_Z:
                base_z = -base_z
            base_x += EXTRA_TX
            base_y += EXTRA_TY
            base_z += EXTRA_TZ

            # publish final TF: base_link -> <TEAM>_bad_fruit_<fid>
            child_name = f"{TEAM_ID}_bad_fruit_{fid}"
            t_final = TransformStamped()
            t_final.header.stamp = self.get_clock().now().to_msg()
            t_final.header.frame_id = BASE_FRAME
            t_final.child_frame_id = child_name
            t_final.transform.translation.x = base_x-0.2
            t_final.transform.translation.y = base_y-0.1
            t_final.transform.translation.z = base_z-2.08
            t_final.transform.rotation.x = 0.0
            t_final.transform.rotation.y = 0.0
            t_final.transform.rotation.z = 0.0
            t_final.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(t_final)
            
            formula_used = "CORRECTED" if USE_CORRECTED_FORMULA else "ORIGINAL"
            rotation_info = " [+90°CW]" if APPLY_ROTATION_FIX else ""
            height_info = f" [Z:{original_base_z:.3f}→{base_z:.3f}]" if FIX_HEIGHT and abs(original_base_z - base_z) > 0.001 else ""
            self.get_logger().info(
                f"[{formula_used}{rotation_info}{height_info}] Published {child_name}: "
                f"base({base_x:.3f},{base_y:.3f},{base_z:.3f}) "
                f"cam_link({cam_x:.3f},{cam_y:.3f},{cam_z:.3f}) "
                f"depth={depth_m:.3f}m pixel=({gx},{gy}) method={method}"
            )

            # visualization
            cv2.drawContours(img, [cnt], -1, (0,255,0), 2)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(img, (gx, gy), 6, (255,0,0), -1)
            cv2.circle(img, (int(used_x), int(used_y)), 3, (0,255,255), -1)
            
            # Add formula, rotation, and height fix indicator
            formula_color = (0,255,0) if USE_CORRECTED_FORMULA else (0,0,255)
            formula_text = "CORRECTED" if USE_CORRECTED_FORMULA else "ORIGINAL"
            rotation_text = " +90°CW" if APPLY_ROTATION_FIX else ""
            height_text = f" Z={base_z:.3f}" if FIX_HEIGHT else ""
            cv2.putText(img, f"[{formula_text}{rotation_text}{height_text}]", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, formula_color, 2)
            
            cv2.putText(img, f"ID:{fid} {depth_m:.2f}m", (gx+8, gy-6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(img, f"cam({cam_x:.3f},{cam_y:.3f},{cam_z:.3f})", 
                       (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            # Show original vs fixed height if applicable
            if FIX_HEIGHT and abs(original_base_z - base_z) > 0.001:
                cv2.putText(img, f"base({base_x:.3f},{base_y:.3f},{base_z:.3f}) [fixed from {original_base_z:.3f}]", 
                           (x, y+h+32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            else:
                cv2.putText(img, f"base({base_x:.3f},{base_y:.3f},{base_z:.3f})", 
                           (x, y+h+32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # show image
        if SHOW_IMAGE:
            try:
                cv2.imshow(WIN, img)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warning(f"OpenCV display error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FruitsTFCorrected()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTFCorrected")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()