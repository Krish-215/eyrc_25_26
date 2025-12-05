#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Perception Node - ArUco Markers + Bad Fruits Detection
Detects ArUco markers (ID 3, 6) and bad fruits, publishing TF frames for all
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, PointStamped, Pose, PoseArray
from std_msgs.msg import Int32
import tf2_ros
from tf2_geometry_msgs import do_transform_point, do_transform_pose
from std_srvs.srv import Trigger
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

# ============ USER CONFIG ============
TEAM_ID = 3251
SHOW_IMAGE = True

# Topics
RGB_TOPIC = '/camera/image_raw'
DEPTH_TOPIC = '/camera/depth/image_raw'
CAMERA_INFO_TOPICS = ['/camera/camera_info', '/camera/depth/camera_info']

# Fruit detection params
LOWER_GRAY = np.array([0, 0, 50])
UPPER_GRAY = np.array([180, 50, 220])
LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([85, 255, 255])
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 5000
TOP_REGION_HEIGHT_RATIO = 0.30
MIN_GREEN_PIXELS_ABS = 700
MIN_GREEN_RATIO = 0.01

# Depth reading params
DEPTH_MEDIAN_KERNEL = 7
DEPTH_MEDIAN_KERNEL_LARGE = 11
VERTICAL_OFFSETS = [0, 3, 6, 9]
MAX_NEAREST_SEARCH_RADIUS = 40
MIN_DEPTH = 0.01
MAX_DEPTH = 3.0

# Frames
OPTICAL_FRAME = 'camera_optical_frame'
CAMERA_LINK_FRAME = 'camera_link'
BASE_FRAME = 'base_link'

# Projection formula settings
USE_CORRECTED_FORMULA = True
APPLY_ROTATION_FIX = True
FIX_HEIGHT = True
HEIGHT_MODE = 'average'  # 'fixed', 'average', or 'first'
FIXED_HEIGHT_VALUE = None

# Optional corrections
FLIP_X = True
FLIP_Y = False
FLIP_Z = False
EXTRA_TX = 0.0
EXTRA_TY = 0.0
EXTRA_TZ = 0.0

# ArUco settings
SIZE_OF_ARUCO_M = 0.15
OBJ_3_OFFSET = (-0.15, 0.08, 0.06)  # x, y, z
OBJ_6_OFFSET = (-0.27, 0.0, 0.15)

# Camera intrinsics
sizeCamX = 1280.0
sizeCamY = 720.0
centerCamX = 642.724365234375
centerCamY = 361.9780578613281
focalX = 915.3003540039062
focalY = 914.0320434570312

WIN = 'Combined Perception: ArUco + Bad Fruits'

# ============ HELPER FUNCTIONS ============
def patch_median_to_meters(depth_img, x, y, k):
    if depth_img is None:
        return None
    h, w = depth_img.shape[:2]
    px = int(np.clip(x, 0, w-1))
    py = int(np.clip(y, 0, h-1))
    r = k // 2
    x0, x1 = max(0, px-r), min(w-1, px+r)
    y0, y1 = max(0, py-r), min(h-1, py+r)
    patch = depth_img[y0:y1+1, x0:x1+1]
    if patch.size == 0:
        return None
    if patch.dtype == np.uint16:
        pf = patch.astype(np.float32)
        pf[pf == 0] = np.nan
        med = float(np.nanmedian(pf)) / 1000.0
    else:
        pf = patch.astype(np.float32)
        pf[pf <= 0.0] = np.nan
        med = float(np.nanmedian(pf))
    if math.isnan(med) or med < MIN_DEPTH or med > MAX_DEPTH:
        return None
    return med

def find_nearest_valid(depth_img, x, y, max_r=MAX_NEAREST_SEARCH_RADIUS):
    if depth_img is None:
        return (None, None, None)
    h, w = depth_img.shape[:2]
    cx = int(np.clip(x, 0, w-1))
    cy = int(np.clip(y, 0, h-1))
    for r in range(1, max_r+1):
        x0, x1 = max(0, cx-r), min(w-1, cx+r)
        y0, y1 = max(0, cy-r), min(h-1, cy+r)
        coords = []
        for xi in range(x0, x1+1):
            coords.append((xi, y0))
            coords.append((xi, y1))
        for yi in range(y0+1, y1):
            coords.append((x0, yi))
            coords.append((x1, yi))
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

# ============ NODE CLASS ============
class CombinedPerception(Node):
    def __init__(self):
        super().__init__('combined_perception')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        
        # Camera params
        self.cam_mat = np.array([[focalX, 0.0, centerCamX], 
                                 [0.0, focalY, centerCamY], 
                                 [0.0, 0.0, 1.0]])
        self.dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Height normalization
        self.height_reference = None
        self.detected_heights = []
        
        # ArUco storage
        self.recorded_poses = {}
        self.markers_detected = set()
        
        # Bad fruit tracking with stable IDs
        self.fruit_database = {}  # {id: {'position': (x,y,z), 'first_seen': timestamp}}
        self.next_fruit_id = 1
        self.position_tolerance = 0.08  # meters - further reduced for 3 distinct fruits
        self.fruit_poses_published = False
        self.initial_scan_complete = False
        self.initial_scan_frames = 0
        self.frames_needed_for_scan = 50  # Increased to 50 frames for better initial scan
        self.min_detections_for_stable = 5  # Must see fruit 5 times before considering stable
        
        # TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscribers
        self.create_subscription(Image, RGB_TOPIC, self.rgb_cb, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, 10)
        for t in CAMERA_INFO_TOPICS:
            try:
                self.create_subscription(CameraInfo, t, self.camera_info_cb, 10)
            except Exception:
                pass
        
        # Publishers
        self.marker_poses_pub = self.create_publisher(PoseArray, '/detected_markers', 10)
        self.aruco_poses_pub = self.create_publisher(PoseArray, '/aruco_poses', 10)
        self.fruit_count_pub = self.create_publisher(Int32, '/bad_fruit_count', 10)
        self.fruit_poses_pub = self.create_publisher(PoseArray, '/bad_fruit_poses', 10)
        
        # Services
        self.get_poses_service = self.create_service(
            Trigger, '/get_marker_poses', self.get_marker_poses_callback
        )
        
        if SHOW_IMAGE:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        
        # Timers
        self.create_timer(0.12, self.tick)
        
        formula_type = "CORRECTED" if USE_CORRECTED_FORMULA else "ORIGINAL"
        rotation_status = "WITH 90° CW rotation" if APPLY_ROTATION_FIX else "NO rotation"
        height_status = f"HEIGHT FIX: {HEIGHT_MODE}" if FIX_HEIGHT else "NO height fix"
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("🤖 COMBINED PERCEPTION NODE INITIALIZED")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"📷 ArUco: Detecting markers ID 3 and 6")
        self.get_logger().info(f"🍎 Fruits: {formula_type} formula, {rotation_status}")
        self.get_logger().info(f"📏 Height: {height_status}")
        self.get_logger().info(f"🔧 Service: /get_marker_poses")
        self.get_logger().info("=" * 70)
    
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
        pass  # Using hardcoded intrinsics
    
    # ============ ARUCO DETECTION ============
    def process_aruco(self, image_copy):
        """Detect and publish ArUco markers"""
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_copy, corners, marker_ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, SIZE_OF_ARUCO_M, self.cam_mat, self.dist_mat
            )
            
            for i, marker_id in enumerate(marker_ids):
                marker_id = marker_id[0]
                
                if marker_id not in [3, 6]:
                    continue
                
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                
                cv2.drawFrameAxes(image_copy, self.cam_mat, self.dist_mat, rvec, tvec, 0.1)
                
                corner_pts = corners[i][0]
                cX = int(np.mean(corner_pts[:, 0]))
                cY = int(np.mean(corner_pts[:, 1]))
                cv2.circle(image_copy, (cX, cY), 5, (255, 0, 255), -1)
                cv2.putText(image_copy, f"ArUco {marker_id}", (cX - 40, cY - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Convert to ROS
                R_marker_to_cam, _ = cv2.Rodrigues(rvec)
                R_opencv_to_ros = np.array([
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0]
                ])
                R_additional = np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0]
                ])
                
                t_ros = R_opencv_to_ros @ tvec
                R_final = R_opencv_to_ros @ R_marker_to_cam @ R_additional
                quat = R.from_matrix(R_final).as_quat()
                
                marker_pose_cam = Pose()
                marker_pose_cam.position.x = float(t_ros[0])
                marker_pose_cam.position.y = float(t_ros[1])
                marker_pose_cam.position.z = float(t_ros[2])
                marker_pose_cam.orientation.x = float(quat[0])
                marker_pose_cam.orientation.y = float(quat[1])
                marker_pose_cam.orientation.z = float(quat[2])
                marker_pose_cam.orientation.w = float(quat[3])
                
                try:
                    base_to_camera = self.tf_buffer.lookup_transform(
                        BASE_FRAME, CAMERA_LINK_FRAME, rclpy.time.Time()
                    )
                    marker_pose_base = do_transform_pose(marker_pose_cam, base_to_camera)
                    
                    # Apply offsets
                    if marker_id == 3:
                        offset = OBJ_3_OFFSET
                    else:
                        offset = OBJ_6_OFFSET
                    
                    # Publish TF
                    t_base = TransformStamped()
                    t_base.header.stamp = self.get_clock().now().to_msg()
                    t_base.header.frame_id = BASE_FRAME
                    t_base.child_frame_id = f'obj_{marker_id}'
                    
                    t_base.transform.translation.x = marker_pose_base.position.x + offset[0]
                    t_base.transform.translation.y = marker_pose_base.position.y + offset[1]
                    t_base.transform.translation.z = marker_pose_base.position.z + offset[2]
                    t_base.transform.rotation = marker_pose_base.orientation
                    
                    self.tf_broadcaster.sendTransform(t_base)
                    
                    # Record pose
                    if marker_id not in self.markers_detected:
                        self.markers_detected.add(marker_id)
                        self.get_logger().info(f"   ✓ ArUco Marker ID {marker_id} detected")
                    
                    self.recorded_poses[marker_id] = {
                        'position': (
                            t_base.transform.translation.x,
                            t_base.transform.translation.y,
                            t_base.transform.translation.z
                        ),
                        'orientation': (
                            t_base.transform.rotation.x,
                            t_base.transform.rotation.y,
                            t_base.transform.rotation.z,
                            t_base.transform.rotation.w
                        )
                    }
                    
                except Exception as ex:
                    pass
        
        # Publish ArUco poses to topic
        self.publish_aruco_data()
    
    # ============ FRUIT DETECTION ============
    def find_or_assign_fruit_id(self, position):
        """
        Find existing fruit ID based on position proximity, or assign new ID.
        Uses improved matching to distinguish 3 separate fruits.
        Returns: fruit_id
        """
        x, y, z = position
        
        # Check if this position matches any existing fruit
        min_distance = float('inf')
        closest_id = None
        
        for fid, data in self.fruit_database.items():
            fx, fy, fz = data['position']
            # Calculate distance in XY plane primarily (fruits are at similar heights)
            distance_xy = math.sqrt((x - fx)**2 + (y - fy)**2)
            distance_3d = math.sqrt((x - fx)**2 + (y - fy)**2 + (z - fz)**2)
            
            # Use XY distance for primary matching, 3D as tiebreaker
            if distance_xy < min_distance:
                min_distance = distance_xy
                closest_id = fid
                closest_3d_dist = distance_3d
        
        # If we found a close match within tolerance, update that fruit
        if closest_id is not None and min_distance < self.position_tolerance:
            data = self.fruit_database[closest_id]
            fx, fy, fz = data['position']
            # Use weighted average for smoother updates (less aggressive)
            alpha = 0.2  # Weight for new measurement (reduced from 0.3)
            data['position'] = (
                alpha * x + (1 - alpha) * fx,
                alpha * y + (1 - alpha) * fy,
                alpha * z + (1 - alpha) * fz
            )
            data['last_seen'] = self.get_clock().now()
            data['detection_count'] = data.get('detection_count', 1) + 1
            return closest_id
        
        # New fruit - assign new ID during initial scan or if we have fewer than 3 fruits
        if not self.initial_scan_complete or len(self.fruit_database) < 3:
            new_id = self.next_fruit_id
            self.fruit_database[new_id] = {
                'position': position,
                'first_seen': self.get_clock().now(),
                'last_seen': self.get_clock().now(),
                'detection_count': 1
            }
            self.next_fruit_id += 1
            
            self.get_logger().info(
                f"   🆕 NEW Bad Fruit {new_id} detected at ({x:.3f}, {y:.3f}, {z:.3f})"
            )
            
            # Log distances to other fruits for debugging
            for other_id, other_data in self.fruit_database.items():
                if other_id != new_id:
                    ox, oy, oz = other_data['position']
                    dist = math.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
                    self.get_logger().info(f"      📏 Distance to Fruit {other_id}: {dist:.3f}m")
            
            return new_id
        
        # After initial scan, only accept detection if reasonably close to an existing fruit
        # This prevents noise but allows for minor position variations
        if closest_id is not None and min_distance < self.position_tolerance * 2.0:
            # Log warning but still update
            data = self.fruit_database[closest_id]
            data['last_seen'] = self.get_clock().now()
            data['detection_count'] = data.get('detection_count', 1) + 1
            return closest_id
        
        # Detection is too far from all known fruits - likely noise, ignore it
        self.get_logger().debug(
            f"   ⚠️ Detection at ({x:.3f}, {y:.3f}, {z:.3f}) rejected "
            f"(min_dist={min_distance:.3f}m to closest fruit)"
        )
        return None
    
    def publish_aruco_data(self):
        """Publish ArUco marker poses to topic"""
        if len(self.recorded_poses) > 0:
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = BASE_FRAME
            
            # Add obj_3 first, then obj_6
            for marker_id in [3, 6]:
                if marker_id in self.recorded_poses:
                    data = self.recorded_poses[marker_id]
                    x, y, z = data['position']
                    qx, qy, qz, qw = data['orientation']
                    
                    pose = Pose()
                    pose.position.x = x
                    pose.position.y = y
                    pose.position.z = z
                    pose.orientation.x = qx
                    pose.orientation.y = qy
                    pose.orientation.z = qz
                    pose.orientation.w = qw
                    
                    pose_array.poses.append(pose)
            
            if len(pose_array.poses) > 0:
                self.aruco_poses_pub.publish(pose_array)
    
    def publish_fruit_data(self):
        """Publish fruit count and poses to topics"""
        # Mark initial scan complete after enough frames
        if not self.initial_scan_complete:
            self.initial_scan_frames += 1
            
            # Log progress every 10 frames during scan
            if self.initial_scan_frames % 10 == 0:
                self.get_logger().info(
                    f"   🔍 Scanning... Frame {self.initial_scan_frames}/{self.frames_needed_for_scan}, "
                    f"Found {len(self.fruit_database)} fruits so far"
                )
            
            if self.initial_scan_frames >= self.frames_needed_for_scan:
                self.initial_scan_complete = True
                self.get_logger().info("=" * 70)
                self.get_logger().info(f"✅ INITIAL FRUIT SCAN COMPLETE: {len(self.fruit_database)} fruits identified")
                for fid in sorted(self.fruit_database.keys()):
                    x, y, z = self.fruit_database[fid]['position']
                    count = self.fruit_database[fid].get('detection_count', 0)
                    self.get_logger().info(f"   🍎 Fruit {fid}: ({x:.3f}, {y:.3f}, {z:.3f}) - {count} detections")
                
                # Show distances between all fruits for verification
                fruit_ids = sorted(self.fruit_database.keys())
                if len(fruit_ids) >= 2:
                    self.get_logger().info("   📏 Inter-fruit distances:")
                    for i, fid1 in enumerate(fruit_ids):
                        for fid2 in fruit_ids[i+1:]:
                            x1, y1, z1 = self.fruit_database[fid1]['position']
                            x2, y2, z2 = self.fruit_database[fid2]['position']
                            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                            self.get_logger().info(f"      Fruit {fid1} ↔ Fruit {fid2}: {dist:.3f}m")
                
                self.get_logger().info("=" * 70)
        
        # Publish count
        count_msg = Int32()
        count_msg.data = len(self.fruit_database)
        self.fruit_count_pub.publish(count_msg)
        
        # Publish poses (only stable fruits with enough detections)
        if len(self.fruit_database) > 0:
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = BASE_FRAME
            
            for fid in sorted(self.fruit_database.keys()):
                data = self.fruit_database[fid]
                x, y, z = data['position']
                
                # Only publish if fruit has been seen multiple times (more stable)
                if data.get('detection_count', 0) >= self.min_detections_for_stable or self.initial_scan_complete:
                    pose = Pose()
                    pose.position.x = x
                    pose.position.y = y
                    pose.position.z = z
                    pose.orientation.w = 1.0
                    
                    pose_array.poses.append(pose)
            
            if len(pose_array.poses) > 0:
                self.fruit_poses_pub.publish(pose_array)
            
            # Log once when all fruits are stable
            if not self.fruit_poses_published and self.initial_scan_complete:
                self.get_logger().info("=" * 70)
                self.get_logger().info(f"📊 PUBLISHING {len(pose_array.poses)} STABLE BAD FRUIT POSES")
                self.get_logger().info("   Topics: /bad_fruit_count, /bad_fruit_poses")
                self.get_logger().info("=" * 70)
                self.fruit_poses_published = True
    
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
                gx_local = int(np.mean(xs))
                gy_local = int(np.mean(ys))
            else:
                gx_local = int(Mg['m10']/Mg['m00'])
                gy_local = int(Mg['m01']/Mg['m00'])
            gx = x + gx_local
            gy = y_top_start + gy_local
            Mb = cv2.moments(cnt)
            if Mb.get('m00', 0) == 0:
                body_cx, body_cy = x + w//2, y + h//2
            else:
                body_cx = int(Mb['m10']/Mb['m00'])
                body_cy = int(Mb['m01']/Mb['m00'])
            res.append({'id': fid, 'contour': cnt, 'bbox': (x,y,w,h),
                        'green_px': (gx, gy), 'body_px': (body_cx, body_cy), 'green_count': green_count})
            fid += 1
        return res
    
    def get_depth_for_point(self, gx, gy, body_px):
        d = patch_median_to_meters(self.depth_image, gx, gy, DEPTH_MEDIAN_KERNEL)
        if d is not None:
            return d, gx, gy, 'patch'
        for off in VERTICAL_OFFSETS:
            if off == 0:
                continue
            y_try = gy + off
            d = patch_median_to_meters(self.depth_image, gx, y_try, DEPTH_MEDIAN_KERNEL)
            if d is not None:
                return d, gx, y_try, f'offset_{off}'
        bx, by = body_px
        d = patch_median_to_meters(self.depth_image, bx, by, DEPTH_MEDIAN_KERNEL)
        if d is not None:
            return d, bx, by, 'body'
        val, px, py = find_nearest_valid(self.depth_image, gx, gy, max_r=MAX_NEAREST_SEARCH_RADIUS)
        if val is not None:
            dref = patch_median_to_meters(self.depth_image, px, py, DEPTH_MEDIAN_KERNEL)
            if dref is not None:
                return dref, px, py, 'nearest_patch'
            return val, px, py, 'nearest_raw'
        d = patch_median_to_meters(self.depth_image, gx, gy, DEPTH_MEDIAN_KERNEL_LARGE)
        if d is not None:
            return d, gx, gy, 'large_patch'
        return None, None, None, 'no_depth'
    
    def pixel_to_camera_optical(self, u, v, distance):
        if USE_CORRECTED_FORMULA:
            x = distance * (float(u) - centerCamX) / focalX
            y = distance * (float(v) - centerCamY) / focalY
            z = float(distance)
        else:
            x = distance * (sizeCamX - float(u) - centerCamX) / focalX
            y = distance * (sizeCamY - float(v) - centerCamY) / focalY
            z = float(distance)
        return x, y, z
    
    def process_fruits(self, img, detections):
        """Process fruit detections and publish TF frames with stable IDs"""
        # Temporary storage for current frame detections
        temp_fruit_data = []
        
        # First pass: collect heights for averaging
        if FIX_HEIGHT and HEIGHT_MODE == 'average':
            temp_heights = []
            for d in detections:
                gx, gy = d['green_px']
                bx, by = d['body_px']
                depth_m, _, _, method = self.get_depth_for_point(gx, gy, (bx, by))
                if depth_m is None:
                    continue
                Xc_opt, Yc_opt, Zc_opt = self.pixel_to_camera_optical(gx, gy, depth_m)
                pt_opt = PointStamped()
                pt_opt.header.stamp = self.get_clock().now().to_msg()
                pt_opt.header.frame_id = OPTICAL_FRAME
                pt_opt.point.x = float(Xc_opt)
                pt_opt.point.y = float(Yc_opt)
                pt_opt.point.z = float(Zc_opt)
                try:
                    trans_camlink = self.tf_buffer.lookup_transform(
                        CAMERA_LINK_FRAME, OPTICAL_FRAME, rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.5)
                    )
                    pt_camlink = do_transform_point(pt_opt, trans_camlink)
                    trans_base_camlink = self.tf_buffer.lookup_transform(
                        BASE_FRAME, CAMERA_LINK_FRAME, rclpy.time.Time(),
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
        
        # Second pass: process all detections and assign stable IDs
        for d in detections:
            detection_id = d['id']  # Temporary detection ID from contour finding
            cnt = d['contour']
            x,y,w,h = d['bbox']
            gx, gy = d['green_px']
            bx, by = d['body_px']
            
            depth_m, used_x, used_y, method = self.get_depth_for_point(gx, gy, (bx, by))
            if depth_m is None:
                cv2.drawContours(img, [cnt], -1, (80,80,80), 1)
                cv2.putText(img, f"NO DEPTH", (x, y-6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                continue
            
            Xc_opt, Yc_opt, Zc_opt = self.pixel_to_camera_optical(gx, gy, depth_m)
            
            pt_opt = PointStamped()
            pt_opt.header.stamp = self.get_clock().now().to_msg()
            pt_opt.header.frame_id = OPTICAL_FRAME
            pt_opt.point.x = float(Xc_opt)
            pt_opt.point.y = float(Yc_opt)
            pt_opt.point.z = float(Zc_opt)
            
            try:
                trans_camlink = self.tf_buffer.lookup_transform(
                    CAMERA_LINK_FRAME, OPTICAL_FRAME, rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.5)
                )
                pt_camlink = do_transform_point(pt_opt, trans_camlink)
                cam_x = float(pt_camlink.point.x)
                cam_y = float(pt_camlink.point.y)
                cam_z = float(pt_camlink.point.z)
            except Exception:
                cam_x, cam_y, cam_z = Xc_opt, Yc_opt, Zc_opt
            
            # Transform to base_link
            try:
                trans_base_camlink = self.tf_buffer.lookup_transform(
                    BASE_FRAME, CAMERA_LINK_FRAME, rclpy.time.Time(),
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
            except Exception:
                continue
            
            # Apply rotation fix
            if APPLY_ROTATION_FIX:
                base_x_original = base_x
                base_y_original = base_y
                base_x = base_y_original
                base_y = -base_x_original
            
            # Height normalization
            original_base_z = base_z
            if FIX_HEIGHT:
                if HEIGHT_MODE == 'fixed' and FIXED_HEIGHT_VALUE is not None:
                    base_z = FIXED_HEIGHT_VALUE
                elif HEIGHT_MODE == 'first':
                    if self.height_reference is None:
                        self.height_reference = base_z
                    base_z = self.height_reference
                elif HEIGHT_MODE == 'average':
                    if self.height_reference is not None:
                        base_z = self.height_reference
            
            # Apply corrections
            if FLIP_X:
                base_x = -base_x
            if FLIP_Y:
                base_y = -base_y
            if FLIP_Z:
                base_z = -base_z
            base_x += EXTRA_TX
            base_y += EXTRA_TY
            base_z += EXTRA_TZ
            
            # Final position with offsets
            final_x = base_x - 0.2
            final_y = base_y - 0.1
            final_z = base_z - 2.08
            
            # Find or assign stable fruit ID based on position
            stable_fruit_id = self.find_or_assign_fruit_id((final_x, final_y, final_z))
            
            # Skip if ID assignment failed
            if stable_fruit_id is None:
                continue
            
            # Store for this frame
            temp_fruit_data.append({
                'stable_id': stable_fruit_id,
                'position': (final_x, final_y, final_z),
                'detection_id': detection_id,
                'bbox': (x, y, w, h),
                'contour': cnt,
                'green_px': (gx, gy),
                'used_px': (used_x, used_y),
                'depth': depth_m
            })
        
        # Now publish TFs using stable IDs
        for fruit_data in temp_fruit_data:
            fid = fruit_data['stable_id']
            final_x, final_y, final_z = fruit_data['position']
            
            # Publish intermediate TF: camera_link -> cam_<stable_id>
            cam_child = f"cam_{fid}"
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = CAMERA_LINK_FRAME
            t_cam.child_frame_id = cam_child
            t_cam.transform.translation.x = 0.0  # Placeholder
            t_cam.transform.translation.y = 0.0
            t_cam.transform.translation.z = 0.0
            t_cam.transform.rotation.x = 0.0
            t_cam.transform.rotation.y = 0.0
            t_cam.transform.rotation.z = 0.0
            t_cam.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t_cam)
            
            # Publish final TF: base_link -> TEAM_bad_fruit_<stable_id>
            child_name = f"{TEAM_ID}_bad_fruit_{fid}"
            t_final = TransformStamped()
            t_final.header.stamp = self.get_clock().now().to_msg()
            t_final.header.frame_id = BASE_FRAME
            t_final.child_frame_id = child_name
            t_final.transform.translation.x = final_x
            t_final.transform.translation.y = final_y
            t_final.transform.translation.z = final_z
            t_final.transform.rotation.x = 0.0
            t_final.transform.rotation.y = 0.0
            t_final.transform.rotation.z = 0.0
            t_final.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t_final)
            
            # Visualization
            x, y, w, h = fruit_data['bbox']
            cnt = fruit_data['contour']
            gx, gy = fruit_data['green_px']
            used_x, used_y = fruit_data['used_px']
            depth_m = fruit_data['depth']
            
            cv2.drawContours(img, [cnt], -1, (0,255,0), 2)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            cv2.circle(img, (gx, gy), 4, (255,0,0), -1)
            cv2.circle(img, (int(used_x), int(used_y)), 2, (0,255,255), -1)
            cv2.putText(img, f"ID:{fid} {depth_m:.2f}m", (x, y-6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        
        # Publish fruit data to topics
        self.publish_fruit_data()
    
    # ============ MAIN TICK ============
    def tick(self):
        if self.cv_image is None:
            return
        
        img = self.cv_image.copy()
        
        # Process ArUco markers
        self.process_aruco(img)
        
        # Process bad fruits
        detections = self.detect_bad_fruits(img)
        self.process_fruits(img, detections)
        
        # Display status
        formula_text = "CORRECTED" if USE_CORRECTED_FORMULA else "ORIGINAL"
        formula_color = (0,255,0) if USE_CORRECTED_FORMULA else (0,0,255)
        rotation_text = " +90°CW" if APPLY_ROTATION_FIX else ""
        height_text = f" Z-fix:{HEIGHT_MODE}" if FIX_HEIGHT else ""
        
        cv2.putText(img, f"[{formula_text}{rotation_text}{height_text}]", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, formula_color, 2)
        
        # ArUco status
        aruco_status = f"ArUco: "
        if 3 in self.markers_detected:
            aruco_status += "[3] "
        if 6 in self.markers_detected:
            aruco_status += "[6]"
        if not self.markers_detected:
            aruco_status += "None"
        cv2.putText(img, aruco_status, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Fruit status
        fruit_status = f"Bad Fruits: {len(self.fruit_database)} (stable IDs)"
        cv2.putText(img, fruit_status, (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show image
        if SHOW_IMAGE:
            try:
                cv2.imshow(WIN, img)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warning(f"OpenCV display error: {e}")
    
    # ============ SERVICE CALLBACK ============
    def get_marker_poses_callback(self, request, response):
        """Service to get recorded ArUco marker poses"""
        if 3 in self.recorded_poses and 6 in self.recorded_poses:
            response.success = True
            response.message = f"Markers detected: ID 3 at {self.recorded_poses[3]['position']}, ID 6 at {self.recorded_poses[6]['position']}"
            self.get_logger().info("✅ Marker poses requested - both markers available")
        else:
            response.success = False
            detected = list(self.recorded_poses.keys())
            response.message = f"Not all markers detected. Available: {detected}"
            self.get_logger().warn(f"⚠️ Marker poses requested but missing markers. Available: {detected}")
        return response


# ============ MAIN ============
def main(args=None):
    rclpy.init(args=args)
    node = CombinedPerception()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Combined Perception Node")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()