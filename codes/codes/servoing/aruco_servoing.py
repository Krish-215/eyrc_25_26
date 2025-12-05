#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  Step-by-Step Pick and Place with ArUco Tracking
*  1. Scan and record marker poses (ID 3 and 6)
*  2. Move to ID 3
*  3. Attach fertiliser_can
*  4. Move to ID 6
*  5. Detach fertiliser_can
*
*****************************************************************************************
'''

import rclpy
import sys
import cv2
import math
import time
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, TransformStamped, Pose
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_pose
from tf_transformations import quaternion_multiply, quaternion_inverse
from linkattacher_msgs.srv import AttachLink, DetachLink


##################### COMBINED VISUAL SERVOING CLASS #######################

class PickAndPlaceAruco(Node):
    '''
    Step-by-step pick and place node
    '''
    
    def __init__(self):
        super().__init__('pick_and_place_aruco')
        
        ############ SUBSCRIBERS ############
        self.color_cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self.color_image_cb, 10
        )
        
        ############ PUBLISHERS ############
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        ############ SERVICE CLIENTS ############
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        
        ############ TF SETUP ############
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        
        ############ IMAGE PROCESSING ############
        self.bridge = CvBridge()
        self.cv_image = None
        
        ############ CAMERA PARAMETERS ############
        self.cam_mat = np.array([[915.3003540039062, 0.0, 642.724365234375], 
                                 [0.0, 914.0320434570312, 361.9780578613281], 
                                 [0.0, 0.0, 1.0]])
        self.dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.size_of_aruco_m = 0.15
        
        ############ ARUCO OFFSETS ############
        self.obj_3_offset_x = -0.2
        self.obj_3_offset_y = 0.05
        self.obj_3_offset_z = 0.067
        
        self.obj_6_offset_x = -0.27
        self.obj_6_offset_y = 0.0
        self.obj_6_offset_z = 0.15
        
        ############ STATE MACHINE ############
        self.state = "STEP1_SCANNING"
        self.recorded_poses = {}
        self.markers_detected = set()
        
        ############ STEP TRACKING ############
        self.current_step = 1
        self.max_steps = 5
        
        ############ SCAN PARAMETERS ############
        self.scan_start_time = time.time()
        self.scan_duration = 5.0
        
        ############ SERVOING PARAMETERS ############
        self.tolerance_pos = 0.08
        self.tolerance_ang = 0.15
        self.kp_pos = 0.4
        self.kp_ang = 0.5
        self.max_vel = 0.4
        self.max_omega = 0.4
        self.min_vel = 0.02
        self.min_omega = 0.04
        
        ############ PICK AND PLACE HEIGHTS ############
        self.approach_height = 0.15  # 15cm above marker
        self.pick_height = -0.02     # 2cm below marker for contact
        
        ############ OBJECT CONFIGURATION ############
        self.object_name = 'fertiliser_can'
        self.object_attached = False
        
        ############ STUCK DETECTION ############
        self.stuck_counter = 0
        self.last_pos_error = None
        self.last_orient_error = None
        self.max_stuck_count = 100
        
        ############ TIMERS ############
        self.aruco_timer = self.create_timer(0.1, self.process_aruco)
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🤖 STEP-BY-STEP PICK AND PLACE INITIALIZED")
        self.get_logger().info("=" * 60)
        self.get_logger().info("📋 MISSION STEPS:")
        self.get_logger().info("   STEP 1: Scan and record marker poses")
        self.get_logger().info("   STEP 2: Move to ID 3")
        self.get_logger().info("   STEP 3: Attach fertiliser_can")
        self.get_logger().info("   STEP 4: Move to ID 6")
        self.get_logger().info("   STEP 5: Detach fertiliser_can")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"⏱️  Starting STEP 1: Scanning for {self.scan_duration}s...")
    
    
    def color_image_cb(self, data):
        '''Callback for color camera'''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
    
    
    def process_aruco(self):
        '''Process ArUco markers and publish TF frames'''
        if self.cv_image is None:
            return
        
        image_copy = self.cv_image.copy()
        
        # Detect ArUco markers
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_copy, corners, marker_ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.size_of_aruco_m, self.cam_mat, self.dist_mat
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
                cv2.circle(image_copy, (cX, cY), 5, (0, 255, 0), -1)
                cv2.putText(image_copy, f"ID {marker_id}", (cX - 20, cY - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Convert to ROS coordinate system
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
                        'base_link', 'camera_link', rclpy.time.Time()
                    )
                    marker_pose_base = do_transform_pose(marker_pose_cam, base_to_camera)
                    
                    # Get offsets
                    if marker_id == 3:
                        obj_offset_x = self.obj_3_offset_x
                        obj_offset_y = self.obj_3_offset_y
                        obj_offset_z = self.obj_3_offset_z
                    else:
                        obj_offset_x = self.obj_6_offset_x
                        obj_offset_y = self.obj_6_offset_y
                        obj_offset_z = self.obj_6_offset_z
                    
                    # Publish TF
                    t_base = TransformStamped()
                    t_base.header.stamp = self.get_clock().now().to_msg()
                    t_base.header.frame_id = 'base_link'
                    t_base.child_frame_id = f'obj_{marker_id}'
                    
                    t_base.transform.translation.x = marker_pose_base.position.x + obj_offset_x
                    t_base.transform.translation.y = marker_pose_base.position.y + obj_offset_y
                    t_base.transform.translation.z = marker_pose_base.position.z + obj_offset_z
                    t_base.transform.rotation = marker_pose_base.orientation
                    
                    self.br.sendTransform(t_base)
                    
                    # Record pose during STEP 1
                    if self.state == "STEP1_SCANNING":
                        if marker_id not in self.markers_detected:
                            self.markers_detected.add(marker_id)
                            self.get_logger().info(f"   ✓ Marker ID {marker_id} detected")
                        
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
                    
                except TransformException as ex:
                    pass
        
        # Display info
        step_text = f"STEP {self.current_step}/{self.max_steps}: {self.state}"
        cv2.putText(image_copy, step_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        detected_text = f"Markers: "
        if 3 in self.markers_detected:
            detected_text += "[3] "
        if 6 in self.markers_detected:
            detected_text += "[6]"
        cv2.putText(image_copy, detected_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.object_attached:
            cv2.putText(image_copy, f"[ATTACHED: {self.object_name}]", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Pick and Place - ArUco', image_copy)
        cv2.waitKey(1)
    
    
    def get_downward_orientation(self):
        '''Returns quaternion for end effector facing straight down'''
        # Rotation: X=180° (flip upside down)
        r = R.from_euler('xyz', [math.pi, 0, 0])
        quat = r.as_quat()  # [x, y, z, w]
        return quat
    
    
    def quaternion_to_angular_error(self, q_current, q_target):
        '''Calculate angular error between two quaternions'''
        def normalize_quat(q):
            norm = math.sqrt(sum([x**2 for x in q]))
            return [x / norm for x in q]
        
        q_curr_norm = normalize_quat(q_current)
        q_targ_norm = normalize_quat(q_target)
        q_curr_inv = quaternion_inverse(q_curr_norm)
        q_error = quaternion_multiply(q_targ_norm, q_curr_inv)
        
        error_mag = 2.0 * math.acos(min(1.0, abs(q_error[3])))
        
        if error_mag < 0.001:
            return 0.0, 0.0, 0.0, error_mag
        
        sin_half = math.sqrt(q_error[0]**2 + q_error[1]**2 + q_error[2]**2)
        if sin_half < 0.001:
            return 0.0, 0.0, 0.0, error_mag
        
        scale = error_mag / sin_half
        wx = q_error[0] * scale
        wy = q_error[1] * scale
        wz = q_error[2] * scale
        
        return wx, wy, wz, error_mag
    
    
    def call_attach_service(self):
        '''Call attach service with delay'''
        self.get_logger().info(f"🧲 Calling ATTACH service for {self.object_name}...")
        
        request = AttachLink.Request()
        request.model1_name = self.object_name
        request.link1_name = 'body'
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        
        try:
            future = self.attach_client.call_async(request)
            time.sleep(0.5)  # Give Gazebo time to process attachment
            self.object_attached = True
            self.get_logger().info(f"   ✅ ATTACH command sent!")
            return True
        except Exception as e:
            self.get_logger().error(f"   ❌ ATTACH error: {e}")
            return False
    
    
    def call_detach_service(self):
        '''Call detach service with delay'''
        self.get_logger().info(f"🔓 Calling DETACH service for {self.object_name}...")
        
        request = DetachLink.Request()
        request.model1_name = self.object_name
        request.link1_name = 'body'
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        
        try:
            future = self.detach_client.call_async(request)
            time.sleep(0.5)  # Give Gazebo time to process detachment
            self.object_attached = False
            self.get_logger().info(f"   ✅ DETACH command sent!")
            return True
        except Exception as e:
            self.get_logger().error(f"   ❌ DETACH error: {e}")
            return False
    
    
    def control_loop(self):
        '''Main control loop - step by step execution'''
        
        # ========== STEP 1: SCANNING ==========
        if self.state == "STEP1_SCANNING":
            self.twist_pub.publish(Twist())
            elapsed = time.time() - self.scan_start_time
            
            if elapsed >= self.scan_duration:
                if 3 in self.recorded_poses and 6 in self.recorded_poses:
                    self.get_logger().info("=" * 60)
                    self.get_logger().info("✅ STEP 1 COMPLETE: Both markers recorded")
                    self.get_logger().info(f"   ID 3: {self.recorded_poses[3]['position']}")
                    self.get_logger().info(f"   ID 6: {self.recorded_poses[6]['position']}")
                    self.get_logger().info("=" * 60)
                    self.current_step = 2
                    self.state = "STEP2_APPROACH_ID3"
                    self.get_logger().info(f"▶️  Starting STEP 2: Moving to ID 3...")
                else:
                    self.get_logger().error("❌ STEP 1 FAILED: Markers not detected, rescanning...")
                    self.scan_start_time = time.time()
                    self.markers_detected.clear()
                    self.recorded_poses.clear()
            return
        
        # ========== STEP 2: MOVE TO ID 3 ==========
        if self.state == "STEP2_APPROACH_ID3":
            target_id = 3
            target_z_offset = self.approach_height
            
            if self.servo_to_target(target_id, target_z_offset):
                self.get_logger().info("=" * 60)
                self.get_logger().info("✅ STEP 2 COMPLETE: Reached ID 3 approach position")
                self.get_logger().info("=" * 60)
                self.state = "STEP2_DESCEND_ID3"
                self.get_logger().info("   ⬇️  Descending to pick position...")
            return
        
        if self.state == "STEP2_DESCEND_ID3":
            target_id = 3
            target_z_offset = self.pick_height
            
            if self.servo_to_target(target_id, target_z_offset):
                self.get_logger().info("   ✓ Reached pick position")
                self.get_logger().info("=" * 60)
                self.current_step = 3
                self.state = "STEP3_ATTACH"
                self.get_logger().info(f"▶️  Starting STEP 3: Attaching {self.object_name}...")
                time.sleep(1.0)  # Wait for stability
            return
        
        # ========== STEP 3: ATTACH ==========
        if self.state == "STEP3_ATTACH":
            self.twist_pub.publish(Twist())
            
            if self.call_attach_service():
                self.get_logger().info("=" * 60)
                self.get_logger().info("✅ STEP 3 COMPLETE: Object attached")
                self.get_logger().info("=" * 60)
                self.current_step = 4
                self.state = "STEP3_LIFT"
                self.get_logger().info("   ⬆️  Lifting object...")
            else:
                self.get_logger().error("❌ STEP 3 FAILED: Could not attach, retrying...")
                time.sleep(1.0)
            return
        
        if self.state == "STEP3_LIFT":
            target_id = 3
            target_z_offset = self.approach_height
            
            if self.servo_to_target(target_id, target_z_offset):
                self.get_logger().info("   ✓ Object lifted")
                self.state = "STEP3_RETRACT"
                self.get_logger().info("   ⬅️  Retracting back...")
            return
        
        if self.state == "STEP3_RETRACT":
            # Move back in Y axis to a safe position
            if not hasattr(self, 'retract_target_set'):
                pose = self.recorded_poses[3]
                tx, ty, tz = pose['position']
                tqx, tqy, tqz, tqw = pose['orientation']
                
                # Store retract target: same X and Z, but move back in Y by 0.3m
                self.retract_target = {
                    'position': (tx, ty + 0.3, tz + self.approach_height ),
                    'orientation': (tqx, tqy, tqz, tqw)
                }
                self.retract_target_set = True
                self.get_logger().info(f"   Retracting to Y={ty + 0.3:.3f}")
            
            if self.servo_to_retract_target():
                self.get_logger().info("   ✓ Retracted to safe position")
                delattr(self, 'retract_target_set')
                self.state = "STEP4_APPROACH_ID6_HIGH"
                self.get_logger().info(f"▶️  Starting STEP 4: Moving to ID 6 (high approach)...")
            return
        
        # ========== STEP 4: MOVE TO ID 6 ==========
        if self.state == "STEP4_APPROACH_ID6_HIGH":
            target_id = 6
            target_z_offset = self.approach_height + 0.15  # Higher approach (30cm above marker)
            
            if self.servo_to_target(target_id, target_z_offset):
                self.get_logger().info("=" * 60)
                self.get_logger().info("✅ STEP 4 COMPLETE: Reached ID 6 high approach position")
                self.get_logger().info("=" * 60)
                self.state = "STEP4_ORIENT_DOWNWARD"
                self.get_logger().info("   🔄 Orienting end effector downward...")
            return
        
        if self.state == "STEP4_ORIENT_DOWNWARD":
            target_id = 6
            target_z_offset = self.approach_height + 0.15  # Stay at high position while orienting
            
            if self.servo_to_target(target_id, target_z_offset, use_downward_orientation=True):
                self.get_logger().info("   ✓ End effector oriented downward")
                self.state = "STEP4_DESCEND_ID6"
                self.get_logger().info("   ⬇️  Descending to place position...")
            return
        
        if self.state == "STEP4_DESCEND_ID6":
            target_id = 6
            target_z_offset = 0.05  # Place 5cm above
            
            if self.servo_to_target(target_id, target_z_offset, use_downward_orientation=True):
                self.get_logger().info("   ✓ Reached place position")
                self.get_logger().info("=" * 60)
                self.current_step = 5
                self.state = "STEP5_DETACH"
                self.get_logger().info(f"▶️  Starting STEP 5: Detaching {self.object_name}...")
                time.sleep(1.0)
            return
        
        # ========== STEP 5: DETACH ==========
        if self.state == "STEP5_DETACH":
            self.twist_pub.publish(Twist())
            
            if self.call_detach_service():
                self.get_logger().info("=" * 60)
                self.get_logger().info("✅ STEP 5 COMPLETE: Object detached")
                self.get_logger().info("=" * 60)
                self.state = "STEP5_RETRACT"
                self.get_logger().info("   ⬆️  Retracting...")
            else:
                self.get_logger().error("❌ STEP 5 FAILED: Could not detach")
                self.state = "STEP5_RETRACT"  # Continue anyway
            return
        
        if self.state == "STEP5_RETRACT":
            target_id = 6
            target_z_offset = self.approach_height
            
            if self.servo_to_target(target_id, target_z_offset, use_downward_orientation=True):
                self.get_logger().info("   ✓ Retracted")
                self.state = "COMPLETE"
                self.get_logger().info("=" * 60)
                self.get_logger().info("🎉 ALL STEPS COMPLETE!")
                self.get_logger().info("=" * 60)
            return
        
        # ========== COMPLETE ==========
        if self.state == "COMPLETE":
            self.twist_pub.publish(Twist())
            return
    
    
    def servo_to_target(self, marker_id, z_offset, use_downward_orientation=False):
        '''Servo to target marker with specified Z offset. Returns True when reached.'''
        try:
            # Get current EE pose
            ee_trans = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", rclpy.time.Time()
            )
            cur_x = ee_trans.transform.translation.x
            cur_y = ee_trans.transform.translation.y
            cur_z = ee_trans.transform.translation.z
            q = ee_trans.transform.rotation
            cur_quat = [q.x, q.y, q.z, q.w]
            
            # Get target pose from recorded
            if marker_id not in self.recorded_poses:
                self.get_logger().error(f"No recorded pose for marker {marker_id}")
                return False
            
            pose = self.recorded_poses[marker_id]
            tx_base, ty_base, tz_base = pose['position']
            tqx, tqy, tqz, tqw = pose['orientation']
            
            # Apply Z offset
            tx = tx_base
            ty = ty_base
            tz = tz_base + z_offset
            
            # Use downward orientation for marker 6 placement
            if use_downward_orientation:
                target_quat = self.get_downward_orientation()
            else:
                target_quat = [tqx, tqy, tqz, tqw]
            
            # Calculate errors
            dx, dy, dz = tx - cur_x, ty - cur_y, tz - cur_z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            wx, wy, wz, ang_error = self.quaternion_to_angular_error(cur_quat, target_quat)
            
            # Stuck detection
            if self.last_pos_error is not None:
                pos_change = abs(dist - self.last_pos_error)
                orient_change = abs(ang_error - self.last_orient_error)
                if pos_change < 0.0005 and orient_change < 0.001:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = max(0, self.stuck_counter - 1)
                
                if self.stuck_counter > self.max_stuck_count:
                    self.get_logger().warn(f"⚠️ Stuck detected, considering reached")
                    self.stuck_counter = 0
                    return True
            
            self.last_pos_error = dist
            self.last_orient_error = ang_error
            
            # Check if reached
            if dist <= self.tolerance_pos and ang_error <= self.tolerance_ang:
                self.get_logger().info(f"   ✓ Target reached (dist={dist:.3f}m, ang={math.degrees(ang_error):.1f}°)")
                self.stuck_counter = 0
                self.last_pos_error = None
                self.last_orient_error = None
                return True
            
            # Generate twist command
            twist = Twist()
            
            if dist > self.tolerance_pos:
                ux, uy, uz = dx / dist, dy / dist, dz / dist
                v = self.kp_pos * dist
                v = min(v, self.max_vel * 0.5)  # Slow and careful
                v = max(v, self.min_vel)
                twist.linear.x = ux * v
                twist.linear.y = uy * v
                twist.linear.z = uz * v
            
            if ang_error > self.tolerance_ang:
                twist.angular.x = max(-self.max_omega, min(self.kp_ang * wx, self.max_omega))
                twist.angular.y = max(-self.max_omega, min(self.kp_ang * wy, self.max_omega))
                twist.angular.z = max(-self.max_omega, min(self.kp_ang * wz, self.max_omega))
                
                if abs(twist.angular.x) < self.min_omega: twist.angular.x = 0.0
                if abs(twist.angular.y) < self.min_omega: twist.angular.y = 0.0
                if abs(twist.angular.z) < self.min_omega: twist.angular.z = 0.0
            
            self.twist_pub.publish(twist)
            return False
            
        except TransformException as ex:
            self.get_logger().warn(f'TF error: {ex}')
            self.twist_pub.publish(Twist())
            return False
    
    
    def servo_to_retract_target(self):
        '''Servo to retract target position. Returns True when reached.'''
        try:
            # Get current EE pose
            ee_trans = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", rclpy.time.Time()
            )
            cur_x = ee_trans.transform.translation.x
            cur_y = ee_trans.transform.translation.y
            cur_z = ee_trans.transform.translation.z
            q = ee_trans.transform.rotation
            cur_quat = [q.x, q.y, q.z, q.w]
            
            # Get retract target
            tx, ty, tz = self.retract_target['position']
            tqx, tqy, tqz, tqw = self.retract_target['orientation']
            target_quat = [tqx, tqy, tqz, tqw]
            
            # Calculate errors
            dx, dy, dz = tx - cur_x, ty - cur_y, tz - cur_z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            wx, wy, wz, ang_error = self.quaternion_to_angular_error(cur_quat, target_quat)
            
            # Check if reached
            if dist <= self.tolerance_pos and ang_error <= self.tolerance_ang:
                self.get_logger().info(f"   ✓ Retract target reached (dist={dist:.3f}m)")
                return True
            
            # Generate twist command
            twist = Twist()
            
            if dist > self.tolerance_pos:
                ux, uy, uz = dx / dist, dy / dist, dz / dist
                v = self.kp_pos * dist
                v = min(v, self.max_vel * 0.5)
                v = max(v, self.min_vel)
                twist.linear.x = ux * v
                twist.linear.y = uy * v
                twist.linear.z = uz * v
            
            if ang_error > self.tolerance_ang:
                twist.angular.x = max(-self.max_omega, min(self.kp_ang * wx, self.max_omega))
                twist.angular.y = max(-self.max_omega, min(self.kp_ang * wy, self.max_omega))
                twist.angular.z = max(-self.max_omega, min(self.kp_ang * wz, self.max_omega))
                
                if abs(twist.angular.x) < self.min_omega: twist.angular.x = 0.0
                if abs(twist.angular.y) < self.min_omega: twist.angular.y = 0.0
                if abs(twist.angular.z) < self.min_omega: twist.angular.z = 0.0
            
            self.twist_pub.publish(twist)
            return False
            
        except TransformException as ex:
            self.get_logger().warn(f'TF error: {ex}')
            self.twist_pub.publish(Twist())
            return False


##################### MAIN FUNCTION #######################

def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceAruco()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()