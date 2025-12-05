#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  Advanced Arm Manipulation Node
*  - Picks and places fertiliser_can from obj_3 to obj_6
*  - Picks bad fruits one by one and drops at disposal point P3
*  - Uses TF frames from combined perception node
*
*****************************************************************************************
'''

import rclpy
import math
import time
import tf2_ros
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import quaternion_multiply, quaternion_inverse
from linkattacher_msgs.srv import AttachLink, DetachLink
from std_srvs.srv import Trigger


class AdvancedArmManipulation(Node):
    '''
    Advanced arm manipulation for fertilizer can and bad fruits pick-and-place
    '''
    
    def __init__(self):
        super().__init__('advanced_arm_manipulation')
        
        ############ PUBLISHERS ############
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        ############ SUBSCRIBERS ############
        self.fruit_count_sub = self.create_subscription(
            Int32, '/bad_fruit_count', self.fruit_count_callback, 10
        )
        self.fruit_poses_sub = self.create_subscription(
            PoseArray, '/bad_fruit_poses', self.fruit_poses_callback, 10
        )
        self.aruco_poses_sub = self.create_subscription(
            PoseArray, '/aruco_poses', self.aruco_poses_callback, 10
        )
        
        ############ SERVICE CLIENTS ############
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        
        ############ TF SETUP ############
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # End effector link
        self.ee_link = 'ee_link'  # Using ee_link from reference code
        
        ############ STATE MACHINE ############
        self.state = "INIT"
        self.recorded_poses = {}
        self.aruco_data_received = False
        self.bad_fruits = []  # List of bad fruit IDs (stable IDs from perception)
        self.bad_fruit_poses = {}  # {id: (x, y, z)} - positions from perception
        self.current_fruit_idx = 0
        self.fruit_data_received = False
        
        ############ TEAM ID ############
        self.team_id = 3251
        
        ############ SERVOING PARAMETERS ############
        self.tolerance_pos = 0.12
        self.tolerance_ang = 0.12
        self.kp_pos = 0.5
        self.kp_ang = 0.6
        self.max_vel = 0.8
        self.max_omega = 0.4
        self.min_vel = 0.2
        self.min_omega = 0.5
        
        ############ PICK AND PLACE HEIGHTS ############
        self.approach_height = 0.20  # 20cm above for safe approach
        self.fertilizer_pick_height = 0.06  # Lower for fertilizer can
        self.fruit_pick_height = 0.02  # Very low for bad fruits
        self.disposal_height = 0.25  # Height above disposal point
        
        ############ DISPOSAL POINT P3 ############
        self.disposal_point = {
            'position': (-0.806, 0.010, 0.182),
            'orientation': (-0.684, 0.726, 0.05, 0.008)
        }
        
        ############ OBJECT CONFIGURATION ############
        self.fertilizer_name = 'fertiliser_can'
        self.current_object = None
        self.object_attached = False
        
        ############ STUCK DETECTION ############
        self.stuck_counter = 0
        self.last_pos_error = None
        self.last_orient_error = None
        self.max_stuck_count = 100
        self.stuck_threshold_pos = 0.0005
        self.stuck_threshold_ang = 0.0001
        
        ############ WAITING STATE ############
        self.waiting = False
        self.wait_start_time = None
        self.wait_duration = 1.0
        
        ############ TIMERS ############
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("🦾 ADVANCED ARM MANIPULATION NODE INITIALIZED")
        self.get_logger().info("=" * 70)
        self.get_logger().info("📋 MISSION PLAN:")
        self.get_logger().info("   PHASE 1: Pick fertiliser_can from obj_3 → place at obj_6")
        self.get_logger().info("   PHASE 2: Pick all bad fruits → drop at disposal point P3")
        self.get_logger().info("=" * 70)
        self.get_logger().info("⏳ Initializing...")
    
    
    def aruco_poses_callback(self, msg):
        '''Callback for ArUco marker poses'''
        if len(msg.poses) >= 2:
            # First pose is obj_3, second is obj_6
            pose_3 = msg.poses[0]
            pose_6 = msg.poses[1]
            
            self.recorded_poses[3] = {
                'position': (pose_3.position.x, pose_3.position.y, pose_3.position.z),
                'orientation': (pose_3.orientation.x, pose_3.orientation.y, 
                               pose_3.orientation.z, pose_3.orientation.w)
            }
            
            self.recorded_poses[6] = {
                'position': (pose_6.position.x, pose_6.position.y, pose_6.position.z),
                'orientation': (pose_6.orientation.x, pose_6.orientation.y, 
                               pose_6.orientation.z, pose_6.orientation.w)
            }
            
            if not self.aruco_data_received:
                self.get_logger().info("=" * 70)
                self.get_logger().info("📍 RECEIVED ARUCO MARKER POSES:")
                for marker_id in [3, 6]:
                    x, y, z = self.recorded_poses[marker_id]['position']
                    self.get_logger().info(f"   📌 obj_{marker_id}: ({x:.3f}, {y:.3f}, {z:.3f})")
                self.get_logger().info("=" * 70)
                self.aruco_data_received = True
    
    
    def fruit_count_callback(self, msg):
        '''Callback for bad fruit count'''
        count = msg.data
        if count > 0 and not self.fruit_data_received:
            self.get_logger().info(f"📊 Received fruit count: {count} bad fruits")
    
    
    def fruit_poses_callback(self, msg):
        '''Callback for bad fruit poses'''
        if len(msg.poses) > 0:
            # Clear and rebuild fruit list with stable IDs
            self.bad_fruit_poses.clear()
            
            for idx, pose in enumerate(msg.poses):
                fruit_id = idx + 1  # IDs start from 1
                x = pose.position.x
                y = pose.position.y
                z = pose.position.z
                self.bad_fruit_poses[fruit_id] = (x, y, z)
            
            # Update fruit ID list
            self.bad_fruits = sorted(self.bad_fruit_poses.keys())
            
            if not self.fruit_data_received:
                self.get_logger().info("=" * 70)
                self.get_logger().info(f"📍 RECEIVED {len(self.bad_fruits)} BAD FRUIT POSES:")
                for fid in self.bad_fruits:
                    x, y, z = self.bad_fruit_poses[fid]
                    self.get_logger().info(f"   🍎 Fruit {fid}: ({x:.3f}, {y:.3f}, {z:.3f})")
                self.get_logger().info("=" * 70)
                self.fruit_data_received = True
    
    
    def scan_for_bad_fruits(self):
        '''Scan TF tree for bad fruit frames (backup method)'''
        if self.fruit_data_received and len(self.bad_fruits) > 0:
            # Already have fruits from topic subscription
            return True
        
        # Fallback: scan TF tree
        self.bad_fruits = []
        
        for fruit_id in range(1, 20):
            frame_name = f"{self.team_id}_bad_fruit_{fruit_id}"
            try:
                trans = self.tf_buffer.lookup_transform(
                    'base_link', frame_name, 
                    rclpy.time.Time(), 
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                self.bad_fruits.append(fruit_id)
                pos = trans.transform.translation
                self.bad_fruit_poses[fruit_id] = (pos.x, pos.y, pos.z)
                self.get_logger().info(f"   ✓ Found bad_fruit_{fruit_id} at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
            except TransformException:
                continue
        
        if len(self.bad_fruits) > 0:
            self.get_logger().info(f"📊 Total bad fruits found via TF: {len(self.bad_fruits)}")
            return True
        return False
    
    
    def read_marker_poses_from_tf(self):
        '''Read ArUco marker poses from TF frames (backup method)'''
        if self.aruco_data_received and len(self.recorded_poses) >= 2:
            # Already have poses from topic subscription
            return True
        
        # Fallback: read from TF
        try:
            poses_found = {}
            
            for marker_id in [3, 6]:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        'base_link', f'obj_{marker_id}', 
                        rclpy.time.Time(), 
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                    
                    poses_found[marker_id] = {
                        'position': (
                            trans.transform.translation.x,
                            trans.transform.translation.y,
                            trans.transform.translation.z
                        ),
                        'orientation': (
                            trans.transform.rotation.x,
                            trans.transform.rotation.y,
                            trans.transform.rotation.z,
                            trans.transform.rotation.w
                        )
                    }
                    pos = poses_found[marker_id]['position']
                    self.get_logger().info(f"   ✓ obj_{marker_id} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                except TransformException as ex:
                    self.get_logger().warn(f"   ⚠️ Could not find obj_{marker_id}")
            
            if len(poses_found) == 2:
                self.recorded_poses = poses_found
                self.get_logger().info("✅ Both ArUco markers found via TF")
                return True
            else:
                return False
            
        except Exception as ex:
            self.get_logger().error(f"❌ Error reading markers: {ex}")
            return False
    
    
    def get_downward_orientation(self):
        '''Returns quaternion for end effector facing straight down'''
        r = R.from_euler('xyz', [math.pi, 0, 0])
        quat = r.as_quat()
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
    
    
    def call_attach_service(self, object_name, link_name='body'):
        '''Attach object to gripper'''
        self.get_logger().info(f"🧲 Attaching {object_name}...")
        
        request = AttachLink.Request()
        request.model1_name = object_name
        request.link1_name = link_name
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        
        try:
            future = self.attach_client.call_async(request)
            time.sleep(0.5)
            self.object_attached = True
            self.current_object = object_name
            self.get_logger().info(f"   ✅ {object_name} attached!")
            return True
        except Exception as e:
            self.get_logger().error(f"   ❌ Attach failed: {e}")
            return False
    
    
    def call_detach_service(self, object_name, link_name='body'):
        '''Detach object from gripper'''
        self.get_logger().info(f"🔓 Detaching {object_name}...")
        
        request = DetachLink.Request()
        request.model1_name = object_name
        request.link1_name = link_name
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        
        try:
            future = self.detach_client.call_async(request)
            time.sleep(0.5)
            self.object_attached = False
            self.current_object = None
            self.get_logger().info(f"   ✅ {object_name} detached!")
            return True
        except Exception as e:
            self.get_logger().error(f"   ❌ Detach failed: {e}")
            return False
    
    
    def wait_at_waypoint(self, duration=None):
        '''Start waiting at current position'''
        if duration is None:
            duration = self.wait_duration
        self.waiting = True
        self.wait_start_time = time.time()
        self.wait_duration = duration
        self.twist_pub.publish(Twist())
    
    
    def check_wait_complete(self):
        '''Check if waiting period is complete'''
        if self.waiting:
            if time.time() - self.wait_start_time >= self.wait_duration:
                self.waiting = False
                self.stuck_counter = 0
                self.last_pos_error = None
                self.last_orient_error = None
                return True
            else:
                self.twist_pub.publish(Twist())
                return False
        return True
    
    
    def control_loop(self):
        '''Main control loop - state machine'''
        
        # Handle waiting state
        if self.waiting:
            if not self.check_wait_complete():
                return
        
        # ========== INIT: Get marker poses and scan for fruits ==========
        if self.state == "INIT":
            self.twist_pub.publish(Twist())
            time.sleep(3.0)
            
            # Check if we have data from topics or need to read from TF
            markers_ready = self.aruco_data_received or self.read_marker_poses_from_tf()
            
            if markers_ready:
                # Wait a bit more for fruit data
                if not self.fruit_data_received:
                    self.get_logger().info("⏳ Waiting for fruit data from perception...")
                    time.sleep(2.0)
                    # Try scanning TF as backup
                    if not self.fruit_data_received:
                        self.scan_for_bad_fruits()
                
                self.get_logger().info("=" * 70)
                self.get_logger().info("✅ INITIALIZATION COMPLETE")
                self.get_logger().info(f"   📌 ArUco markers: {len(self.recorded_poses)}")
                self.get_logger().info(f"   🍎 Bad fruits: {len(self.bad_fruits)}")
                self.get_logger().info("=" * 70)
                self.state = "PHASE1_APPROACH_FERTILIZER"
                self.get_logger().info("▶️ PHASE 1: Moving to fertiliser_can at obj_3...")
            else:
                self.get_logger().error("❌ Init failed, retrying in 3s...")
                time.sleep(3.0)
            return
        
        # ========== PHASE 1: FERTILIZER CAN - APPROACH ==========
        if self.state == "PHASE1_APPROACH_FERTILIZER":
            if self.servo_to_marker(3, self.approach_height):
                self.get_logger().info("   ✓ Reached approach position above obj_3")
                self.state = "PHASE1_DESCEND_FERTILIZER"
                self.wait_at_waypoint(1.0)
            return
        
        if self.state == "PHASE1_DESCEND_FERTILIZER":
            if self.servo_to_marker(3, self.fertilizer_pick_height):
                self.get_logger().info("   ✓ Reached pick position")
                self.state = "PHASE1_ATTACH_FERTILIZER"
                self.wait_at_waypoint(1.0)
            return
        
        if self.state == "PHASE1_ATTACH_FERTILIZER":
            if self.call_attach_service(self.fertilizer_name):
                self.state = "PHASE1_LIFT_FERTILIZER"
                self.wait_at_waypoint(1.0)
            else:
                self.get_logger().error("❌ Attach failed, retrying...")
                time.sleep(1.0)
            return
        
        if self.state == "PHASE1_LIFT_FERTILIZER":
            if self.servo_to_marker(3, self.approach_height):
                self.get_logger().info("   ✓ Fertilizer lifted")
                self.state = "PHASE1_APPROACH_OBJ6"
                self.wait_at_waypoint(1.0)
            return
        
        # ========== PHASE 1: FERTILIZER CAN - MOVE TO OBJ_6 ==========
        if self.state == "PHASE1_APPROACH_OBJ6":
            if self.servo_to_marker(6, self.approach_height + 0.15):
                self.get_logger().info("   ✓ Reached obj_6 approach position")
                self.state = "PHASE1_DESCEND_OBJ6"
                self.wait_at_waypoint(1.0)
            return
        
        if self.state == "PHASE1_DESCEND_OBJ6":
            if self.servo_to_marker(6, self.fertilizer_pick_height + 0.05, use_downward_orientation=True):
                self.get_logger().info("   ✓ Reached obj_6 placement position")
                self.state = "PHASE1_DETACH_FERTILIZER"
                self.wait_at_waypoint(1.0)
            return
        
        if self.state == "PHASE1_DETACH_FERTILIZER":
            if self.call_detach_service(self.fertilizer_name):
                self.state = "PHASE1_RETRACT_OBJ6"
                self.wait_at_waypoint(1.0)
            else:
                self.get_logger().error("❌ Detach failed")
                self.state = "PHASE1_RETRACT_OBJ6"
            return
        
        if self.state == "PHASE1_RETRACT_OBJ6":
            if self.servo_to_marker(6, self.approach_height):
                self.get_logger().info("=" * 70)
                self.get_logger().info("✅ PHASE 1 COMPLETE: Fertilizer placed at obj_6")
                self.get_logger().info("=" * 70)
                
                if len(self.bad_fruits) > 0:
                    self.current_fruit_idx = 0
                    self.state = "PHASE2_APPROACH_FRUIT"
                    self.get_logger().info(f"▶️ PHASE 2: Processing {len(self.bad_fruits)} bad fruits...")
                    self.wait_at_waypoint(2.0)
                else:
                    self.state = "COMPLETE"
                    self.get_logger().info("⚠️ No bad fruits detected, mission complete")
            return
        
        # ========== PHASE 2: BAD FRUITS - PICK ==========
        if self.state == "PHASE2_APPROACH_FRUIT":
            fruit_id = self.bad_fruits[self.current_fruit_idx]
            if self.servo_to_fruit(fruit_id, self.approach_height):
                self.get_logger().info(f"   ✓ Reached approach above fruit_{fruit_id}")
                self.state = "PHASE2_DESCEND_FRUIT"
                self.wait_at_waypoint(1.0)
            return
        
        if self.state == "PHASE2_DESCEND_FRUIT":
            fruit_id = self.bad_fruits[self.current_fruit_idx]
            if self.servo_to_fruit(fruit_id, self.fruit_pick_height, use_downward_orientation=True):
                self.get_logger().info(f"   ✓ Reached pick position for fruit_{fruit_id}")
                self.state = "PHASE2_ATTACH_FRUIT"
                self.wait_at_waypoint(1.0)
            return
        
        if self.state == "PHASE2_ATTACH_FRUIT":
            fruit_id = self.bad_fruits[self.current_fruit_idx]
            fruit_name = f"{self.team_id}_bad_fruit_{fruit_id}"
            
            if self.call_attach_service(fruit_name, link_name='link'):
                self.state = "PHASE2_LIFT_FRUIT"
                self.wait_at_waypoint(1.0)
            else:
                self.get_logger().error(f"❌ Failed to attach fruit_{fruit_id}, skipping...")
                self.current_fruit_idx += 1
                if self.current_fruit_idx < len(self.bad_fruits):
                    self.state = "PHASE2_APPROACH_FRUIT"
                else:
                    self.state = "COMPLETE"
            return
        
        if self.state == "PHASE2_LIFT_FRUIT":
            fruit_id = self.bad_fruits[self.current_fruit_idx]
            if self.servo_to_fruit(fruit_id, self.approach_height):
                self.get_logger().info(f"   ✓ Fruit_{fruit_id} lifted")
                self.state = "PHASE2_APPROACH_DISPOSAL"
                self.wait_at_waypoint(1.0)
            return
        
        # ========== PHASE 2: BAD FRUITS - DISPOSAL ==========
        if self.state == "PHASE2_APPROACH_DISPOSAL":
            if self.servo_to_disposal(self.disposal_height):
                self.get_logger().info(f"   ✓ Reached disposal approach")
                self.state = "PHASE2_DESCEND_DISPOSAL"
                self.wait_at_waypoint(1.0)
            return
        
        if self.state == "PHASE2_DESCEND_DISPOSAL":
            if self.servo_to_disposal(0.05, use_downward_orientation=True):
                self.get_logger().info(f"   ✓ Reached disposal position")
                self.state = "PHASE2_DETACH_FRUIT"
                self.wait_at_waypoint(1.0)
            return
        
        if self.state == "PHASE2_DETACH_FRUIT":
            fruit_id = self.bad_fruits[self.current_fruit_idx]
            fruit_name = f"{self.team_id}_bad_fruit_{fruit_id}"
            
            if self.call_detach_service(fruit_name, link_name='link'):
                self.get_logger().info(f"✅ Fruit_{fruit_id} disposed at P3")
                self.state = "PHASE2_RETRACT_DISPOSAL"
                self.wait_at_waypoint(1.0)
            else:
                self.state = "PHASE2_RETRACT_DISPOSAL"
            return
        
        if self.state == "PHASE2_RETRACT_DISPOSAL":
            if self.servo_to_disposal(self.disposal_height):
                self.get_logger().info(f"   ✓ Retracted from disposal")
                
                # Move to next fruit or complete
                self.current_fruit_idx += 1
                if self.current_fruit_idx < len(self.bad_fruits):
                    self.state = "PHASE2_APPROACH_FRUIT"
                    remaining = len(self.bad_fruits) - self.current_fruit_idx
                    self.get_logger().info(f"➡️ Moving to next fruit ({remaining} remaining)...")
                    self.wait_at_waypoint(1.0)
                else:
                    self.state = "COMPLETE"
                    self.get_logger().info("=" * 70)
                    self.get_logger().info(f"✅ PHASE 2 COMPLETE: All {len(self.bad_fruits)} fruits disposed")
                    self.get_logger().info("=" * 70)
            return
        
        # ========== COMPLETE ==========
        if self.state == "COMPLETE":
            self.twist_pub.publish(Twist())
            if not hasattr(self, 'finished_logged'):
                self.get_logger().info("=" * 70)
                self.get_logger().info("🎉 MISSION COMPLETE!")
                self.get_logger().info("=" * 70)
                self.finished_logged = True
            return
    
    
    def servo_to_marker(self, marker_id, z_offset, use_downward_orientation=False):
        '''Servo to ArUco marker with Z offset'''
        if marker_id not in self.recorded_poses:
            self.get_logger().error(f"No pose for marker {marker_id}")
            return False
        
        pose = self.recorded_poses[marker_id]
        tx, ty, tz = pose['position']
        qx, qy, qz, qw = pose['orientation']
        
        target_pos = (tx, ty, tz + z_offset)
        
        # Use downward orientation if specified, otherwise use marker orientation
        if use_downward_orientation:
            target_quat = self.get_downward_orientation()
        else:
            target_quat = [qx, qy, qz, qw]
        
        return self.servo_to_target(target_pos, target_quat)
    
    
    def servo_to_fruit(self, fruit_id, z_offset, use_downward_orientation=False):
        '''Servo to bad fruit with Z offset'''
        frame_name = f"{self.team_id}_bad_fruit_{fruit_id}"
        
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link', frame_name, 
                rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            
            tx = trans.transform.translation.x
            ty = trans.transform.translation.y
            tz = trans.transform.translation.z + z_offset
            
            # Use downward orientation if specified
            if use_downward_orientation:
                target_quat = self.get_downward_orientation()
            else:
                q = trans.transform.rotation
                target_quat = [q.x, q.y, q.z, q.w]
            
            return self.servo_to_target((tx, ty, tz), target_quat)
            
        except TransformException as ex:
            self.get_logger().warn(f"TF error for fruit_{fruit_id}: {ex}")
            return False
    
    
    def servo_to_disposal(self, z_offset, use_downward_orientation=False):
        '''Servo to disposal point P3 with Z offset'''
        tx, ty, tz = self.disposal_point['position']
        target_pos = (tx, ty, tz + z_offset)
        
        # Use downward orientation if specified, otherwise use disposal point orientation
        if use_downward_orientation:
            target_quat = self.get_downward_orientation()
        else:
            target_quat = list(self.disposal_point['orientation'])
        
        return self.servo_to_target(target_pos, target_quat)
    
    
    def servo_to_target(self, target_pos, target_quat):
        '''Generic servo to target position and orientation'''
        try:
            ee_trans = self.tf_buffer.lookup_transform(
                "base_link", self.ee_link, rclpy.time.Time()
            )
            
            cur_x = ee_trans.transform.translation.x
            cur_y = ee_trans.transform.translation.y
            cur_z = ee_trans.transform.translation.z
            q = ee_trans.transform.rotation
            cur_quat = [q.x, q.y, q.z, q.w]
            
            tx, ty, tz = target_pos
            dx, dy, dz = tx - cur_x, ty - cur_y, tz - cur_z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            
            wx, wy, wz, ang_error = self.quaternion_to_angular_error(cur_quat, target_quat)
            
            # Stuck detection
            if self.last_pos_error is not None and self.last_orient_error is not None:
                pos_change = abs(dist - self.last_pos_error)
                orient_change = abs(ang_error - self.last_orient_error)
                
                if pos_change < self.stuck_threshold_pos and orient_change < self.stuck_threshold_ang:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = max(0, self.stuck_counter - 1)
                
                if self.stuck_counter > self.max_stuck_count:
                    self.get_logger().warn(f"⚠️ Stuck detected, considering reached (dist={dist:.4f}m)")
                    self.stuck_counter = 0
                    self.last_pos_error = None
                    self.last_orient_error = None
                    return True
            
            self.last_pos_error = dist
            self.last_orient_error = ang_error
            
            # Check if reached
            if dist <= self.tolerance_pos and ang_error <= self.tolerance_ang:
                self.stuck_counter = 0
                self.last_pos_error = None
                self.last_orient_error = None
                return True
            
            # Compute twist command
            twist = Twist()
            
            # Position control with dynamic scaling
            if dist > self.tolerance_pos:
                ux, uy, uz = dx / dist, dy / dist, dz / dist
                v = self.kp_pos * dist
                if dist > 0.2:
                    v = min(v, self.max_vel)
                else:
                    v = min(v, self.max_vel * 0.5)
                v = max(v, self.min_vel)
                twist.linear.x = ux * v
                twist.linear.y = uy * v
                twist.linear.z = uz * v
            
            # Orientation control
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


def main(args=None):
    rclpy.init(args=args)
    node = AdvancedArmManipulation()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()