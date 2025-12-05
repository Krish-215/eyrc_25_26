#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  Arm Manipulation Node
*  - Performs pick and place operations
*  - Uses TF frames published by perception node
*  - Manages gripper attach/detach operations
*
*****************************************************************************************
'''

import rclpy
import math
import time
import tf2_ros
from rclpy.node import Node
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import quaternion_multiply, quaternion_inverse
from linkattacher_msgs.srv import AttachLink, DetachLink
from std_srvs.srv import Trigger


class ArmManipulation(Node):
    '''
    Arm manipulation and pick-and-place control node
    '''
    
    def __init__(self):
        super().__init__('arm_manipulation')
        
        ############ PUBLISHERS ############
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        ############ SERVICE CLIENTS ############
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        self.get_poses_client = self.create_client(Trigger, '/get_marker_poses')
        
        ############ TF SETUP ############
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # End effector link name - try common options
        self.ee_link = 'tool0'  # Common for UR robots, alternatives: 'ee_link', 'wrist_3_link', 'tool0'
        
        ############ STATE MACHINE ############
        self.state = "INIT"
        self.recorded_poses = {}
        
        ############ STEP TRACKING ############
        self.current_step = 1
        self.max_steps = 5
        
        ############ SERVOING PARAMETERS ############
        self.tolerance_pos = 0.05  # Tighter tolerance
        self.tolerance_ang = 0.1   # Tighter tolerance
        self.kp_pos = 0.5
        self.kp_ang = 0.6
        self.max_vel = 0.7
        self.max_omega = 0.7
        self.min_vel = 0.1
        self.min_omega = 0.1
        
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
        self.max_stuck_count = 25  # Increased from 100
        self.stuck_threshold_pos = 0.0002  # Smaller threshold
        self.stuck_threshold_ang = 0.0005  # Smaller threshold
        
        ############ TIMERS ############
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🦾 ARM MANIPULATION NODE INITIALIZED")
        self.get_logger().info("=" * 60)
        self.get_logger().info("📋 MISSION STEPS:")
        self.get_logger().info("   STEP 1: Get marker poses from perception")
        self.get_logger().info("   STEP 2: Move to ID 3 and pick")
        self.get_logger().info("   STEP 3: Attach fertiliser_can")
        self.get_logger().info("   STEP 4: Move to ID 6")
        self.get_logger().info("   STEP 5: Detach fertiliser_can")
        self.get_logger().info("=" * 60)
        self.get_logger().info("⏳ Waiting to initialize...")
    
    
    def get_marker_poses_from_perception(self):
        '''Request marker poses from perception node'''
        self.get_logger().info("📡 Requesting marker poses from perception node...")
        
        # First, try to read poses directly from TF (simpler approach)
        if self.read_poses_from_tf():
            return True
        
        # If TF read fails, wait and retry
        self.get_logger().warn("⚠️ Could not read from TF, waiting for perception...")
        time.sleep(1.0)
        return self.read_poses_from_tf()
    
    
    def read_poses_from_tf(self):
        '''Read marker poses from TF frames'''
        try:
            poses_found = {}
            
            for marker_id in [3, 6]:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        'base_link', f'obj_{marker_id}', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
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
                    self.get_logger().info(f"   ✓ ID {marker_id}: {poses_found[marker_id]['position']}")
                except TransformException as ex:
                    self.get_logger().warn(f"   ⚠️ Could not find obj_{marker_id}: {ex}")
            
            if len(poses_found) == 2:
                self.recorded_poses = poses_found
                self.get_logger().info("✅ Both markers found in TF")
                return True
            else:
                self.get_logger().warn(f"⚠️ Only found {len(poses_found)}/2 markers")
                return False
            
        except Exception as ex:
            self.get_logger().error(f"❌ Error reading from TF: {ex}")
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
            time.sleep(0.5)
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
            time.sleep(0.5)
            self.object_attached = False
            self.get_logger().info(f"   ✅ DETACH command sent!")
            return True
        except Exception as e:
            self.get_logger().error(f"   ❌ DETACH error: {e}")
            return False
    
    
    def control_loop(self):
        '''Main control loop - step by step execution'''
        
        # ========== INIT: Get marker poses ==========
        if self.state == "INIT":
            self.twist_pub.publish(Twist())
            time.sleep(3.0)  # Wait longer for perception to detect and publish TF
            
            if self.get_marker_poses_from_perception():
                self.get_logger().info("=" * 60)
                self.get_logger().info("✅ INITIALIZATION COMPLETE")
                self.get_logger().info("=" * 60)
                self.current_step = 2
                self.state = "STEP2_APPROACH_ID3"
                self.get_logger().info(f"▶️  Starting STEP 2: Moving to ID 3...")
            else:
                self.get_logger().error("❌ Failed to get marker poses, retrying in 3s...")
                time.sleep(3.0)
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
                time.sleep(1.0)
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
            if not hasattr(self, 'retract_target_set'):
                pose = self.recorded_poses[3]
                tx, ty, tz = pose['position']
                tqx, tqy, tqz, tqw = pose['orientation']
                
                self.retract_target = {
                    'position': (tx, ty + 0.2, tz + self.approach_height + 0.025),
                    'orientation': (tqx, tqy, tqz, tqw)
                }
                self.retract_target_set = True
                self.get_logger().info(f"   Retracting to Y={ty + 0.2:.3f}")
            
            if self.servo_to_retract_target():
                self.get_logger().info("   ✓ Retracted to safe position")
                delattr(self, 'retract_target_set')
                self.state = "STEP4_APPROACH_ID6_HIGH"
                self.get_logger().info(f"▶️  Starting STEP 4: Moving to ID 6 (high approach)...")
            return
        
        # ========== STEP 4: MOVE TO ID 6 ==========
        if self.state == "STEP4_APPROACH_ID6_HIGH":
            target_id = 6
            target_z_offset = self.approach_height + 0.15
            
            if self.servo_to_target(target_id, target_z_offset):
                self.get_logger().info("=" * 60)
                self.get_logger().info("✅ STEP 4 COMPLETE: Reached ID 6 high approach position")
                self.get_logger().info("=" * 60)
                self.state = "STEP5_DETACH"
                self.get_logger().info(f"▶️  Starting STEP 5: Detaching {self.object_name}...")
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
                self.state = "STEP5_RETRACT"
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
        '''Servo to target marker with specified Z offset'''
        try:
            ee_trans = self.tf_buffer.lookup_transform(
                "base_link", self.ee_link, rclpy.time.Time()
            )
            cur_x = ee_trans.transform.translation.x
            cur_y = ee_trans.transform.translation.y
            cur_z = ee_trans.transform.translation.z
            q = ee_trans.transform.rotation
            cur_quat = [q.x, q.y, q.z, q.w]
            
            if marker_id not in self.recorded_poses:
                self.get_logger().error(f"No recorded pose for marker {marker_id}")
                return False
            
            pose = self.recorded_poses[marker_id]
            tx_base, ty_base, tz_base = pose['position']
            tqx, tqy, tqz, tqw = pose['orientation']
            
            tx = tx_base
            ty = ty_base
            tz = tz_base + z_offset
            
            if use_downward_orientation:
                target_quat = self.get_downward_orientation()
            else:
                target_quat = [tqx, tqy, tqz, tqw]
            
            dx, dy, dz = tx - cur_x, ty - cur_y, tz - cur_z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            wx, wy, wz, ang_error = self.quaternion_to_angular_error(cur_quat, target_quat)
            
            if self.last_pos_error is not None:
                pos_change = abs(dist - self.last_pos_error)
                orient_change = abs(ang_error - self.last_orient_error)
                
                # Only count as stuck if BOTH position and orientation are not changing
                if pos_change < self.stuck_threshold_pos and orient_change < self.stuck_threshold_ang:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0  # Reset completely if there's any movement
                
                if self.stuck_counter > self.max_stuck_count:
                    self.get_logger().warn(f"⚠️ Stuck detected after {self.stuck_counter} iterations, considering reached")
                    self.get_logger().warn(f"   Position error: {dist:.4f}m, Angular error: {math.degrees(ang_error):.2f}°")
                    self.stuck_counter = 0
                    self.last_pos_error = None
                    self.last_orient_error = None
                    return True
            
            self.last_pos_error = dist
            self.last_orient_error = ang_error
            
            if dist <= self.tolerance_pos and ang_error <= self.tolerance_ang:
                self.get_logger().info(f"   ✓ Target reached (dist={dist:.4f}m, ang={math.degrees(ang_error):.2f}°)")
                self.stuck_counter = 0
                self.last_pos_error = None
                self.last_orient_error = None
                return True
            
            twist = Twist()
            
            if dist > self.tolerance_pos:
                ux, uy, uz = dx / dist, dy / dist, dz / dist
                v = self.kp_pos * dist
                v = min(v, self.max_vel * 0.6)  # Increased from 0.5
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
        '''Servo to retract target position'''
        try:
            ee_trans = self.tf_buffer.lookup_transform(
                "base_link", self.ee_link, rclpy.time.Time()
            )
            cur_x = ee_trans.transform.translation.x
            cur_y = ee_trans.transform.translation.y
            cur_z = ee_trans.transform.translation.z
            q = ee_trans.transform.rotation
            cur_quat = [q.x, q.y, q.z, q.w]
            
            tx, ty, tz = self.retract_target['position']
            tqx, tqy, tqz, tqw = self.retract_target['orientation']
            target_quat = [tqx, tqy, tqz, tqw]
            
            dx, dy, dz = tx - cur_x, ty - cur_y, tz - cur_z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            wx, wy, wz, ang_error = self.quaternion_to_angular_error(cur_quat, target_quat)
            
            if dist <= self.tolerance_pos and ang_error <= self.tolerance_ang:
                self.get_logger().info(f"   ✓ Retract target reached (dist={dist:.4f}m)")
                self.stuck_counter = 0
                self.last_pos_error = None
                self.last_orient_error = None
                return True
            
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


def main(args=None):
    rclpy.init(args=args)
    node = ArmManipulation()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()