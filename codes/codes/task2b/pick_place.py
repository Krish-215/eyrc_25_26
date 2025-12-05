#!/usr/bin/env python3
"""
ROS2 Agricultural Arm Servo Control Node
Automated robotic arm for precision agriculture tasks:
1. Fertilizer Management: Pick → Lift → Transport → Drop
2. Fruit Harvesting: Sequential pickup of 6 fruits from grid pattern
3. Final Delivery: Transport harvested fruits to collection point
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformException
import math
import time
from tf_transformations import quaternion_multiply, quaternion_inverse


class AgriculturalArmNode(Node):
    def __init__(self):
        super().__init__('agricultural_arm_node')
        
        # Publisher for twist servoing commands
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        # TF2 setup for end-effector tracking
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # End effector link name
        self.ee_link = 'ee_link'
        
        # Define agricultural mission waypoints
        # Format: (x, y, z, qx, qy, qz, qw)
        self.waypoints = [
            # FERTILIZER HANDLING SEQUENCE
            (-0.214, -0.532, 0.657, 0.707, 0.028, 0.034, 0.707),  # 0: Pick fertilizer
            (-0.214, -0.532, 0.7, 0.707, 0.028, 0.034, 0.707),    # 1: Lift fertilizer
            (-0.214, -0.232, 0.7, 0.707, 0.0, 0.0, 0.707),        # 2: Retract
            (0.214, -0.0, 0.7, 0.0, 3.14, 0.0, 0.707),            # 3: Intermediate position
            (0.65, 0.0, 0.27, 0.0, 3.14, 0.0, 0.0),               # 4: Drop fertilizer
            
            # FRUIT HARVESTING SEQUENCE (2x3 grid pattern)
            (-0.05, 0.5, 0.5, 0.0, 3.14, 0.0, 0.0),               # 5: Approach fruit area
            (-0.05, 0.5, 0.33, 0.0, 3.14, 0.0, 0.0),              # 6: Fruit 1 (row 1, col 1)
            (-0.03, 0.63, 0.35, 0.0, 3.14, 0.0, 0.0),             # 7: Fruit 2 (row 1, col 2)
            (-0.01, 0.76, 0.35, 0.0, 3.14, 0.0, 0.0),             # 8: Fruit 3 (row 1, col 3)
            (-0.17, 0.5, 0.33, 0.0, 3.14, 0.0, 0.0),              # 9: Fruit 4 (row 2, col 1)
            (-0.17, 0.63, 0.35, 0.0, 3.14, 0.0, 0.0),             # 10: Fruit 5 (row 2, col 2)
            (-0.17, 0.76, 0.35, 0.0, 3.14, 0.0, 0.0),             # 11: Fruit 6 (row 2, col 3)
            
            # FINAL DELIVERY
            (-0.8, 0.0, 0.2, 0.0, 3.14, 0.0, 0.0)                 # 12: Drop harvested fruits
        ]
        
        # Descriptive waypoint names for logging
        self.waypoint_names = [
            "🥤 Fertilizer Pick",
            "⬆️  Fertilizer Lift",
            "↩️  Retract",
            "➡️  Intermediate",
            "📦 Fertilizer Drop",
            "🍎 Fruit Area Approach",
            "🍎 Fruit 1 (R1C1)",
            "🍎 Fruit 2 (R1C2)",
            "🍎 Fruit 3 (R1C3)",
            "🍏 Fruit 4 (R2C1)",
            "🍏 Fruit 5 (R2C2)",
            "🍏 Fruit 6 (R2C3)",
            "🧺 Fruit Collection Drop"
        ]
        
        # Mission phase tracking
        self.mission_phases = {
            'fertilizer': (0, 4),    # Waypoints 0-4
            'fruit_harvest': (5, 11), # Waypoints 5-11
            'final_drop': (12, 12)    # Waypoint 12
        }
        
        # State management
        self.current_wp = 0
        self.waiting = False
        self.wait_start_time = None
        self.consecutive_no_progress = 0
        self.last_pos_error = None
        self.last_orient_error = None
        self.mission_start_time = time.time()
        
        # Control parameters - optimized for agricultural precision
        self.tolerance_pos = 0.015          # 15mm position tolerance (tighter for fruit picking)
        self.tolerance_ang = 0.12           # ~6.9° angular tolerance
        
        # Proportional gains
        self.kp_pos = 0.6                   # Position control gain (increased for responsiveness)
        self.kp_ang = 0.7                   # Orientation control gain
        
        # Velocity limits (safety-constrained for agricultural environment)
        self.max_vel = 0.25                 # Max linear velocity (m/s) - reduced for safety
        self.max_omega = 0.45               # Max angular velocity (rad/s)
        self.min_vel = 0.012                # Min linear velocity threshold
        self.min_omega = 0.025              # Min angular velocity threshold
        
        # Stuck detection parameters
        self.max_stuck_count = 250          # ~12.5 seconds at 50Hz
        self.stuck_threshold_pos = 0.0008   # Position change threshold (tighter)
        self.stuck_threshold_ang = 0.0015   # Angular change threshold
        
        # Adaptive wait times for different operations
        self.wait_times = {
            'default': 1.5,
            'pick': 2.0,        # Extra time for gripper to secure
            'drop': 2.5,        # Extra time for release
            'fruit_pick': 1.8   # Time for gentle fruit pickup
        }
        
        # Control loop at 50Hz
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("🌾 AGRICULTURAL ARM SERVO CONTROL NODE INITIALIZED")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"📋 Mission Profile:")
        self.get_logger().info(f"   └─ Phase 1: Fertilizer Management (5 waypoints)")
        self.get_logger().info(f"   └─ Phase 2: Fruit Harvesting (7 waypoints, 2x3 grid)")
        self.get_logger().info(f"   └─ Phase 3: Collection Delivery (1 waypoint)")
        self.get_logger().info(f"🎯 Precision: pos={self.tolerance_pos*1000:.0f}mm, ang={math.degrees(self.tolerance_ang):.1f}°")
        self.get_logger().info(f"⚡ Speed limits: linear={self.max_vel}m/s, angular={self.max_omega}rad/s")
        self.get_logger().info("=" * 80)
        self.get_logger().info("🚀 Starting mission...")
    
    def get_current_phase(self):
        """Determine which mission phase we're in"""
        for phase, (start, end) in self.mission_phases.items():
            if start <= self.current_wp <= end:
                return phase
        return 'complete'
    
    def get_wait_time(self):
        """Get appropriate wait time based on current waypoint"""
        # Special wait times for critical operations
        if self.current_wp == 0:  # Fertilizer pick
            return self.wait_times['pick']
        elif self.current_wp == 4:  # Fertilizer drop
            return self.wait_times['drop']
        elif 6 <= self.current_wp <= 11:  # Fruit picking
            return self.wait_times['fruit_pick']
        elif self.current_wp == 12:  # Final drop
            return self.wait_times['drop']
        else:
            return self.wait_times['default']
    
    def quaternion_to_angular_error(self, q_current, q_target):
        """
        Compute angular error between current and target quaternions.
        Returns rotation vector (wx, wy, wz) and error magnitude.
        Uses robust normalization and error computation.
        """
        def normalize_quat(q):
            norm = math.sqrt(sum([x**2 for x in q]))
            if norm < 1e-10:
                return [0, 0, 0, 1]
            return [x / norm for x in q]
        
        q_curr_norm = normalize_quat(q_current)
        q_targ_norm = normalize_quat(q_target)
        
        # Compute error quaternion: q_error = q_target * q_current^-1
        q_curr_inv = quaternion_inverse(q_curr_norm)
        q_error = quaternion_multiply(q_targ_norm, q_curr_inv)
        
        # Extract rotation angle
        error_mag = 2.0 * math.acos(min(1.0, abs(q_error[3])))
        
        if error_mag < 0.001:
            return 0.0, 0.0, 0.0, error_mag
        
        # Extract rotation axis
        sin_half = math.sqrt(q_error[0]**2 + q_error[1]**2 + q_error[2]**2)
        if sin_half < 0.001:
            return 0.0, 0.0, 0.0, error_mag
        
        # Scale to get angular velocity components
        scale = error_mag / sin_half
        wx = q_error[0] * scale
        wy = q_error[1] * scale
        wz = q_error[2] * scale
        
        return wx, wy, wz, error_mag
    
    def control_loop(self):
        """Main control loop - executed at 50Hz"""
        
        # Check if all waypoints completed
        if self.current_wp >= len(self.waypoints):
            self.twist_pub.publish(Twist())
            if not hasattr(self, 'finished_logged'):
                mission_duration = time.time() - self.mission_start_time
                self.get_logger().info("=" * 80)
                self.get_logger().info("✅ AGRICULTURAL MISSION COMPLETED SUCCESSFULLY!")
                self.get_logger().info(f"⏱️  Total mission time: {mission_duration:.1f} seconds")
                self.get_logger().info(f"📊 Operations completed:")
                self.get_logger().info(f"   ✓ Fertilizer delivered")
                self.get_logger().info(f"   ✓ 6 fruits harvested")
                self.get_logger().info(f"   ✓ Harvest delivered to collection point")
                self.get_logger().info("=" * 80)
                self.finished_logged = True
            return
        
        # Handle waiting period at waypoint
        if self.waiting:
            elapsed = time.time() - self.wait_start_time
            wait_time = self.get_wait_time()
            
            if elapsed >= wait_time:
                self.waiting = False
                self.current_wp += 1
                self.consecutive_no_progress = 0
                self.last_pos_error = None
                self.last_orient_error = None
                
                if self.current_wp < len(self.waypoints):
                    wp_name = self.waypoint_names[self.current_wp]
                    phase = self.get_current_phase()
                    self.get_logger().info(f"➡️  Moving to: {wp_name} [{phase}]")
            else:
                self.twist_pub.publish(Twist())
            return
        
        # Servo to current waypoint
        reached = self.servo_to_waypoint(self.current_wp)
        
        if reached:
            wp_name = self.waypoint_names[self.current_wp]
            
            # Enhanced milestone logging with context
            if self.current_wp == 0:
                self.get_logger().info(f"✅ {wp_name} | Fertilizer secured, preparing to lift")
            elif self.current_wp == 1:
                self.get_logger().info(f"✅ {wp_name} | Clearance achieved")
            elif self.current_wp == 4:
                self.get_logger().info(f"✅ {wp_name} | Fertilizer delivery complete")
                self.get_logger().info("🌾 Phase 1 complete: Fertilizer management ✓")
            elif self.current_wp == 5:
                self.get_logger().info(f"✅ {wp_name} | Starting fruit harvest sequence")
            elif 6 <= self.current_wp <= 11:
                fruit_num = self.current_wp - 5
                row = 1 if fruit_num <= 3 else 2
                col = ((fruit_num - 1) % 3) + 1
                self.get_logger().info(f"✅ {wp_name} | Fruit {fruit_num}/6 harvested [Row {row}, Col {col}]")
                if self.current_wp == 11:
                    self.get_logger().info("🍎 Phase 2 complete: All fruits harvested ✓")
            elif self.current_wp == 12:
                self.get_logger().info(f"✅ {wp_name} | Harvest delivered to collection point")
            else:
                self.get_logger().info(f"✅ {wp_name}")
            
            # Stop and wait before moving to next waypoint
            self.twist_pub.publish(Twist())
            self.wait_start_time = time.time()
            self.waiting = True
            self.consecutive_no_progress = 0
    
    def servo_to_waypoint(self, wp_index):
        """
        Servo to specified waypoint using visual servoing.
        Returns True when waypoint is reached.
        Implements adaptive control with stuck detection and recovery.
        """
        try:
            # Get current end-effector pose from TF
            ee_trans = self.tf_buffer.lookup_transform(
                "base_link", self.ee_link, 
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Extract current position and orientation
            cur_x = ee_trans.transform.translation.x
            cur_y = ee_trans.transform.translation.y
            cur_z = ee_trans.transform.translation.z
            q = ee_trans.transform.rotation
            cur_quat = [q.x, q.y, q.z, q.w]
            
            # Get target waypoint
            tx, ty, tz, qx, qy, qz, qw = self.waypoints[wp_index]
            target_quat = [qx, qy, qz, qw]
            
            # Compute position error
            dx, dy, dz = tx - cur_x, ty - cur_y, tz - cur_z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # Compute orientation error
            wx, wy, wz, ang_error = self.quaternion_to_angular_error(cur_quat, target_quat)
            
            # Stuck detection - check if making progress
            if self.last_pos_error is not None and self.last_orient_error is not None:
                pos_change = abs(dist - self.last_pos_error)
                orient_change = abs(ang_error - self.last_orient_error)
                
                # Check if we're making meaningful progress
                is_improving = (dist < self.last_pos_error - 0.0003) or (ang_error < self.last_orient_error - 0.0008)
                is_moving = pos_change > self.stuck_threshold_pos or orient_change > self.stuck_threshold_ang
                
                if not is_improving and not is_moving:
                    self.consecutive_no_progress += 1
                else:
                    self.consecutive_no_progress = 0
                
                # If truly stuck, consider it reached and move on
                if self.consecutive_no_progress > self.max_stuck_count:
                    wp_name = self.waypoint_names[wp_index]
                    self.get_logger().warn(
                        f"⚠️  Reached tolerance limit at {wp_name} | "
                        f"pos_err={dist*1000:.1f}mm, ang_err={math.degrees(ang_error):.1f}° | "
                        f"Proceeding to next waypoint"
                    )
                    self.consecutive_no_progress = 0
                    self.last_pos_error = None
                    self.last_orient_error = None
                    return True
                
                # Log progress every 2.5 seconds for longer moves
                if self.consecutive_no_progress % 125 == 0 and self.consecutive_no_progress > 0:
                    wp_name = self.waypoint_names[wp_index]
                    self.get_logger().info(
                        f"📊 Progress to {wp_name}: "
                        f"pos_err={dist*1000:.1f}mm, ang_err={math.degrees(ang_error):.1f}°",
                        throttle_duration_sec=2.0
                    )
            
            self.last_pos_error = dist
            self.last_orient_error = ang_error
            
            # Check if waypoint reached
            if dist <= self.tolerance_pos and ang_error <= self.tolerance_ang:
                self.last_pos_error = None
                self.last_orient_error = None
                return True
            
            # Compute twist command
            twist = Twist()
            
            # === POSITION CONTROL WITH ADAPTIVE GAINS ===
            if dist > self.tolerance_pos:
                # Compute unit direction vector
                ux, uy, uz = dx / dist, dy / dist, dz / dist
                
                # Adaptive proportional gain based on distance
                if dist > 0.3:
                    kp_adaptive = self.kp_pos * 1.1  # Faster for long distances
                elif dist < 0.05:
                    kp_adaptive = self.kp_pos * 0.8  # Gentler for final approach
                else:
                    kp_adaptive = self.kp_pos
                
                # Proportional control with velocity scaling
                v = kp_adaptive * dist
                
                # Adaptive velocity limit based on distance and phase
                phase = self.get_current_phase()
                if phase == 'fruit_harvest' and dist < 0.1:
                    # Extra gentle for fruit picking final approach
                    v = min(v, self.max_vel * 0.4)
                elif dist > 0.2:
                    v = min(v, self.max_vel)
                elif dist > 0.1:
                    v = min(v, self.max_vel * 0.7)
                else:
                    v = min(v, self.max_vel * 0.5)
                
                # Apply minimum velocity threshold to prevent stalling
                v = max(v, self.min_vel)
                
                # Set linear velocities
                twist.linear.x = ux * v
                twist.linear.y = uy * v
                twist.linear.z = uz * v
            
            # === ORIENTATION CONTROL WITH SATURATION ===
            if ang_error > self.tolerance_ang:
                # Proportional control with saturation
                twist.angular.x = max(-self.max_omega, min(self.kp_ang * wx, self.max_omega))
                twist.angular.y = max(-self.max_omega, min(self.kp_ang * wy, self.max_omega))
                twist.angular.z = max(-self.max_omega, min(self.kp_ang * wz, self.max_omega))
                
                # Apply minimum velocity threshold (deadband) to prevent jitter
                if abs(twist.angular.x) < self.min_omega:
                    twist.angular.x = 0.0
                if abs(twist.angular.y) < self.min_omega:
                    twist.angular.y = 0.0
                if abs(twist.angular.z) < self.min_omega:
                    twist.angular.z = 0.0
            
            # Publish velocity command
            self.twist_pub.publish(twist)
            return False
            
        except TransformException as ex:
            self.get_logger().warn(
                f"⚠️  TF lookup failed: {ex}", 
                throttle_duration_sec=2.0
            )
            self.twist_pub.publish(Twist())
            return False


def main(args=None):
    rclpy.init(args=args)
    node = AgriculturalArmNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\n🛑 Keyboard interrupt - Emergency stop initiated")
        node.get_logger().info("🔒 Arm stopped safely")
    finally:
        # Ensure arm stops on shutdown
        stop_cmd = Twist()
        node.twist_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()