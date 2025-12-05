#!/usr/bin/env python3
"""
ROS2 Agricultural Arm Servo Control Node with Attach/Detach
Automated robotic arm for precision agriculture tasks:
1. Fertilizer Management: Pick → Lift → Transport → Drop
2. Bad Fruit Harvesting: Pick individual bad fruits with lift → drop cycles
3. Final Delivery: Transport harvested bad fruits to collection point
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformException
from linkattacher_msgs.srv import AttachLink, DetachLink
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
        
        # Service clients for attach/detach
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        
        # Team ID for bad fruit naming
        self.team_id = 3251
        
        # Define agricultural mission waypoints
        # Format: (x, y, z, qx, qy, qz, qw)
        self.waypoints = [
            # FERTILIZER HANDLING SEQUENCE
            (-0.214, -0.532, 0.657, 0.707, 0.028, 0.034, 0.707),  # 0: Pick fertilizer
            (-0.214, -0.532, 0.7, 0.707, 0.028, 0.034, 0.707),    # 1: Lift fertilizer
            (-0.214, -0.232, 0.7, 0.707, 0.0, 0.0, 0.707),        # 2: Retract from pick area
            (0.214, -0.0, 0.7, 0.0, 3.14, 0.0, 0.707),            # 3: Intermediate position
            (0.65, 0.0, 0.27, 0.0, 3.14, 0.0, 0.0),               # 4: Drop fertilizer
            
            # BAD FRUIT HARVESTING SEQUENCE
            (-0.05, 0.5, 0.5, 0.0, 3.14, 0.0, 0.0),               # 5: Approach bad fruit area (high)
            (-0.03, 0.63, 0.35, 0.0, 3.14, 0.0, 0.0),               # 6: Pick Bad Fruit 1
            (-0.03, 0.63, 0.7, 0.0, 3.14, 0.0, 0.0),              # 7: Lift Bad Fruit 1
            (-0.8, 0.0, 0.2, 0.0, 3.14, 0.0, 0.0),                # 8: Drop Bad Fruit 1
            
            
            (-0.17, 0.5, 0.5, 0.0, 3.14, 0.0, 0.0),              # 9: Approach Bad Fruit 2 (high)
            (-0.17, 0.5, 0.33, 0.0, 3.14, 0.0, 0.0),             # 10: Pick Bad Fruit 2
            (-0.17, 0.5, 0.5, 0.0, 3.14, 0.0, 0.0),              # 11: Lift Bad Fruit 2
            (-0.8, 0.0, 0.2, 0.0, 3.14, 0.0, 0.0),                # 12: Drop Bad Fruit 2
            
            (-0.17, 0.63, 0.5, 0.0, 3.14, 0.0, 0.0),              # 13: Approach Bad Fruit 3 (high)
            (-0.17, 0.63, 0.35, 0.0, 3.14, 0.0, 0.0),             # 14: Pick Bad Fruit 3
            (-0.17, 0.63, 0.5, 0.0, 3.14, 0.0, 0.0),              # 15: Lift Bad Fruit 3
            (-0.8, 0.0, 0.2, 0.0, 3.14, 0.0, 0.0)                 # 16: Drop Bad Fruit 3 (final)
        ]
        
        # Descriptive waypoint names for logging
        self.waypoint_names = [
            "🥤 Fertilizer Pick",           # 0
            "⬆️  Fertilizer Lift",          # 1
            "↩️  Retract",                  # 2
            "➡️  Intermediate",             # 3
            "📦 Fertilizer Drop",           # 4
            "🍎 Bad Fruit Area Approach",   # 5
            "🍎 Bad Fruit 1 Pick",          # 6
            "⬆️  Bad Fruit 1 Lift",         # 7
            "🧺 Bad Fruit 1 Drop",          # 8
            "🍏 Bad Fruit 2 Approach",      # 9
            "🍏 Bad Fruit 2 Pick",          # 10
            "⬆️  Bad Fruit 2 Lift",         # 11
            "🧺 Bad Fruit 2 Drop",          # 12
            "🍊 Bad Fruit 3 Approach",      # 13
            "🍊 Bad Fruit 3 Pick",          # 14
            "⬆️  Bad Fruit 3 Lift",         # 15
            "🧺 Bad Fruit 3 Drop (Final)"   # 16
        ]
        
        # Mission phase tracking
        self.mission_phases = {
            'fertilizer': (0, 4),        # Waypoints 0-4
            'bad_fruit_1': (5, 8),       # Waypoints 5-8
            'bad_fruit_2': (9, 12),      # Waypoints 9-12
            'bad_fruit_3': (13, 16)      # Waypoints 13-16
        }
        
        # Object names for attach/detach
        self.fertilizer_name = 'fertiliser_can'
        self.bad_fruit_names = {
            1: f'bad_fruit',
            2: f'bad_fruit',
            3: f'bad_fruit'
        }
        
        # State management
        self.current_wp = 0
        self.waiting = False
        self.wait_start_time = None
        self.consecutive_no_progress = 0
        self.last_pos_error = None
        self.last_orient_error = None
        self.mission_start_time = time.time()
        self.bad_fruits_harvested = 0
        self.object_attached = False
        self.current_object = None
        
        # Control parameters - optimized for agricultural precision
        self.tolerance_pos = 0.015          # 15mm position tolerance
        self.tolerance_ang = 0.12           # ~6.9° angular tolerance
        
        # Proportional gains
        self.kp_pos = 0.6                   # Position control gain
        self.kp_ang = 0.7                   # Orientation control gain
        
        # Velocity limits
        self.max_vel = 0.45                 # Max linear velocity (m/s)
        self.max_omega = 0.45               # Max angular velocity (rad/s)
        self.min_vel = 0.2                # Min linear velocity threshold
        self.min_omega = 0.2              # Min angular velocity threshold
        
        # Stuck detection parameters
        self.max_stuck_count = 250          # ~12.5 seconds at 50Hz
        self.stuck_threshold_pos = 0.0008   # Position change threshold
        self.stuck_threshold_ang = 0.0015   # Angular change threshold
        
        # Adaptive wait times for different operations
        self.wait_times = {
            'default': 1.5,
            'pick': 2.0,        # Extra time for gripper to secure
            'drop': 2.5,        # Extra time for release
            'bad_fruit_pick': 2.0,  # Time for gentle bad fruit pickup
            'lift': 1.2,        # Quick transition after picking
            'approach': 1.0     # Brief pause at approach positions
        }
        
        # Control loop at 50Hz
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("🌾 AGRICULTURAL ARM SERVO CONTROL NODE INITIALIZED")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"📋 Mission Profile:")
        self.get_logger().info(f"   └─ Phase 1: Fertilizer Management (5 waypoints)")
        self.get_logger().info(f"   └─ Phase 2: Bad Fruit 1 Harvest (4 waypoints: approach → pick → lift → drop)")
        self.get_logger().info(f"   └─ Phase 3: Bad Fruit 2 Harvest (4 waypoints: approach → pick → lift → drop)")
        self.get_logger().info(f"   └─ Phase 4: Bad Fruit 3 Harvest (4 waypoints: approach → pick → lift → drop)")
        self.get_logger().info(f"🎯 Precision: pos={self.tolerance_pos*1000:.0f}mm, ang={math.degrees(self.tolerance_ang):.1f}°")
        self.get_logger().info(f"⚡ Speed limits: linear={self.max_vel}m/s, angular={self.max_omega}rad/s")
        self.get_logger().info("=" * 80)
        self.get_logger().info("🚀 Starting mission...")
    
    def call_attach_service(self, object_name, link_name='body'):
        """Attach object to gripper using magnetic end effector"""
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
            self.get_logger().info(f"   ✅ {object_name} attached successfully!")
            return True
        except Exception as e:
            self.get_logger().error(f"   ❌ Attach failed: {e}")
            return False
    
    def call_detach_service(self, object_name, link_name='body'):
        """Detach object from gripper"""
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
            self.get_logger().info(f"   ✅ {object_name} detached successfully!")
            return True
        except Exception as e:
            self.get_logger().error(f"   ❌ Detach failed: {e}")
            return False
    
    def get_current_phase(self):
        """Determine which mission phase we're in"""
        for phase, (start, end) in self.mission_phases.items():
            if start <= self.current_wp <= end:
                return phase
        return 'complete'
    
    def get_wait_time(self):
        """Get appropriate wait time based on current waypoint"""
        # Fertilizer handling
        if self.current_wp == 0:  # Fertilizer pick
            return self.wait_times['pick']
        elif self.current_wp == 4:  # Fertilizer drop
            return self.wait_times['drop']
        
        # Bad fruit harvesting - pick operations (waypoints 6, 10, 14)
        elif self.current_wp in [6, 10, 14]:
            return self.wait_times['bad_fruit_pick']
        
        # Bad fruit harvesting - drop operations (waypoints 8, 12, 16)
        elif self.current_wp in [8, 12, 16]:
            return self.wait_times['drop']
        
        # Bad fruit harvesting - lift operations (waypoints 7, 11, 15)
        elif self.current_wp in [7, 11, 15]:
            return self.wait_times['lift']
        
        # Approach positions (waypoints 5, 9, 13)
        elif self.current_wp in [5, 9, 13]:
            return self.wait_times['approach']
        
        else:
            return self.wait_times['default']
    
    def quaternion_to_angular_error(self, q_current, q_target):
        """
        Compute angular error between current and target quaternions.
        Returns rotation vector (wx, wy, wz) and error magnitude.
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
                self.get_logger().info(f"   ✓ {self.bad_fruits_harvested} bad fruits harvested and delivered")
                self.get_logger().info(f"   ✓ All operations completed successfully")
                self.get_logger().info("=" * 80)
                self.finished_logged = True
            return
        
        # Handle waiting period at waypoint
        if self.waiting:
            elapsed = time.time() - self.wait_start_time
            wait_time = self.get_wait_time()
            
            if elapsed >= wait_time:
                self.waiting = False
                
                # ATTACH/DETACH LOGIC
                # Attach at pick positions
                if self.current_wp == 0:  # Fertilizer pick
                    self.call_attach_service(self.fertilizer_name, link_name='body')
                elif self.current_wp == 6:  # Bad Fruit 1 pick
                    self.call_attach_service(self.bad_fruit_names[1], link_name='link')
                elif self.current_wp == 10:  # Bad Fruit 2 pick
                    self.call_attach_service(self.bad_fruit_names[2], link_name='link')
                elif self.current_wp == 14:  # Bad Fruit 3 pick
                    self.call_attach_service(self.bad_fruit_names[3], link_name='link')
                
                # Detach at drop positions
                elif self.current_wp == 4:  # Fertilizer drop
                    self.call_detach_service(self.fertilizer_name, link_name='body')
                elif self.current_wp == 8:  # Bad Fruit 1 drop
                    self.call_detach_service(self.bad_fruit_names[1], link_name='link')
                elif self.current_wp == 12:  # Bad Fruit 2 drop
                    self.call_detach_service(self.bad_fruit_names[2], link_name='link')
                elif self.current_wp == 16:  # Bad Fruit 3 drop
                    self.call_detach_service(self.bad_fruit_names[3], link_name='link')
                
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
            
            # Enhanced milestone logging
            if self.current_wp == 0:
                self.get_logger().info(f"✅ {wp_name} | Position reached, attaching...")
            elif self.current_wp == 1:
                self.get_logger().info(f"✅ {wp_name} | Fertilizer secured, clearance achieved")
            elif self.current_wp == 4:
                self.get_logger().info(f"✅ {wp_name} | Position reached, detaching...")
                self.get_logger().info("🌾 Phase 1 complete: Fertilizer management ✓")
            elif self.current_wp == 5:
                self.get_logger().info(f"✅ {wp_name} | Ready for bad fruit harvesting")
            
            # Bad Fruit 1 sequence
            elif self.current_wp == 6:
                self.get_logger().info(f"✅ {wp_name} | Position reached, attaching...")
            elif self.current_wp == 7:
                self.get_logger().info(f"✅ {wp_name} | Bad Fruit 1 secured, transporting")
            elif self.current_wp == 8:
                self.bad_fruits_harvested += 1
                self.get_logger().info(f"✅ {wp_name} | Position reached, detaching... [{self.bad_fruits_harvested}/3]")
            
            # Bad Fruit 2 sequence
            elif self.current_wp == 9:
                self.get_logger().info(f"✅ {wp_name} | Positioning for Bad Fruit 2")
            elif self.current_wp == 10:
                self.get_logger().info(f"✅ {wp_name} | Position reached, attaching...")
            elif self.current_wp == 11:
                self.get_logger().info(f"✅ {wp_name} | Bad Fruit 2 secured, transporting")
            elif self.current_wp == 12:
                self.bad_fruits_harvested += 1
                self.get_logger().info(f"✅ {wp_name} | Position reached, detaching... [{self.bad_fruits_harvested}/3]")
            
            # Bad Fruit 3 sequence
            elif self.current_wp == 13:
                self.get_logger().info(f"✅ {wp_name} | Positioning for Bad Fruit 3")
            elif self.current_wp == 14:
                self.get_logger().info(f"✅ {wp_name} | Position reached, attaching...")
            elif self.current_wp == 15:
                self.get_logger().info(f"✅ {wp_name} | Bad Fruit 3 secured, transporting")
            elif self.current_wp == 16:
                self.bad_fruits_harvested += 1
                self.get_logger().info(f"✅ {wp_name} | Position reached, detaching... [{self.bad_fruits_harvested}/3]")
                self.get_logger().info("🍎 All bad fruits harvested and delivered ✓")
            
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
            
            # Stuck detection
            if self.last_pos_error is not None and self.last_orient_error is not None:
                pos_change = abs(dist - self.last_pos_error)
                orient_change = abs(ang_error - self.last_orient_error)
                
                is_improving = (dist < self.last_pos_error - 0.0003) or (ang_error < self.last_orient_error - 0.0008)
                is_moving = pos_change > self.stuck_threshold_pos or orient_change > self.stuck_threshold_ang
                
                if not is_improving and not is_moving:
                    self.consecutive_no_progress += 1
                else:
                    self.consecutive_no_progress = 0
                
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
            
            # Position control
            if dist > self.tolerance_pos:
                ux, uy, uz = dx / dist, dy / dist, dz / dist
                
                # Adaptive proportional gain
                if dist > 0.3:
                    kp_adaptive = self.kp_pos * 1.1
                elif dist < 0.05:
                    kp_adaptive = self.kp_pos * 0.8
                else:
                    kp_adaptive = self.kp_pos
                
                v = kp_adaptive * dist
                
                # Adaptive velocity limit
                if self.current_wp in [6, 10, 14] and dist < 0.1:
                    v = min(v, self.max_vel * 0.4)
                elif self.current_wp in [5, 9, 13] and dist < 0.15:
                    v = min(v, self.max_vel * 0.5)
                elif dist > 0.2:
                    v = min(v, self.max_vel)
                elif dist > 0.1:
                    v = min(v, self.max_vel * 0.7)
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
                
                if abs(twist.angular.x) < self.min_omega:
                    twist.angular.x = 0.0
                if abs(twist.angular.y) < self.min_omega:
                    twist.angular.y = 0.0
                if abs(twist.angular.z) < self.min_omega:
                    twist.angular.z = 0.0
            
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