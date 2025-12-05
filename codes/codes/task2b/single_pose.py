#!/usr/bin/env python3
"""
Minimal servo test - Move to a single pose and report final position
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, TransformException
import math
from tf_transformations import quaternion_multiply, quaternion_inverse


class MinimalServoTest(Node):
    def __init__(self):
        super().__init__('minimal_servo_test')
        
        # Publisher
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # TARGET POSE - Change this to test different positions
        # Format: (x, y, z, qx, qy, qz, qw)
        self.target = (0.214, -0.0, 0.7, 0.0, 3.14, 0.0, 0.707)
        
        # Control parameters
        self.tolerance_pos = 0.02       # 20mm
        self.tolerance_ang = 0.15       # ~8.6°
        self.kp_pos = 0.5
        self.kp_ang = 0.6
        self.max_vel = 0.3
        self.max_omega = 0.5
        
        # State
        self.reached = False
        self.start_logged = False
        
        # Timer at 50Hz
        self.timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🎯 MINIMAL SERVO TEST")
        self.get_logger().info(f"Target: ({self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f})")
        self.get_logger().info("=" * 60)
    
    def quaternion_to_angular_error(self, q_current, q_target):
        """Calculate angular error between quaternions"""
        def normalize(q):
            norm = math.sqrt(sum([x**2 for x in q]))
            return [x/norm for x in q] if norm > 1e-10 else [0,0,0,1]
        
        q_curr = normalize(q_current)
        q_targ = normalize(q_target)
        q_inv = quaternion_inverse(q_curr)
        q_error = quaternion_multiply(q_targ, q_inv)
        
        error_mag = 2.0 * math.acos(min(1.0, abs(q_error[3])))
        
        if error_mag < 0.001:
            return 0.0, 0.0, 0.0, error_mag
        
        sin_half = math.sqrt(q_error[0]**2 + q_error[1]**2 + q_error[2]**2)
        if sin_half < 0.001:
            return 0.0, 0.0, 0.0, error_mag
        
        scale = error_mag / sin_half
        return q_error[0]*scale, q_error[1]*scale, q_error[2]*scale, error_mag
    
    def control_loop(self):
        """Main control loop"""
        if self.reached:
            self.twist_pub.publish(Twist())
            return
        
        try:
            # Get current end-effector pose
            trans = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", 
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Current pose
            cur_x = trans.transform.translation.x
            cur_y = trans.transform.translation.y
            cur_z = trans.transform.translation.z
            q = trans.transform.rotation
            cur_quat = [q.x, q.y, q.z, q.w]
            
            # Log start position once
            if not self.start_logged:
                self.get_logger().info(f"Start: ({cur_x:.3f}, {cur_y:.3f}, {cur_z:.3f})")
                self.start_logged = True
            
            # Target pose
            tx, ty, tz, qx, qy, qz, qw = self.target
            target_quat = [qx, qy, qz, qw]
            
            # Position error
            dx, dy, dz = tx - cur_x, ty - cur_y, tz - cur_z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # Orientation error
            wx, wy, wz, ang_error = self.quaternion_to_angular_error(cur_quat, target_quat)
            
            # Check if reached
            if dist <= self.tolerance_pos and ang_error <= self.tolerance_ang:
                self.twist_pub.publish(Twist())
                self.reached = True
                self.get_logger().info("=" * 60)
                self.get_logger().info(f"✅ REACHED TARGET!")
                self.get_logger().info(f"Final position: ({cur_x:.3f}, {cur_y:.3f}, {cur_z:.3f})")
                self.get_logger().info(f"Position error: {dist*1000:.1f}mm")
                self.get_logger().info(f"Angular error: {math.degrees(ang_error):.1f}°")
                self.get_logger().info("=" * 60)
                return
            
            # Compute velocity command
            twist = Twist()
            
            # Position control
            if dist > self.tolerance_pos:
                ux, uy, uz = dx/dist, dy/dist, dz/dist
                v = self.kp_pos * dist
                v = min(v, self.max_vel * (0.7 if dist < 0.1 else 1.0))
                v = max(v, 0.015)
                twist.linear.x = ux * v
                twist.linear.y = uy * v
                twist.linear.z = uz * v
            
            # Orientation control
            if ang_error > self.tolerance_ang:
                twist.angular.x = max(-self.max_omega, min(self.kp_ang * wx, self.max_omega))
                twist.angular.y = max(-self.max_omega, min(self.kp_ang * wy, self.max_omega))
                twist.angular.z = max(-self.max_omega, min(self.kp_ang * wz, self.max_omega))
                
                # Deadband
                if abs(twist.angular.x) < 0.03: twist.angular.x = 0.0
                if abs(twist.angular.y) < 0.03: twist.angular.y = 0.0
                if abs(twist.angular.z) < 0.03: twist.angular.z = 0.0
            
            self.twist_pub.publish(twist)
            
            # Log progress every 50 iterations (1 second)
            if hasattr(self, 'log_counter'):
                self.log_counter += 1
                if self.log_counter % 50 == 0:
                    self.get_logger().info(
                        f"Progress: pos_err={dist*1000:.1f}mm, ang_err={math.degrees(ang_error):.1f}°"
                    )
            else:
                self.log_counter = 0
            
        except TransformException as ex:
            self.get_logger().warn(f"TF error: {ex}", throttle_duration_sec=2.0)
            self.twist_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = MinimalServoTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()