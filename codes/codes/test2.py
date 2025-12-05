#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import TransformStamped
import math

class ArmServoNode(Node):
    def __init__(self):
        super().__init__('arm_servo_node')

        # Publisher for twist commands
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Define waypoints (x, y, z) in base_link frame
        self.waypoints = [
            (0.4, 0.0, 0.3),
            (0.5, 0.2, 0.3),
            (0.4, 0.2, 0.2),
            (0.3, 0.0, 0.2)
        ]
        self.current_wp = 0

        # Parameters
        self.tolerance = 0.01   # meters
        self.max_vel = 0.05     # m/s
        self.kp = 0.6           # proportional gain for velocity scaling

        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

    def control_loop(self):
        if self.current_wp >= len(self.waypoints):
            # All waypoints done
            stop = Twist()
            self.pub.publish(stop)
            self.get_logger().info_once("✅ All waypoints reached. Stopping servo.")
            return

        try:
            # Get current end-effector pose
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", rclpy.time.Time()
            )
            cur_x = trans.transform.translation.x
            cur_y = trans.transform.translation.y
            cur_z = trans.transform.translation.z

            # Target waypoint
            tx, ty, tz = self.waypoints[self.current_wp]

            # Compute distance and direction
            dx = tx - cur_x
            dy = ty - cur_y
            dz = tz - cur_z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            twist = Twist()

            if dist > self.tolerance:
                # Unit direction vector
                ux, uy, uz = dx / dist, dy / dist, dz / dist

                # Proportional velocity (scaled by distance)
                speed = min(self.kp * dist, self.max_vel)
                twist.linear.x = ux * speed
                twist.linear.y = uy * speed
                twist.linear.z = uz * speed

                self.pub.publish(twist)
                self.get_logger().info(f"→ WP{self.current_wp+1}: dist={dist:.3f}m vel={speed:.3f}m/s")

            else:
                # Reached current waypoint
                self.get_logger().info(f"✅ Reached WP{self.current_wp+1}")
                self.current_wp += 1
                stop = Twist()
                self.pub.publish(stop)

        except TransformException as ex:
            self.get_logger().warn(f"TF not ready: {ex}")

def main(args=None):
    rclpy.init(args=args)
    node = ArmServoNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
