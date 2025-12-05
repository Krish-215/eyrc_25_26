#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import TransformStamped
import time

class ArmWaypointServoNode(Node):
    def __init__(self):
        super().__init__('arm_waypoint_servo_node')

        # Publisher for twist servoing
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # TF buffer/listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Define waypoints (position only here, can extend to orientation later)
        self.waypoints = [
            {'pos': [-0.214, -0.532, 0.557], 'reached': False},
            {'pos': [0.214, 0.532, 0.557], 'reached': False},
            {'pos': [-0.159, 0.501, 0.415], 'reached': False},
            {'pos': [-0.806, 0.010, 0.182], 'reached': False}
        ]
        self.current_wp_idx = 0
        self.tolerance = 0.15  # ±15 cm for position

        # Timer loop
        self.timer = self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        if self.current_wp_idx >= len(self.waypoints):
            self.get_logger().info("All waypoints reached. Stopping node.")
            self.pub.publish(Twist())  # stop
            return

        target = self.waypoints[self.current_wp_idx]['pos']

        try:
            # Get current EE position
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", rclpy.time.Time()
            )
            current_pos = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ]

            # Compute positional errors
            error = [target[i] - current_pos[i] for i in range(3)]
            distance = sum(e**2 for e in error)**0.5

            twist = Twist()
            if distance > self.tolerance:
                # Move proportionally along each axis (simple P controller)
                gain = 0.5  # tweak as needed
                twist.linear.x = gain * error[0]
                twist.linear.y = gain * error[1]
                twist.linear.z = gain * error[2]
                self.get_logger().info(
                    f"Moving to WP{self.current_wp_idx+1} → dist: {distance:.3f} m"
                )
            else:
                # Stop & mark waypoint as reached
                twist = Twist()
                self.pub.publish(twist)
                self.get_logger().info(f"Waypoint {self.current_wp_idx+1} reached. Pausing 1s")
                time.sleep(1.0)
                self.current_wp_idx += 1  # move to next waypoint

            self.pub.publish(twist)

        except TransformException as ex:
            self.get_logger().warn(f"TF not ready: {ex}")


def main(args=None):
    rclpy.init(args=args)
    node = ArmWaypointServoNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()

