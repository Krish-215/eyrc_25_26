#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import TransformStamped

class ArmServoNode(Node):
    def __init__(self):
        super().__init__('arm_servo_node')

        # Publisher for twist servoing
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # TF buffer/listener for getting current pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Define target (example: move end effector to x=0.5m in base_link frame)
        self.target_x = 0.5
        self.tolerance = 0.01   # stop within 1 cm

        # Timer loop
        self.timer = self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        try:
            # Get current transform of ee_link wrt base_link
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", rclpy.time.Time()
            )
            current_x = trans.transform.translation.x
            error = self.target_x - current_x

            twist = Twist()
            if abs(error) > self.tolerance:
                # Move along +X or -X depending on error
                twist.linear.x = 0.05 if error > 0 else -0.05
                self.get_logger().info(f"Moving → error: {error:.3f} m")
            else:
                # Stop once within tolerance
                twist.linear.x = 0.0
                self.get_logger().info("Target reached, stopping")

            # Always publish (servo expects continuous stream)
            self.pub.publish(twist)

        except TransformException as ex:
            self.get_logger().warn(f"TF not ready: {ex}")

def main(args=None):
    rclpy.init(args=args)
    node = ArmServoNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()

