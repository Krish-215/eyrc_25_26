#!/usr/bin/env python3
# This script will publish to /delta_twist_cmds.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class ArmServoNode(Node):
    def __init__(self):
        super().__init__('arm_servo_node')
        # Publisher for delta twist commands
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.publish_twist)

        # Command to move forward along X axis
        self.twist_cmd = Twist()
        self.twist_cmd.linear.x = 0.1  # Forward speed (meters/sec)
        self.twist_cmd.linear.y = 0.0
        self.twist_cmd.linear.z = 0.0
        self.twist_cmd.angular.x = 0.5
        self.twist_cmd.angular.y = 0.0
        self.twist_cmd.angular.z = 0.0

    def publish_twist(self):
        self.pub.publish(self.twist_cmd)
        self.get_logger().info(f"Publishing forward twist command")

def main(args=None):
    rclpy.init(args=args)
    node = ArmServoNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
