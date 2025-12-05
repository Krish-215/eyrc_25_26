#!/usr/bin/env python3
# This script will publish to /delta_joint_cmds.
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

class JointServoNode(Node):
    def __init__(self):
        super().__init__('joint_servo_node')
        self.pub = self.create_publisher(Float64MultiArray, '/delta_joint_cmds', 10)

        self.timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(self.timer_period, self.publish_joint_cmd)

        # UR5 has 6 joints → we send a 6-element array
        self.delta_cmd = Float64MultiArray()
        self.delta_cmd.data = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]  
        # Moves only base joint (joint1) by small increments

    def publish_joint_cmd(self):
        self.pub.publish(self.delta_cmd)
        self.get_logger().info(f"Publishing base joint delta")

def main(args=None):
    rclpy.init(args=args)
    node = JointServoNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
