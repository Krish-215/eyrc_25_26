# Team ID:          3251
# Theme:            Krishi coBot
# Author List:      Hari Sathish S , Krishnaswamy S
# Filename:         ebot_nav_task2a.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math


class Navigator(Node):

    def __init__(self):
        super().__init__('navigator')


        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)


        self.create_subscription(LaserScan, '/scan', self.scanner, 10)
        self.create_subscription(Odometry, '/odom', self.odom_listener, 10)

       
        self.target_points = [
            [0.11, -5.54, 0.20],
            [0.26, -1.95, 1.57],
            [0.2, 0.41, 1.89],
            [-0.35, 1.19, 2.80],
            [-0.77, 1.15, 3.12],
            [-0.75, 1.01, -3.06],
            [-1.0, 1.02, -2.80],
            [-1.48, 0.07, -1.57],
            [-1.53, -6.61, -1.57]
        ]
        self.active_goal = 0
        self.current_pose = [-1.534, -6.615, 1.57]
        self.obstacle_ahead = False


        self.create_timer(0.1, self.move_to_goal)


    def scanner(self, scan_data: LaserScan):

        if not scan_data.ranges:
            return

        ranges = scan_data.ranges
        mid = len(ranges) // 2

        left = max(0, mid - 15)
        right = min(len(ranges), mid + 15)

        front_window = ranges[left:right]

        valid = [r for r in front_window if r is not None and not math.isinf(r)]

        if not valid:
            self.obstacle_ahead = False
            return

        min_front_distance = min(valid)

        self.obstacle_ahead = min_front_distance < 0.6


    def odom_listener(self, odom_data: Odometry):
        pos = odom_data.pose.pose.position
        ori = odom_data.pose.pose.orientation

        _, _, yaw = self.ori_yaw(ori)

        self.current_pose = [pos.x, pos.y, yaw]

    def ori_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (0, 0, yaw)


    def move_to_goal(self):

        if self.active_goal >= len(self.target_points):
            stop_msg = Twist()
            self.velocity_publisher.publish(stop_msg)
            self.get_logger().info("All waypoints reached. Stopped.")
            return

        target = self.target_points[self.active_goal]

        x_diff = target[0] - self.current_pose[0]
        y_diff = target[1] - self.current_pose[1]
        distance = math.hypot(x_diff, y_diff)

        target_yaw = math.atan2(y_diff, x_diff)
        yaw_error = (target_yaw - self.current_pose[2] + math.pi) % (2 * math.pi) - math.pi

        velocity_msg = Twist()

        # If obstacle ahead, avoid
        if self.obstacle_ahead:
            velocity_msg.linear.x = -0.1
            velocity_msg.angular.z = 0.5
            self.get_logger().warn("Obstacle detected ahead!")
        # Move towards waypoint
        elif distance > 0.25:
            velocity_msg.linear.x = min(0.3, 0.8 * distance)
            velocity_msg.angular.z = 1.0 * yaw_error
        else:
            # Close to waypoint → adjust orientation
            yaw_diff = (target[2] - self.current_pose[2] + math.pi) % (2 * math.pi) - math.pi

            if abs(yaw_diff) > math.radians(8):
                velocity_msg.angular.z = 0.7 * yaw_diff
            else:
                self.get_logger().info(f"Reached waypoint {self.active_goal + 1}")
                self.active_goal += 1

        self.velocity_publisher.publish(velocity_msg)



def main(args=None):
    rclpy.init(args=args)
    navigator_node = Navigator()
    rclpy.spin(navigator_node)
    navigator_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

