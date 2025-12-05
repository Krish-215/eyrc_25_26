# Team ID:          3251
# Theme:            Krishi coBot
# Author List:      Hari Sathish S , Krishnaswamy S
# Filename:         ebot_nav_task2a.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math
import time

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        
        # Publishers and Subscribers
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detection_publisher = self.create_publisher(String, '/detection_status', 10)
        self.create_subscription(LaserScan, '/scan', self.scanner, 10)
        self.create_subscription(Odometry, '/odom', self.odom_listener, 10)
        self.create_subscription(String, '/shape_detected', self.shape_detection_callback, 10)
        
        # Original navigation waypoints - DO NOT CHANGE
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
        
        # Shape detection state management
        self.shape_detected_flag = False
        self.shape_stop_start = None
        self.detected_shape_info = None
        self.published_detections = set()
        
        # Create timer for navigation
        self.create_timer(0.1, self.move_to_goal)
        
        self.get_logger().info("Navigator initialized with shape detection support.")
    
    def scanner(self, scan_data: LaserScan):
        """Original scanner logic"""
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
        """Original odom listener logic"""
        pos = odom_data.pose.pose.position
        ori = odom_data.pose.pose.orientation
        _, _, yaw = self.ori_yaw(ori)
        self.current_pose = [pos.x, pos.y, yaw]
    
    def ori_yaw(self, q):
        """Original orientation to yaw conversion"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (0, 0, yaw)
    
    def shape_detection_callback(self, msg: String):
        """Handle shape detection messages from shape detector node"""
        if not self.shape_detected_flag:
            self.detected_shape_info = msg.data
            self.shape_detected_flag = True
            self.shape_stop_start = time.time()
            self.get_logger().info(f"🔺 Shape detected! Stopping for 2 seconds: {msg.data}")
    
    def publish_detection(self, shape_info):
        """Publish detection to /detection_status topic"""
        detection_key = shape_info
        
        if detection_key not in self.published_detections:
            msg = String()
            msg.data = shape_info
            self.detection_publisher.publish(msg)
            self.published_detections.add(detection_key)
            self.get_logger().info(f"📢 Published to /detection_status: {shape_info}")
    
    def check_if_dock_station(self):
        """Check if at dock station (P1: waypoint index 1)"""
        if self.active_goal == 1:
            target = self.target_points[1]
            x_diff = target[0] - self.current_pose[0]
            y_diff = target[1] - self.current_pose[1]
            distance = math.hypot(x_diff, y_diff)
            
            if distance < 0.25:
                return True
        return False
    
    def move_to_goal(self):
        """Original navigation logic with shape detection integration"""
        
        # Handle shape detection stop (2 seconds hold)
        if self.shape_detected_flag:
            velocity_msg = Twist()
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            self.velocity_publisher.publish(velocity_msg)
            
            elapsed = time.time() - self.shape_stop_start
            if elapsed >= 2.0:
                # Publish detection after 2 seconds
                self.publish_detection(self.detected_shape_info)
                self.shape_detected_flag = False
                self.detected_shape_info = None
                self.get_logger().info("✓ 2 seconds elapsed. Resuming navigation...")
            return
        
        # Check if all waypoints reached
        if self.active_goal >= len(self.target_points):
            stop_msg = Twist()
            self.velocity_publisher.publish(stop_msg)
            self.get_logger().info("🏁 All waypoints reached. Stopped at final position.")
            return
        
        # Original navigation logic
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
            self.get_logger().warn("⚠️  Obstacle detected ahead!")
        
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
                # Waypoint reached - check if it's the dock station
                if self.check_if_dock_station():
                    dock_msg = f"DOCK_STATION,{self.current_pose[0]:.2f},{self.current_pose[1]:.2f}"
                    self.publish_detection(dock_msg)
                    self.get_logger().info(f"🚉 DOCK_STATION reached at waypoint {self.active_goal + 1}")
                else:
                    self.get_logger().info(f"✓ Reached waypoint {self.active_goal + 1}")
                
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