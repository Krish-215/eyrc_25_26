# Team ID:          3251
# Theme:            Krishi coBot
# Author List:      Hari Sathish S , Krishnaswamy S
# Filename:         shape_detector.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import math
import time

class ShapeDetector(Node):
    def __init__(self):
        super().__init__('shape_detector')
        
        # Publishers and Subscribers
        self.shape_pub = self.create_publisher(String, '/shape_detected', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Current robot pose
        self.current_pose = [0.0, 0.0, 0.0]
        
        # Detection parameters - optimized for small shapes
        self.min_detection_distance = 0.15
        self.max_detection_distance = 1.2
        self.min_cluster_points = 6
        
        # RANSAC parameters
        self.ransac_iterations = 60
        self.ransac_threshold = 0.03  # 3cm tolerance for small shapes
        self.min_line_length = 0.08   # Minimum 8cm line (small shapes)
        self.max_line_length = 0.35   # Maximum 35cm line (filter out tables)
        
        # Latest scan
        self.latest_scan = None
        
        # Detection tracking with cooldown
        self.published_shapes = set()
        self.last_publish_time = 0
        self.cooldown_duration = 15.0  # 15 seconds cooldown
        
        # Timer for processing
        self.create_timer(0.2, self.process_detection)
        
        self.get_logger().info("🔍 Shape Detector: Square only (immediate publish with 15s cooldown)")
    
    def scan_callback(self, msg: LaserScan):
        """Store latest scan data"""
        self.latest_scan = msg
    
    def odom_callback(self, msg: Odometry):
        """Update current robot pose"""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(ori)
        self.current_pose = [pos.x, pos.y, yaw]
    
    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def process_detection(self):
        """Main detection loop"""
        if self.latest_scan is None:
            return
        
        # Check cooldown - if less than 15 seconds since last publish, skip detection
        current_time = time.time()
        if current_time - self.last_publish_time < self.cooldown_duration:
            return
        
        # Extract point clusters from scan
        clusters = self.extract_clusters(self.latest_scan)
        
        if not clusters:
            return
        
        # Process each cluster
        for cluster in clusters:
            if len(cluster['points']) < self.min_cluster_points:
                continue
            
            # Filter out large clusters (likely tables/walls)
            cluster_size = self.calculate_cluster_size(cluster['points'])
            if cluster_size > 0.5:  # Ignore clusters larger than 50cm
                continue
            
            # Detect lines in this cluster using RANSAC
            lines = self.detect_lines_ransac(cluster['points'])
            
            if not lines:
                continue
            
            # Filter lines by length (only keep small shape lines)
            valid_lines = [l for l in lines if self.min_line_length <= l['length'] <= self.max_line_length]
            
            if not valid_lines:
                continue
            
            # Classify shape based on detected lines (square only)
            shape_info = self.classify_shape_from_lines(valid_lines, cluster)
            
            if shape_info and shape_info['type'] == 'square':
                # Use bot's current odometry position instead of shape position
                bot_x = self.current_pose[0]
                bot_y = self.current_pose[1]
                
                # Create location key based on bot position (within 0.1m precision)
                pos_key = f"square_{bot_x:.1f}_{bot_y:.1f}"
                
                # Check if already published this location
                if pos_key not in self.published_shapes:
                    # Publish bot's odometry position
                    msg = String()
                    msg.data = f"BAD_HEALTH,{bot_x:.2f},{bot_y:.2f}"
                    self.shape_pub.publish(msg)
                    
                    self.published_shapes.add(pos_key)
                    self.last_publish_time = time.time()
                    
                    self.get_logger().info(
                        f"✅ PUBLISHED: BAD_HEALTH at bot position ({bot_x:.2f}, {bot_y:.2f}) "
                        f"[{len(valid_lines)} lines detected] - 15s cooldown activated"
                    )
                    
                    # Exit after first detection to apply cooldown
                    return
    
    def calculate_cluster_size(self, points):
        """Calculate the maximum span of a cluster"""
        if len(points) < 2:
            return 0.0
        
        # Calculate bounding box diagonal
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        return np.sqrt(width**2 + height**2)
    
    def extract_clusters(self, scan: LaserScan):
        """Extract point clusters from LiDAR scan"""
        ranges = np.array(scan.ranges)
        angles = np.array([scan.angle_min + i * scan.angle_increment 
                          for i in range(len(ranges))])
        
        # Filter valid points
        valid_mask = np.isfinite(ranges) & \
                     (ranges > self.min_detection_distance) & \
                     (ranges < self.max_detection_distance) & \
                     (ranges > 0.0)
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return []
        
        # Convert to Cartesian coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        points = np.column_stack((x, y))
        
        # Cluster points based on proximity
        clusters = []
        current_cluster = {'points': [points[0]], 'indices': [0]}
        
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            
            # Smaller threshold for small shapes
            threshold = 0.06 + 0.01 * valid_ranges[i]
            
            if dist < threshold:
                current_cluster['points'].append(points[i])
                current_cluster['indices'].append(i)
            else:
                if len(current_cluster['points']) >= self.min_cluster_points:
                    current_cluster['points'] = np.array(current_cluster['points'])
                    clusters.append(current_cluster)
                
                current_cluster = {'points': [points[i]], 'indices': [i]}
        
        # Add last cluster
        if len(current_cluster['points']) >= self.min_cluster_points:
            current_cluster['points'] = np.array(current_cluster['points'])
            clusters.append(current_cluster)
        
        return clusters
    
    def detect_lines_ransac(self, points, max_lines=3):
        """Detect lines using RANSAC algorithm"""
        if len(points) < 3:
            return []
        
        lines = []
        remaining_points = points.copy()
        
        for _ in range(max_lines):
            if len(remaining_points) < 4:
                break
            
            best_line = None
            best_inliers = []
            best_inliers_indices = []
            
            # RANSAC iterations
            for iteration in range(self.ransac_iterations):
                if len(remaining_points) < 2:
                    break
                
                # Randomly sample 2 points
                sample_idx = np.random.choice(len(remaining_points), 2, replace=False)
                p1, p2 = remaining_points[sample_idx]
                
                # Calculate line parameters (ax + by + c = 0)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = np.sqrt(dx**2 + dy**2)
                
                if length < 0.04:  # Skip if points too close
                    continue
                
                # Normal vector (perpendicular to line)
                a = -dy / length
                b = dx / length
                c = -(a * p1[0] + b * p1[1])
                
                # Find inliers (points close to the line)
                distances = np.abs(a * remaining_points[:, 0] + 
                                  b * remaining_points[:, 1] + c)
                
                inlier_mask = distances < self.ransac_threshold
                inliers = remaining_points[inlier_mask]
                
                # Keep best fit
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_inliers_indices = np.where(inlier_mask)[0]
                    best_line = (a, b, c, p1, p2)
            
            # Accept line if enough inliers
            if len(best_inliers) >= 4:
                # Calculate actual line endpoints from inliers
                line_direction = np.array([best_line[1], -best_line[0]])
                line_direction = line_direction / np.linalg.norm(line_direction)
                
                # Project inliers onto line direction
                centroid = np.mean(best_inliers, axis=0)
                projections = (best_inliers - centroid).dot(line_direction)
                
                # Get endpoints
                min_proj = np.min(projections)
                max_proj = np.max(projections)
                endpoint1 = centroid + min_proj * line_direction
                endpoint2 = centroid + max_proj * line_direction
                
                line_length = np.linalg.norm(endpoint2 - endpoint1)
                
                # Store line (will filter by length later)
                lines.append({
                    'params': (best_line[0], best_line[1], best_line[2]),
                    'endpoints': (endpoint1, endpoint2),
                    'length': line_length,
                    'inliers': best_inliers,
                    'centroid': centroid
                })
                
                # Remove inliers from remaining points
                remaining_mask = np.ones(len(remaining_points), dtype=bool)
                remaining_mask[best_inliers_indices] = False
                remaining_points = remaining_points[remaining_mask]
            else:
                break
        
        return lines
    
    def classify_shape_from_lines(self, lines, cluster):
        """
        Classify shape based on detected lines
        SQUARE ONLY: 2 or 3 lines at ~90° angle
        """
        num_lines = len(lines)
        
        if num_lines < 2:
            return None
        
        # TWO LINES CASE (most common for horizontal shapes)
        if num_lines == 2:
            # Calculate angle between lines
            a1, b1, _ = lines[0]['params']
            a2, b2, _ = lines[1]['params']
            
            # Dot product of normals
            dot_product = abs(a1 * a2 + b1 * b2)
            angle = math.degrees(math.acos(np.clip(dot_product, 0, 1)))
            
            # Convert to interior angle
            if angle > 90:
                interior_angle = 180 - angle
            else:
                interior_angle = angle
            
            # Get position (midpoint between lines)
            mid1 = (lines[0]['endpoints'][0] + lines[0]['endpoints'][1]) / 2
            mid2 = (lines[1]['endpoints'][0] + lines[1]['endpoints'][1]) / 2
            position = (mid1 + mid2) / 2
            
            # Check line lengths are reasonable for small shapes
            len1, len2 = lines[0]['length'], lines[1]['length']
            avg_length = (len1 + len2) / 2
            
            if avg_length > self.max_line_length:
                return None
            
            # SQUARE: 90° corner (both inside and outside bends)
            if 75 < interior_angle < 105:
                return {
                    'type': 'square',
                    'position': position,
                    'confidence': 0.85
                }
        
        # THREE LINES CASE (square with 3 visible edges)
        elif num_lines == 3:
            # Calculate angles between consecutive lines
            angles = []
            for i in range(3):
                a1, b1, _ = lines[i]['params']
                a2, b2, _ = lines[(i+1) % 3]['params']
                
                dot = abs(a1 * a2 + b1 * b2)
                angle = math.degrees(math.acos(np.clip(dot, 0, 1)))
                
                # Interior angle
                if angle > 90:
                    angle = 180 - angle
                
                angles.append(angle)
            
            avg_angle = np.mean(angles)
            
            # Calculate center position
            all_points = np.vstack([line['inliers'] for line in lines])
            position = np.mean(all_points, axis=0)
            
            # Check if lines are appropriate size
            lengths = [line['length'] for line in lines]
            if any(l > self.max_line_length for l in lengths):
                return None
            
            # SQUARE: 3 edges at ~90° angles
            if 75 < avg_angle < 105:
                return {
                    'type': 'square',
                    'position': position,
                    'confidence': 0.85
                }
        
        return None
    
    def transform_to_world(self, point):
        """Transform from robot frame to world frame"""
        x_robot, y_robot = point
        x_world, y_world, yaw = self.current_pose
        
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        x_global = x_world + x_robot * cos_yaw - y_robot * sin_yaw
        y_global = y_world + x_robot * sin_yaw + y_robot * cos_yaw
        
        return [x_global, y_global]

def main(args=None):
    rclpy.init(args=args)
    detector = ShapeDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()