#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script implements Task 1B - Bad Fruit Detection on Conveyor Belt
*
*****************************************************************************************
'''

# Team ID:          [ Your-Team-ID ]
# Author List:		[ Your Names ]
# Filename:		    task1b.py
# Functions:
#			        colorimagecb, depthimagecb, bad_fruit_detection, process_image
# Nodes:		    
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/color/image_raw, /camera/depth/image_rect_raw ]

import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import cv2
import numpy as np
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# Runtime parameters
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False

class FruitsTF(Node):
    """
    ROS2 Node for detecting bad fruits and publishing their TF transforms.
    """

    def __init__(self):
        super().__init__('fruits_tf')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        
        # TF2 setup
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Callback group handling
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions
        self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.colorimagecb, 
            10, 
            callback_group=self.cb_group
        )
        self.create_subscription(
            Image, 
            '/camera/depth/image_rect_raw', 
            self.depthimagecb, 
            10, 
            callback_group=self.cb_group
        )

        # Timer for periodic processing
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info("FruitsTF node started - detecting bad fruits...")
        
        # Team ID - CHANGE THIS TO YOUR TEAM ID
        self.team_id = 5  # Update with your actual team ID

    def depthimagecb(self, data):
        '''
        Callback function for depth camera topic.
        Converts ROS Image message to CV2 image format.
        '''
        try:
            # Convert ROS Image to CV2 format (depth is typically 16-bit or 32-bit)
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {str(e)}')

    def colorimagecb(self, data):
        '''
        Callback function for color camera topic.
        Converts ROS Image message to CV2 image format.
        '''
        try:
            # Convert ROS Image to CV2 format (BGR8)
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            
            # Check if image needs rotation or flipping
            # Uncomment if needed based on your camera orientation
            # self.cv_image = cv2.flip(self.cv_image, 0)  # Flip vertically
            # self.cv_image = cv2.flip(self.cv_image, 1)  # Flip horizontally
            # self.cv_image = cv2.rotate(self.cv_image, cv2.ROTATE_180)
            
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {str(e)}')

    def bad_fruit_detection(self, rgb_image):
        '''
        Detects bad (spoiled/greyish-white) fruits in the image.
        
        Args:
            rgb_image: Input BGR image from camera
            
        Returns:
            list: List of dictionaries containing detected fruit information
        '''
        bad_fruits = []
        
        if rgb_image is None:
            return bad_fruits

        # Step 1: Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Step 2: Define HSV range for greyish-white bad fruits
        # Greyish-white fruits have low saturation and medium-high value
        lower_bad = np.array([0, 0, 150])      # Low saturation, high brightness
        upper_bad = np.array([180, 60, 255])   # Any hue, low saturation
        
        # Step 3: Create binary mask
        mask = cv2.inRange(hsv, lower_bad, upper_bad)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Step 4: Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 5: Process each contour
        fruit_id = 1
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust threshold based on your setup)
            if area > 500:  # Minimum area threshold
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                cX = x + w // 2
                cY = y + h // 2
                
                # Get depth at center point (distance from camera)
                distance = 0.0
                if self.depth_image is not None and 0 <= cY < self.depth_image.shape[0] and 0 <= cX < self.depth_image.shape[1]:
                    # Depth is usually in mm, convert to meters
                    depth_value = self.depth_image[cY, cX]
                    if isinstance(depth_value, np.ndarray):
                        depth_value = depth_value[0]
                    distance = float(depth_value) / 1000.0  # Convert mm to meters
                    
                    # Handle invalid depth readings
                    if distance == 0.0 or np.isnan(distance) or np.isinf(distance):
                        # Try averaging nearby depth values
                        region = self.depth_image[max(0, cY-5):min(self.depth_image.shape[0], cY+5),
                                                 max(0, cX-5):min(self.depth_image.shape[1], cX+5)]
                        valid_depths = region[region > 0]
                        if len(valid_depths) > 0:
                            distance = float(np.median(valid_depths)) / 1000.0
                
                # Calculate angle (orientation of bounding box)
                angle = 0.0
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]
                
                # Store fruit information
                fruit_info = {
                    'center': (cX, cY),
                    'distance': distance,
                    'angle': angle,
                    'width': w,
                    'id': fruit_id,
                    'contour': contour
                }
                bad_fruits.append(fruit_info)
                fruit_id += 1
        
        return bad_fruits

    def process_image(self):
        '''
        Timer-driven loop for periodic image processing and TF publishing.
        '''
        if self.cv_image is None:
            return
        
        # Camera intrinsic parameters
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312
        
        # Detect bad fruits
        bad_fruits = self.bad_fruit_detection(self.cv_image)
        
        # Create visualization image
        display_image = self.cv_image.copy()
        
        # Process each detected fruit
        for fruit in bad_fruits:
            cX, cY = fruit['center']
            distance = fruit['distance']
            fruit_id = fruit['id']
            contour = fruit['contour']
            
            # Skip if invalid distance
            if distance <= 0.0 or distance > 5.0:  # Reasonable range check
                continue
            
            # Calculate 3D position using camera intrinsics
            x = distance * (sizeCamX - cX - centerCamX) / focalX
            y = distance * (sizeCamY - cY - centerCamY) / focalY
            z = distance
            
            # Draw contour and center on visualization
            cv2.drawContours(display_image, [contour], -1, (0, 255, 0), 2)
            cv2.circle(display_image, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(display_image, f'ID:{fruit_id}', (cX-20, cY-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Publish transform from camera_link to intermediate frame
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = 'camera_link'
            t_cam.child_frame_id = f'cam_{fruit_id}'
            
            t_cam.transform.translation.x = x
            t_cam.transform.translation.y = y
            t_cam.transform.translation.z = z
            t_cam.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(t_cam)
            
            # Lookup transform from base_link to cam frame
            try:
                # Wait a bit for transform to be available
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    f'cam_{fruit_id}',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                # Publish final transform from base_link to fruit frame
                t_fruit = TransformStamped()
                t_fruit.header.stamp = self.get_clock().now().to_msg()
                t_fruit.header.frame_id = 'base_link'
                t_fruit.child_frame_id = f'{self.team_id}_bad_fruit_{fruit_id}'
                
                t_fruit.transform.translation.x = transform.transform.translation.x
                t_fruit.transform.translation.y = transform.transform.translation.y
                t_fruit.transform.translation.z = transform.transform.translation.z
                t_fruit.transform.rotation = transform.transform.rotation
                
                self.tf_broadcaster.sendTransform(t_fruit)
                
                self.get_logger().info(
                    f'Bad fruit {fruit_id} detected at ({x:.3f}, {y:.3f}, {z:.3f})'
                )
                
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f'TF lookup failed for fruit {fruit_id}: {str(e)}')
        
        # Display the image
        if SHOW_IMAGE:
            cv2.imshow('fruits_tf_view', display_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
