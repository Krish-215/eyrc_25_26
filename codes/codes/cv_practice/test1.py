#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script should be used to implement Task 1B of Krishi coBot (KC) Theme (eYRC 2025-26).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    task1b_boiler_plate.py
# Functions:
#			        [ depthimagecb, colorimagecb, bad_fruit_detection, process_image ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/color/image_raw, /camera/depth/image_rect_raw ]


import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from rclpy.duration import Duration
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import math

# runtime parameters
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False

# Detection parameters - TUNED FOR PERFORMANCE
MIN_FRUIT_AREA = 200
MAX_FRUIT_AREA = 30000

class FruitsTF(Node):
    """
    Optimized ROS2 node for fruit detection and TF publishing.
    """

    def __init__(self):
        super().__init__('fruits_tf')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # callback group handling
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions
        self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        # TF broadcaster and buffer
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer for periodic processing (slower rate to avoid overload)
        self.create_timer(0.3, self.process_image, callback_group=self.cb_group)

        # Fruit tracking
        self.fruit_id_counter = 0
        self.team_id = "TEAM_ID"  # Replace with your actual team ID
        self.published_fruits = set()
        
        # Processing flags
        self.is_processing = False

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info("Optimized FruitsTF node started.")

    # ---------------- Callbacks ----------------
    
    def depthimagecb(self, data):
        '''
        Description:    Callback function for aligned depth camera topic. 

        Args:
            data (Image): Input depth image frame received from aligned depth camera topic

        Returns:
            None
        '''
        try:
            # Convert ROS Image message to CV2 image (no heavy filtering in callback)
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {str(e)}')


    def colorimagecb(self, data):
        '''
        Description:    Callback function for colour camera raw topic.

        Args:
            data (Image): Input coloured raw image frame received from image_raw camera topic

        Returns:
            None
        '''
        try:
            # Convert ROS Image message to CV2 image
            raw_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            
            # Apply rotation/flipping if needed (uncomment as required)
            # raw_image = cv2.flip(raw_image, 0)  # Flip vertically
            # raw_image = cv2.flip(raw_image, 1)  # Flip horizontally
            # raw_image = cv2.rotate(raw_image, cv2.ROTATE_180)
            
            self.cv_image = raw_image
            
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {str(e)}')


    def bad_fruit_detection(self, rgb_image):
        '''
        Description:    Optimized function to detect bad fruits using HSV color filtering.

        Args:
            rgb_image (cv2 image): Input coloured raw image frame

        Returns:
            list: A list of detected bad fruit information dictionaries
        '''
        bad_fruits = []

        try:
            # Convert to HSV color space
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

            # Define HSV ranges for bad fruits (brown/dark spots)
            # Adjust these values based on your specific fruits
            lower_hsv1 = np.array([5, 50, 20], dtype=np.uint8)
            upper_hsv1 = np.array([25, 255, 120], dtype=np.uint8)
            
            lower_hsv2 = np.array([0, 0, 0], dtype=np.uint8)
            upper_hsv2 = np.array([180, 255, 50], dtype=np.uint8)

            # Create masks
            mask1 = cv2.inRange(hsv_image, lower_hsv1, upper_hsv1)
            mask2 = cv2.inRange(hsv_image, lower_hsv2, upper_hsv2)
            combined_mask = cv2.bitwise_or(mask1, mask2)

            # Simple morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area < MIN_FRUIT_AREA or area > MAX_FRUIT_AREA:
                    continue

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                cX = x + w // 2
                cY = y + h // 2

                # Validate center is within bounds
                if cX < 0 or cX >= rgb_image.shape[1] or cY < 0 or cY >= rgb_image.shape[0]:
                    continue

                # Get depth information
                distance = 0.0
                if self.depth_image is not None:
                    try:
                        depth_value = self.depth_image[cY, cX]
                        distance = float(depth_value) / 1000.0  # Convert mm to meters
                        
                        # Validate depth
                        if distance < 0.1 or distance > 5.0:
                            distance = 0.0
                            
                    except Exception as e:
                        self.get_logger().debug(f'Error reading depth: {str(e)}')

                # Store fruit info
                fruit_info = {
                    'center': (cX, cY),
                    'distance': distance,
                    'width': w,
                    'height': h,
                    'area': area,
                    'id': self.fruit_id_counter,
                    'contour': contour
                }
                
                bad_fruits.append(fruit_info)
                self.fruit_id_counter += 1

        except Exception as e:
            self.get_logger().error(f'Error in detection: {str(e)}')

        return bad_fruits


    def process_image(self):
        '''
        Description:    Optimized timer-driven loop for image processing.

        Returns:
            None
        '''
        # Prevent re-entry
        if self.is_processing:
            return
            
        self.is_processing = True

        try:
            # Camera parameters
            sizeCamX = 1280
            sizeCamY = 720
            centerCamX = 642.724365234375
            centerCamY = 361.9780578613281
            focalX = 915.3003540039062
            focalY = 914.0320434570312

            # Check if images are available
            if self.cv_image is None:
                self.is_processing = False
                return

            # Get detected bad fruits
            bad_fruits = self.bad_fruit_detection(self.cv_image)

            # Create display image
            display_image = self.cv_image.copy()

            # Process each detected fruit
            for fruit in bad_fruits:
                cX, cY = fruit['center']
                distance_from_rgb = fruit['distance']
                fruit_id = fruit['id']
                width = fruit['width']

                # Skip if distance is invalid
                if distance_from_rgb <= 0.1:
                    continue

                # Calculate 3D coordinates
                x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
                y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
                z = distance_from_rgb

                # Draw on display image
                cv2.circle(display_image, (cX, cY), 5, (0, 255, 0), -1)
                cv2.drawContours(display_image, [fruit['contour']], -1, (0, 255, 0), 2)
                cv2.putText(display_image, f"ID:{fruit_id}", (cX + 10, cY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Publish TF from camera_link to cam_<fruit_id>
                try:
                    t_cam = TransformStamped()
                    t_cam.header.stamp = self.get_clock().now().to_msg()
                    t_cam.header.frame_id = 'camera_link'
                    t_cam.child_frame_id = f'cam_{fruit_id}'
                    
                    t_cam.transform.translation.x = float(x)
                    t_cam.transform.translation.y = float(y)
                    t_cam.transform.translation.z = float(z)
                    t_cam.transform.rotation.x = 0.0
                    t_cam.transform.rotation.y = 0.0
                    t_cam.transform.rotation.z = 0.0
                    t_cam.transform.rotation.w = 1.0

                    self.tf_broadcaster.sendTransform(t_cam)

                    # Try to lookup and publish base_link TF (non-blocking)
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            'base_link',
                            f'cam_{fruit_id}',
                            rclpy.time.Time(),
                            timeout=Duration(seconds=0.01)  # Very short timeout
                        )

                        t_base = TransformStamped()
                        t_base.header.stamp = self.get_clock().now().to_msg()
                        t_base.header.frame_id = 'base_link'
                        t_base.child_frame_id = f'{self.team_id}_bad_fruit_{fruit_id}'
                        
                        t_base.transform.translation.x = transform.transform.translation.x
                        t_base.transform.translation.y = transform.transform.translation.y
                        t_base.transform.translation.z = transform.transform.translation.z
                        t_base.transform.rotation = transform.transform.rotation

                        self.tf_broadcaster.sendTransform(t_base)

                        # Log only once per fruit
                        if fruit_id not in self.published_fruits:
                            self.get_logger().info(
                                f'Detected bad_fruit_{fruit_id} at ({x:.2f}, {y:.2f}, {z:.2f})m'
                            )
                            self.published_fruits.add(fruit_id)

                    except (LookupException, ConnectivityException, ExtrapolationException):
                        # TF not ready yet, skip this iteration
                        pass
                        
                except Exception as e:
                    self.get_logger().error(f'Error publishing TF: {str(e)}')

            # Add simple stats overlay
            cv2.putText(display_image, f"Detected: {len(bad_fruits)} fruits", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show image
            if SHOW_IMAGE:
                cv2.imshow('fruits_tf_view', display_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in process_image: {str(e)}')
        finally:
            self.is_processing = False


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