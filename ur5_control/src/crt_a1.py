#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script implements Task 1B of Krishi coBot (KC) Theme (eYRC 2025-26).
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    task1b_complete.py
# Functions:        calculate_rectangle_area, detect_aruco
# Nodes:		    aruco_tf_publisher
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/color/image_raw, /camera/aligned_depth_to_color/image_raw ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, Pose
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_pose


##################### FUNCTION DEFINITIONS #######################

def calculate_rectangle_area(coordinates):
    '''
    Description:    Function to calculate area or detected aruco

    Args:
        coordinates (list):     coordinates of detected aruco (4 set of (x,y) coordinates)

    Returns:
        area        (float):    area of detected aruco
        width       (float):    width of detected aruco
    '''

    # Extract the 4 corner points
    pts = coordinates[0]
    
    # Calculate width (distance between first two points)
    width = np.linalg.norm(pts[0] - pts[1])
    
    # Calculate height (distance between second and third points)
    height = np.linalg.norm(pts[1] - pts[2])
    
    # Calculate area
    area = width * height

    return area, width


def detect_aruco(image):
    '''
    Description:    Function to perform aruco detection and return each detail of aruco detected 
                    such as marker ID, distance, angle, width, center point location, etc.

    Args:
        image                   (Image):    Input image frame received from respective camera topic

    Returns:
        center_aruco_list       (list):     Center points of all aruco markers detected
        distance_from_rgb_list  (list):     Distance value of each aruco markers detected from RGB camera
        angle_aruco_list        (list):     Angle of all pose estimated for aruco marker
        width_aruco_list        (list):     Width of all detected aruco markers
        ids                     (list):     List of all aruco marker IDs detected in a single frame 
    '''

    ############ Function VARIABLES ############

    aruco_area_threshold = 1500
    
    # Camera matrix from camera info
    cam_mat = np.array([[915.3003540039062, 0.0, 642.724365234375], 
                        [0.0, 914.0320434570312, 361.9780578613281], 
                        [0.0, 0.0, 1.0]])
    
    dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    size_of_aruco_m = 0.15  # 150mm = 0.15m
    
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids = []
 
    ############ CODE IMPLEMENTATION ############

    # Convert BGR image to GRAYSCALE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define ArUco dictionary (4x4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    
    # Detect ArUco markers
    corners, marker_ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Check if any markers were detected
    if marker_ids is not None and len(marker_ids) > 0:
        # Draw detected markers on the image
        cv2.aruco.drawDetectedMarkers(image, corners, marker_ids)
        
        # Estimate pose for all markers
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, size_of_aruco_m, cam_mat, dist_mat
        )
        
        # Loop through each detected marker
        for i, marker_id in enumerate(marker_ids):
            # Calculate area and width
            area, width = calculate_rectangle_area(corners[i])
            
            # Filter based on area threshold
            if area >= aruco_area_threshold:
                # Calculate center point
                corner_pts = corners[i][0]
                cX = int(np.mean(corner_pts[:, 0]))
                cY = int(np.mean(corner_pts[:, 1]))
                center_aruco_list.append([cX, cY])
                
                # Get translation vector (distance)
                tvec = tvecs[i][0]
                distance = np.linalg.norm(tvec)  # Euclidean distance
                distance_from_rgb_list.append(distance)
                
                # Get rotation vector
                rvec = rvecs[i][0]
                
                # Convert rotation vector to rotation matrix
                rot_mat, _ = cv2.Rodrigues(rvec)
                
                # Extract yaw angle (rotation around z-axis)
                sy = math.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
                singular = sy < 1e-6
                
                if not singular:
                    yaw = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
                else:
                    yaw = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
                
                # Convert to degrees
                angle = math.degrees(yaw)
                angle_aruco_list.append(angle)
                
                # Store width
                width_aruco_list.append(width)
                
                # Store marker ID
                ids.append(marker_id[0])
                
                # Draw frame axes on the image
                cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, 0.1)

    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids


##################### CLASS DEFINITION #######################

class aruco_tf(Node):
    '''
    ___CLASS___

    Description:    Class which servers purpose to define process for detecting aruco marker and publishing tf on pose estimated.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class aruco_tf
        '''

        super().__init__('aruco_tf_publisher')
        
        ############ Topic SUBSCRIPTIONS ############
        
        self.color_cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self.colorimagecb, 10
        )
        self.depth_cam_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depthimagecb, 10
        )

        ############ Constructor VARIABLES/OBJECTS ############
        
        image_processing_rate = 0.2  # Process every 0.2 seconds
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(image_processing_rate, self.process_image)
        
        self.cv_image = None
        self.depth_image = None
        
        ############ OFFSET ADJUSTMENTS ############
        # Independent offset adjustments for marker ID 3 and 6
        
        # ===== MARKER ID 3 OFFSETS =====
        # cam_3 frame offsets (relative to camera_link)
        self.cam_3_offset_x = 0.0
        self.cam_3_offset_y = 0.0
        self.cam_3_offset_z = 0.0
        self.cam_3_rot_offset_roll = 0.0
        self.cam_3_rot_offset_pitch = 0.0
        self.cam_3_rot_offset_yaw = 0.0
        
        # obj_3 frame offsets (relative to base_link)
        self.obj_3_offset_x = -0.1
        self.obj_3_offset_y = 0.11
        self.obj_3_offset_z = 0.075
        self.obj_3_rot_offset_roll = 0.0
        self.obj_3_rot_offset_pitch = 0.0
        self.obj_3_rot_offset_yaw = 0.0
        
        # ===== MARKER ID 6 OFFSETS =====
        # cam_6 frame offsets (relative to camera_link)
        self.cam_6_offset_x = 0.0
        self.cam_6_offset_y = 0.0
        self.cam_6_offset_z = 0.0
        self.cam_6_rot_offset_roll = 0.0
        self.cam_6_rot_offset_pitch = 0.0
        self.cam_6_rot_offset_yaw = 0.0
        
        # obj_6 frame offsets (relative to base_link)
        self.obj_6_offset_x = -0.27
        self.obj_6_offset_y = 0.0
        self.obj_6_offset_z = 0.15
        self.obj_6_rot_offset_roll = 0.0
        self.obj_6_rot_offset_pitch = 0.0
        self.obj_6_rot_offset_yaw = 0.0


    def depthimagecb(self, data):
        '''
        Description:    Callback function for aligned depth camera topic.

        Args:
            data (Image):    Input depth image frame received from aligned depth camera topic
        '''
        
        try:
            # Convert ROS Image message to CV2 Image
            # 16UC1 for millimeter depth values or 32FC1 for meter depth values
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='16UC1')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error in depth callback: {e}')


    def colorimagecb(self, data):
        '''
        Description:    Callback function for colour camera raw topic.

        Args:
            data (Image):    Input coloured raw image frame received from image_raw camera topic
        '''
        
        try:
            # Convert ROS Image message to CV2 Image (BGR8)
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error in color callback: {e}')


    def process_image(self):
        '''
        Description:    Timer function used to detect aruco markers and publish tf on estimated poses.
        '''
        
        # Check if images are available
        if self.cv_image is None:
            return
        
        ############ Function VARIABLES ############
        
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375
        
        # Camera matrix
        cam_mat = np.array([[915.3003540039062, 0.0, 642.724365234375], 
                            [0.0, 914.0320434570312, 361.9780578613281], 
                            [0.0, 0.0, 1.0]])
        
        dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        size_of_aruco_m = 0.15
        
        ############ PROCESSING ############
        
        # Create a copy of the image for drawing
        image_copy = self.cv_image.copy()
        
        # Detect ArUco markers using OpenCV
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if marker_ids is not None and len(marker_ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(image_copy, corners, marker_ids)
            
            # Estimate pose for all markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, size_of_aruco_m, cam_mat, dist_mat
            )
            
            # Process each detected marker
            for i, marker_id in enumerate(marker_ids):
                marker_id = marker_id[0]
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                
                # Draw frame axes for visualization
                cv2.drawFrameAxes(image_copy, cam_mat, dist_mat, rvec, tvec, 0.1)
                
                # Calculate center point for visualization
                corner_pts = corners[i][0]
                cX = int(np.mean(corner_pts[:, 0]))
                cY = int(np.mean(corner_pts[:, 1]))
                cv2.circle(image_copy, (cX, cY), 5, (0, 255, 0), -1)
                
                # Convert rotation vector to rotation matrix
                R_marker_to_cam, _ = cv2.Rodrigues(rvec)
                
                # Coordinate system transformation
                # OpenCV: X right, Y down, Z forward
                # ROS: X forward, Y left, Z up
                R_opencv_to_ros = np.array([
                    [0.0, 0.0, 1.0],   # X_ros = Z_opencv
                    [-1.0, 0.0, 0.0],  # Y_ros = -X_opencv
                    [0.0, -1.0, 0.0]   # Z_ros = -Y_opencv
                ])
                
                # Additional rotation to align marker's Z-axis to point into the box
                R_additional = np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0]
                ])
                
                # Transform position from camera to ROS coordinate system
                t_ros = R_opencv_to_ros @ tvec
                
                # Apply rotations: OpenCV -> ROS -> Box alignment
                R_final = R_opencv_to_ros @ R_marker_to_cam @ R_additional
                
                # Convert rotation matrix to quaternion
                quat = R.from_matrix(R_final).as_quat()
                
                # Create pose in camera frame
                marker_pose_cam = Pose()
                marker_pose_cam.position.x = float(t_ros[0])
                marker_pose_cam.position.y = float(t_ros[1])
                marker_pose_cam.position.z = float(t_ros[2])
                marker_pose_cam.orientation.x = float(quat[0])
                marker_pose_cam.orientation.y = float(quat[1])
                marker_pose_cam.orientation.z = float(quat[2])
                marker_pose_cam.orientation.w = float(quat[3])
                
                try:
                    # Lookup transform from base_link to camera_link
                    base_to_camera = self.tf_buffer.lookup_transform(
                        'base_link', 
                        'camera_link', 
                        rclpy.time.Time()
                    )
                    
                    # Transform pose from camera frame to base_link frame
                    marker_pose_base = do_transform_pose(marker_pose_cam, base_to_camera)
                    
                    # Get marker-specific offsets
                    if marker_id == 3:
                        cam_offset_x = self.cam_3_offset_x
                        cam_offset_y = self.cam_3_offset_y
                        cam_offset_z = self.cam_3_offset_z
                        cam_rot_roll = self.cam_3_rot_offset_roll
                        cam_rot_pitch = self.cam_3_rot_offset_pitch
                        cam_rot_yaw = self.cam_3_rot_offset_yaw
                        
                        obj_offset_x = self.obj_3_offset_x
                        obj_offset_y = self.obj_3_offset_y
                        obj_offset_z = self.obj_3_offset_z
                        obj_rot_roll = self.obj_3_rot_offset_roll
                        obj_rot_pitch = self.obj_3_rot_offset_pitch
                        obj_rot_yaw = self.obj_3_rot_offset_yaw
                    elif marker_id == 6:
                        cam_offset_x = self.cam_6_offset_x
                        cam_offset_y = self.cam_6_offset_y
                        cam_offset_z = self.cam_6_offset_z
                        cam_rot_roll = self.cam_6_rot_offset_roll
                        cam_rot_pitch = self.cam_6_rot_offset_pitch
                        cam_rot_yaw = self.cam_6_rot_offset_yaw
                        
                        obj_offset_x = self.obj_6_offset_x
                        obj_offset_y = self.obj_6_offset_y
                        obj_offset_z = self.obj_6_offset_z
                        obj_rot_roll = self.obj_6_rot_offset_roll
                        obj_rot_pitch = self.obj_6_rot_offset_pitch
                        obj_rot_yaw = self.obj_6_rot_offset_yaw
                    else:
                        # For any other marker IDs, use zero offsets
                        cam_offset_x = cam_offset_y = cam_offset_z = 0.0
                        cam_rot_roll = cam_rot_pitch = cam_rot_yaw = 0.0
                        obj_offset_x = obj_offset_y = obj_offset_z = 0.0
                        obj_rot_roll = obj_rot_pitch = obj_rot_yaw = 0.0
                    
                    # Publish TF: base_link -> obj_<marker_id> with adjustable offset
                    t_base = TransformStamped()
                    t_base.header.stamp = self.get_clock().now().to_msg()
                    t_base.header.frame_id = 'base_link'
                    t_base.child_frame_id = f'obj_{marker_id}'
                    
                    # Apply position offset for obj frame
                    t_base.transform.translation.x = marker_pose_base.position.x + obj_offset_x
                    t_base.transform.translation.y = marker_pose_base.position.y + obj_offset_y
                    t_base.transform.translation.z = marker_pose_base.position.z + obj_offset_z
                    
                    # Apply rotation offset for obj frame
                    if obj_rot_roll != 0.0 or obj_rot_pitch != 0.0 or obj_rot_yaw != 0.0:
                        # Get current quaternion
                        current_quat = [marker_pose_base.orientation.x, 
                                       marker_pose_base.orientation.y,
                                       marker_pose_base.orientation.z,
                                       marker_pose_base.orientation.w]
                        current_rot = R.from_quat(current_quat)
                        
                        # Create offset rotation
                        offset_rot = R.from_euler('xyz', [obj_rot_roll, obj_rot_pitch, obj_rot_yaw])
                        
                        # Apply offset rotation
                        final_rot = current_rot * offset_rot
                        final_quat = final_rot.as_quat()
                        
                        t_base.transform.rotation.x = final_quat[0]
                        t_base.transform.rotation.y = final_quat[1]
                        t_base.transform.rotation.z = final_quat[2]
                        t_base.transform.rotation.w = final_quat[3]
                    else:
                        t_base.transform.rotation = marker_pose_base.orientation
                    
                    self.br.sendTransform(t_base)
                    
                    # Optional: Also publish cam_<marker_id> frame with adjustable offset
                    t_camera = TransformStamped()
                    t_camera.header.stamp = self.get_clock().now().to_msg()
                    t_camera.header.frame_id = 'camera_link'
                    t_camera.child_frame_id = f'cam_{marker_id}'
                    
                    # Apply position offset for cam frame
                    t_camera.transform.translation.x = marker_pose_cam.position.x + cam_offset_x
                    t_camera.transform.translation.y = marker_pose_cam.position.y + cam_offset_y
                    t_camera.transform.translation.z = marker_pose_cam.position.z + cam_offset_z
                    
                    # Apply rotation offset for cam frame
                    if cam_rot_roll != 0.0 or cam_rot_pitch != 0.0 or cam_rot_yaw != 0.0:
                        # Get current quaternion
                        current_quat = [marker_pose_cam.orientation.x, 
                                       marker_pose_cam.orientation.y,
                                       marker_pose_cam.orientation.z,
                                       marker_pose_cam.orientation.w]
                        current_rot = R.from_quat(current_quat)
                        
                        # Create offset rotation
                        offset_rot = R.from_euler('xyz', [cam_rot_roll, cam_rot_pitch, cam_rot_yaw])
                        
                        # Apply offset rotation
                        final_rot = current_rot * offset_rot
                        final_quat = final_rot.as_quat()
                        
                        t_camera.transform.rotation.x = final_quat[0]
                        t_camera.transform.rotation.y = final_quat[1]
                        t_camera.transform.rotation.z = final_quat[2]
                        t_camera.transform.rotation.w = final_quat[3]
                    else:
                        t_camera.transform.rotation = marker_pose_cam.orientation
                    
                    self.br.sendTransform(t_camera)
                    
                except TransformException as ex:
                    self.get_logger().warn(f'Could not transform for marker {marker_id}: {ex}')
                except Exception as e:
                    self.get_logger().error(f'Error publishing transform for marker {marker_id}: {e}')
        
        # Display image with detected markers
        cv2.imshow('ArUco Detection', image_copy)
        cv2.waitKey(1)


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the aruco_tf class to perform it's task
    '''

    rclpy.init(args=sys.argv)
    
    node = rclpy.create_node('aruco_tf_process')
    
    node.get_logger().info('Node created: Aruco tf process')
    
    aruco_tf_class = aruco_tf()
    
    rclpy.spin(aruco_tf_class)
    
    aruco_tf_class.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()