#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  ArUco Perception Node
*  - Detects ArUco markers (ID 3 and 6)
*  - Publishes TF frames for detected markers
*  - Provides service to get recorded marker poses
*
*****************************************************************************************
'''

import rclpy
import cv2
import numpy as np
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_pose
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseArray


class ArucoPerception(Node):
    '''
    ArUco marker detection and TF publishing node
    '''
    
    def __init__(self):
        super().__init__('aruco_perception')
        
        ############ SUBSCRIBERS ############
        self.color_cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self.color_image_cb, 10
        )
        
        ############ PUBLISHERS ############
        self.marker_poses_pub = self.create_publisher(PoseArray, '/detected_markers', 10)
        
        ############ SERVICES ############
        self.get_poses_service = self.create_service(
            Trigger, '/get_marker_poses', self.get_marker_poses_callback
        )
        
        ############ TF SETUP ############
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        
        ############ IMAGE PROCESSING ############
        self.bridge = CvBridge()
        self.cv_image = None
        
        ############ CAMERA PARAMETERS ############
        self.cam_mat = np.array([[915.3003540039062, 0.0, 642.724365234375], 
                                 [0.0, 914.0320434570312, 361.9780578613281], 
                                 [0.0, 0.0, 1.0]])
        self.dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.size_of_aruco_m = 0.15
        
        ############ ARUCO OFFSETS ############
        self.obj_3_offset_x = -0.15
        self.obj_3_offset_y = 0.08
        self.obj_3_offset_z = 0.06
        
        self.obj_6_offset_x = -0.27
        self.obj_6_offset_y = 0.0
        self.obj_6_offset_z = 0.15
        
        ############ MARKER STORAGE ############
        self.recorded_poses = {}
        self.markers_detected = set()
        
        ############ TIMERS ############
        self.aruco_timer = self.create_timer(0.1, self.process_aruco)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("👁️  ARUCO PERCEPTION NODE INITIALIZED")
        self.get_logger().info("=" * 60)
        self.get_logger().info("📷 Detecting markers ID 3 and 6...")
        self.get_logger().info("🔧 Service available: /get_marker_poses")
        self.get_logger().info("=" * 60)
    
    
    def color_image_cb(self, data):
        '''Callback for color camera'''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
    
    
    def process_aruco(self):
        '''Process ArUco markers and publish TF frames'''
        if self.cv_image is None:
            return
        
        image_copy = self.cv_image.copy()
        
        # Detect ArUco markers
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_copy, corners, marker_ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.size_of_aruco_m, self.cam_mat, self.dist_mat
            )
            
            for i, marker_id in enumerate(marker_ids):
                marker_id = marker_id[0]
                
                if marker_id not in [3, 6]:
                    continue
                
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                
                cv2.drawFrameAxes(image_copy, self.cam_mat, self.dist_mat, rvec, tvec, 0.1)
                
                corner_pts = corners[i][0]
                cX = int(np.mean(corner_pts[:, 0]))
                cY = int(np.mean(corner_pts[:, 1]))
                cv2.circle(image_copy, (cX, cY), 5, (0, 255, 0), -1)
                cv2.putText(image_copy, f"ID {marker_id}", (cX - 20, cY - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Convert to ROS coordinate system
                R_marker_to_cam, _ = cv2.Rodrigues(rvec)
                
                R_opencv_to_ros = np.array([
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0]
                ])
                
                R_additional = np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0]
                ])
                
                t_ros = R_opencv_to_ros @ tvec
                R_final = R_opencv_to_ros @ R_marker_to_cam @ R_additional
                quat = R.from_matrix(R_final).as_quat()
                
                marker_pose_cam = Pose()
                marker_pose_cam.position.x = float(t_ros[0])
                marker_pose_cam.position.y = float(t_ros[1])
                marker_pose_cam.position.z = float(t_ros[2])
                marker_pose_cam.orientation.x = float(quat[0])
                marker_pose_cam.orientation.y = float(quat[1])
                marker_pose_cam.orientation.z = float(quat[2])
                marker_pose_cam.orientation.w = float(quat[3])
                
                try:
                    base_to_camera = self.tf_buffer.lookup_transform(
                        'base_link', 'camera_link', rclpy.time.Time()
                    )
                    marker_pose_base = do_transform_pose(marker_pose_cam, base_to_camera)
                    
                    # Get offsets
                    if marker_id == 3:
                        obj_offset_x = self.obj_3_offset_x
                        obj_offset_y = self.obj_3_offset_y
                        obj_offset_z = self.obj_3_offset_z
                    else:
                        obj_offset_x = self.obj_6_offset_x
                        obj_offset_y = self.obj_6_offset_y
                        obj_offset_z = self.obj_6_offset_z
                    
                    # Publish TF
                    t_base = TransformStamped()
                    t_base.header.stamp = self.get_clock().now().to_msg()
                    t_base.header.frame_id = 'base_link'
                    t_base.child_frame_id = f'obj_{marker_id}'
                    
                    t_base.transform.translation.x = marker_pose_base.position.x + obj_offset_x
                    t_base.transform.translation.y = marker_pose_base.position.y + obj_offset_y
                    t_base.transform.translation.z = marker_pose_base.position.z + obj_offset_z
                    t_base.transform.rotation = marker_pose_base.orientation
                    
                    self.br.sendTransform(t_base)
                    
                    # Record pose
                    if marker_id not in self.markers_detected:
                        self.markers_detected.add(marker_id)
                        self.get_logger().info(f"   ✓ Marker ID {marker_id} detected")
                    
                    self.recorded_poses[marker_id] = {
                        'position': (
                            t_base.transform.translation.x,
                            t_base.transform.translation.y,
                            t_base.transform.translation.z
                        ),
                        'orientation': (
                            t_base.transform.rotation.x,
                            t_base.transform.rotation.y,
                            t_base.transform.rotation.z,
                            t_base.transform.rotation.w
                        )
                    }
                    
                except TransformException as ex:
                    pass
        
        # Display info
        status_text = f"Markers detected: "
        if 3 in self.markers_detected:
            status_text += "[3] "
        if 6 in self.markers_detected:
            status_text += "[6]"
        cv2.putText(image_copy, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('ArUco Perception', image_copy)
        cv2.waitKey(1)
    
    
    def get_marker_poses_callback(self, request, response):
        '''Service to get recorded marker poses'''
        if 3 in self.recorded_poses and 6 in self.recorded_poses:
            response.success = True
            response.message = f"Markers detected: ID 3 at {self.recorded_poses[3]['position']}, ID 6 at {self.recorded_poses[6]['position']}"
            self.get_logger().info("✅ Marker poses requested - both markers available")
        else:
            response.success = False
            detected = list(self.recorded_poses.keys())
            response.message = f"Not all markers detected. Available: {detected}"
            self.get_logger().warn(f"⚠️ Marker poses requested but missing markers. Available: {detected}")
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPerception()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()