import rclpy
import sys
import cv2
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, Pose, Point, Quaternion, Vector3
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_pose

# Camera intrinsics remain the same
cam_mat = np.array([[931.1829833984375, 0.0, 640.0],
                    [0.0, 931.1829833984375, 360.0],
                    [0.0, 0.0, 1.0]])

dist_mat = np.array([0.0, 0.0, 0.0, 0.0])
ARUCO_MARKER_SIZE = 0.15

def detect_aruco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    marker_poses = []

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, ARUCO_MARKER_SIZE, cam_mat, dist_mat)
        
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # Convert rotation vector to rotation matrix
            R_marker_to_cam, _ = cv2.Rodrigues(rvec)
            
            # Convert OpenCV camera coordinate system to ROS coordinate system
            # In OpenCV: X right, Y down, Z forward
            # In ROS: X forward, Y left, Z up
            R_opencv_to_ros = np.array([
                [0.0, 0.0, 1.0],  # X_ros = Z_cv
                [-1.0, 0.0, 0.0],  # Y_ros = -X_cv
                [0.0, -1.0, 0.0]    # Z_ros = -Y_cv
            ])

            # Additional rotation to align marker's front face with base_link frame
            R_additional = np.array([
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0,-1.0]
            ])

            # Transform position from camera to ROS coordinate system
            t_ros = R_opencv_to_ros @ tvec.T

            # Apply the transformations in order
            R_final = R_opencv_to_ros @ R_marker_to_cam @ R_additional
            
            # Convert to quaternion for ROS
            quat = R.from_matrix(R_final).as_quat()

            # Create pose
            marker_pose = Pose()
            marker_pose.position.x = float(t_ros[0])
            marker_pose.position.y = float(t_ros[1])
            marker_pose.position.z = float(t_ros[2])
            marker_pose.orientation.x = float(quat[0])
            marker_pose.orientation.y = float(quat[1])
            marker_pose.orientation.z = float(quat[2])
            marker_pose.orientation.w = float(quat[3])

            # Store pose
            marker_poses.append((ids[i][0], marker_pose))

            # Draw markers for visualization
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, 0.1)

    return marker_poses, image

class aruco_tf(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')
        self.color_cam_sub = self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10)
        
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.2, self.process_image)
        
        self.cv_image = None
        self.depth_image = None

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            
    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

    def process_image(self):
        if self.cv_image is None or self.depth_image is None:
            return

        marker_poses, annotated_image = detect_aruco(self.cv_image)

        if len(marker_poses) >= 1:
            self.publish_marker_transforms(marker_poses)

        cv2.imshow("Detected Aruco Markers", annotated_image)
        cv2.waitKey(1)

    def publish_marker_transforms(self, marker_poses):
        for marker_id, pose in marker_poses:
            try:
                # Get transform from base_link to camera_link
                base_to_camera = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())

                # Transform pose to base_link frame
                marker_pose_base = do_transform_pose(pose, base_to_camera)

                # Create transform message
                transform = TransformStamped()
                transform.header.stamp = self.get_clock().now().to_msg()
                transform.header.frame_id = 'base_link'
                transform.child_frame_id = f'obj_{marker_id}'
                
                # Set translation
                transform.transform.translation.x = marker_pose_base.position.x
                transform.transform.translation.y = marker_pose_base.position.y
                transform.transform.translation.z = marker_pose_base.position.z
                
                # Set rotation
                transform.transform.rotation = marker_pose_base.orientation

                # Publish transform
                self.br.sendTransform(transform)

                # Log position and orientation in base frame
                self.get_logger().info(
                    f'Marker {marker_id} position (base frame):\n'
                    f'  x={marker_pose_base.position.x:.3f}, y={marker_pose_base.position.y:.3f}, z={marker_pose_base.position.z:.3f}\n'
                    f'  qx={marker_pose_base.orientation.x:.3f}, qy={marker_pose_base.orientation.y:.3f}, '
                    f'qz={marker_pose_base.orientation.z:.3f}, qw={marker_pose_base.orientation.w:.3f}'
                )

            except TransformException as ex:
                self.get_logger().info(f"Could not transform: {ex}")
            except Exception as e:
                self.get_logger().info(f'Error publishing transform: {e}')

def main():
    rclpy.init(args=sys.argv)
    node = aruco_tf()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()