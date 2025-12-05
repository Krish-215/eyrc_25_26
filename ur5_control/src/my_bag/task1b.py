#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
# Team ID:          3251
# Theme:            Krishi coBot
# Author List:      Krishnaswamy, Hari Sathish
# Filename:         task_1b.py
# Functions:        __init__, depthimagecb, colorimagecb, bad_fruit_detection,
#                   get_depth_at_point, process_image, main
# Global variables: TEAM_ID, SHOW_IMAGE, DISABLE_MULTITHREADING, DEPTH_MEDIAN_KERNEL,
#                   MIN_CONTOUR_AREA, MAX_CONTOUR_AREA, MAX_DEPTH, MIN_DEPTH,
#                   TOP_REGION_HEIGHT_RATIO, TOP_REGION_EXTEND_UP, MIN_GREEN_PIXELS,
#                   sizeCamX, sizeCamY, centerCamX, centerCamY, focalX, focalY
'''

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf_transformations import quaternion_matrix

# ==================== USER CONFIGURATION ====================
TEAM_ID = 3251                      # TEAM_ID: Your unique team ID.
SHOW_IMAGE = True                   # SHOW_IMAGE: Set to True to display a window with detections.
DISABLE_MULTITHREADING = False      # DISABLE_MULTITHREADING: Use single-threaded callbacks if True.

# DEPTH_MEDIAN_KERNEL: Kernel size for the median filter to smooth depth readings.
DEPTH_MEDIAN_KERNEL = 5
# MIN_CONTOUR_AREA: Minimum contour area to be considered a fruit, filters noise.
MIN_CONTOUR_AREA = 200
# MAX_CONTOUR_AREA: Maximum contour area, filters out large non-fruit blobs.
MAX_CONTOUR_AREA = 3000
# MAX_DEPTH: Maximum depth in meters to consider for detection.
MAX_DEPTH = 2.0
# MIN_DEPTH: Minimum valid depth in meters to filter out invalid readings.
MIN_DEPTH = 0.01

# TOP_REGION_HEIGHT_RATIO: The percentage of the bounding box height (from the top) to check for a green top.
TOP_REGION_HEIGHT_RATIO = 0.7
# TOP_REGION_EXTEND_UP: Number of pixels to extend the search region above the bounding box.
TOP_REGION_EXTEND_UP = 0
# MIN_GREEN_PIXELS: Minimum number of green pixels required in the top region to classify as a bad fruit.
MIN_GREEN_PIXELS = 1000

# sizeCamX: Width of the camera image in pixels.
sizeCamX = 1280
# sizeCamY: Height of the camera image in pixels.
sizeCamY = 720
# centerCamX: The x-coordinate of the camera's principal point.
centerCamX = 642.724365234375
# centerCamY: The y-coordinate of the camera's principal point.
centerCamY = 361.9780578613281
# focalX: The focal length of the camera in the x-direction.
focalX = 915.3003540039062
# focalY: The focal length of the camera in the y-direction.
focalY = 914.0320434570312
# ===========================================================


class FruitsTF(Node):
    """
    ROS2 Node for detecting bad fruits (grey body + green top) and publishing TF transforms.
    """

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the ROS2 node, sets up subscribers for color and depth images,
        initializes the TF broadcaster and listener, and creates a timer for the main processing loop.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        node = FruitsTF()
        '''
        super().__init__('fruits_tf_node')

        # self.bridge: An instance of CvBridge to convert between ROS Image messages and OpenCV images.
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # Configure callback groups for single or multi-threaded execution.
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscribe to multiple topic names for robustness across different camera drivers.
        self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depthimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        # self.tf_broadcaster: Publishes TF transforms to the /tf topic.
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # self.tf_buffer: Stores received TF transforms for a period of time.
        self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener: Receives TF transforms and uses the buffer to look them up.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create a timer that calls the process_image function every 0.2 seconds (5 Hz).
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info(f"FruitsTF node started (Team ID: {TEAM_ID})")
        self.get_logger().info("Detection method: Grey body + Green top (final, fixed TFs)")

    def depthimagecb(self, data: Image):
        '''
        Purpose:
        ---
        Callback function for the depth image subscriber. It converts the incoming
        ROS Image message into an OpenCV numpy array and stores it in `self.depth_image`.

        Input Arguments:
        ---
        `data` :  [ sensor_msgs.msg.Image ]
            The depth image message received from the camera topic.

        Returns:
        ---
        None

        Example call:
        ---
        This function is called automatically by the ROS2 subscriber when a new message arrives.
        '''
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warning(f"Depth conversion failed: {e}")
            self.depth_image = None

    def colorimagecb(self, data: Image):
        '''
        Purpose:
        ---
        Callback function for the color image subscriber. It converts the incoming
        ROS Image message into an OpenCV BGR image and stores it in `self.cv_image`.

        Input Arguments:
        ---
        `data` :  [ sensor_msgs.msg.Image ]
            The color image message received from the camera topic.

        Returns:
        ---
        None

        Example call:
        ---
        This function is called automatically by the ROS2 subscriber when a new message arrives.
        '''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f"Color conversion failed: {e}")
            self.cv_image = None

    def bad_fruit_detection(self, rgb_image):
        '''
        Purpose:
        ---
        Detects bad fruits by first identifying greyish objects and then verifying
        that they have a sufficient number of green pixels in their top region.

        Input Arguments:
        ---
        `rgb_image` :  [ numpy.ndarray ]
            The BGR color image in which to detect fruits.

        Returns:
        ---
        `bad_fruits` :  [ list ]
            A list of dictionaries, where each dictionary contains information
            about a detected bad fruit (center, contour, bbox, etc.).

        Example call:
        ---
        fruits = self.bad_fruit_detection(self.cv_image)
        '''
        bad_fruits = []

        if rgb_image is None:
            return bad_fruits

        # Convert the image to HSV color space for more robust color filtering.
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Define HSV range for grey colors (low saturation, medium brightness).
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 60, 200])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

        # Define HSV range for green colors.
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Apply morphological operations to the grey mask to remove noise.
        kernel = np.ones((3, 3), np.uint8)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours of the potential grey fruit bodies.
        contours_gray, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 1
        for contour in contours_gray:
            area = cv2.contourArea(contour)

            # Filter contours based on area to ignore small noise and large blobs.
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Define the region of interest (ROI) at the top of the bounding box to check for green.
            y_top_start = max(y - TOP_REGION_EXTEND_UP, 0)
            y_top_end = y + int(TOP_REGION_HEIGHT_RATIO * h)
            x_left = x
            x_right = x + w

            # Ensure ROI coordinates are within image boundaries.
            img_h, img_w = mask_green.shape[:2]
            y_top_end = min(y_top_end, img_h)
            x_right = min(x_right, img_w)
            top_region = mask_green[y_top_start:y_top_end, x_left:x_right]
            green_pixel_count = int(np.sum(top_region > 0))

            # If there are not enough green pixels, this is not a bad fruit.
            if green_pixel_count < MIN_GREEN_PIXELS:
                continue

            # Calculate the centroid of the contour using image moments for better accuracy.
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            fruit_info = {
                'center': (cX, cY), 'distance': None, 'id': fruit_id,
                'contour': contour, 'bbox': (x, y, w, h), 'green_pixels': green_pixel_count
            }
            bad_fruits.append(fruit_info)
            fruit_id += 1

        return bad_fruits

    def get_depth_at_point(self, x, y):
        '''
        Purpose:
        ---
        Calculates a robust depth value for a given pixel (x, y) by taking the median
        of a small patch around it. This helps to mitigate sensor noise and invalid '0' readings.

        Input Arguments:
        ---
        `x` :  [ int ]
            The x-coordinate of the pixel.
        `y` :  [ int ]
            The y-coordinate of the pixel.

        Returns:
        ---
        `depth_m` :  [ float or None ]
            The depth in meters if a valid reading is found, otherwise None.

        Example call:
        ---
        depth = self.get_depth_at_point(320, 240)
        '''
        if self.depth_image is None:
            return None

        try:
            h, w = self.depth_image.shape[:2]
            px = int(np.clip(x, 0, w - 1))
            py = int(np.clip(y, 0, h - 1))

            # Define the patch around the point for median filtering.
            k = max(1, DEPTH_MEDIAN_KERNEL // 2)
            x0, x1 = max(0, px - k), min(w - 1, px + k)
            y0, y1 = max(0, py - k), min(h - 1, py + k)
            patch = self.depth_image[y0:y1 + 1, x0:x1 + 1]

            if patch.size == 0:
                return None

            # Handle different depth image encodings (uint16 for mm, float for m).
            if patch.dtype == np.uint16:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0] = np.nan
                depth_m = float(np.nanmedian(patch_f)) / 1000.0
            elif patch.dtype in [np.float32, np.float64]:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0.0] = np.nan
                depth_m = float(np.nanmedian(patch_f))
            else: # Fallback for other types
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0] = np.nan
                depth_m = float(np.nanmedian(patch_f))

            # Check if the calculated depth is valid and within our defined range.
            if math.isnan(depth_m) or depth_m < MIN_DEPTH or depth_m > MAX_DEPTH:
                return None

            return depth_m
        except Exception as e:
            self.get_logger().warning(f"Depth read error at ({x},{y}): {e}")
            return None

    def process_image(self):
        '''
        Purpose:
        ---
        This is the main processing function, called by a timer. It orchestrates
        the detection of fruits, calculation of their 3D positions, publishing of TF
        transforms, and visualization of the results.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        This function is called automatically by the ROS2 timer.
        '''
        if self.cv_image is None:
            return

        display_img = self.cv_image.copy()
        bad_fruits = self.bad_fruit_detection(self.cv_image)

        for fruit in bad_fruits:
            cX, cY = fruit['center']
            fruit_id, contour, bbox = fruit['id'], fruit['contour'], fruit['bbox']
            x, y, w, h = bbox
            green_pixels = fruit.get('green_pixels', 0)

            depth_m = self.get_depth_at_point(cX, cY)
            if depth_m is None:
                # If depth is invalid, visualize in orange but do not publish a TF frame.
                cv2.drawContours(display_img, [contour], -1, (0, 165, 255), 2)
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                fruit_name = f"{TEAM_ID}_bad_fruit_{fruit_id}"
                cv2.putText(display_img, f"{fruit_name} (no depth)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                continue
            fruit['distance'] = depth_m

            # Convert 2D pixel coordinates to a 3D point in the camera's optical frame.
            # ROS convention for optical frames is X-right, Y-down, Z-forward.
            # To get a right-handed coordinate system for TF, Y is often inverted.
            x_cam = (cX - centerCamX) * depth_m / focalX
            y_cam = -(cY - centerCamY) * depth_m / focalY  # Inverted Y
            z_cam = depth_m

            # Draw visualizations for successfully detected fruits.
            fruit_name = f"{TEAM_ID}_bad_fruit_{fruit_id}"
            cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)
            cv2.circle(display_img, (cX, cY), 5, (0, 0, 255), -1)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(display_img, fruit_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_img, f"D:{depth_m:.2f}m G:{green_pixels}", (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            try:
                # Get the latest transform from 'base_link' to 'camera_link'.
                trans = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            except Exception as e:
                self.get_logger().warning(f"Could not lookup base_link->camera_link: {e}")
                continue

            q_base_cam = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            t_base_cam = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            T_base_cam = quaternion_matrix(q_base_cam)
            T_base_cam[0:3, 3] = t_base_cam

            # Transform the fruit's position from the camera frame to the base_link frame.
            p_cam_homogeneous = np.array([x_cam, y_cam, z_cam, 1.0])
            p_base_homogeneous = T_base_cam @ p_cam_homogeneous
            p_base = p_base_homogeneous[:3]

            # Create the transform message to be published.
            child_name = f"{TEAM_ID}_bad_fruit_{fruit_id}"
            t_base_obj = TransformStamped()
            t_base_obj.header.stamp = self.get_clock().now().to_msg()
            t_base_obj.header.frame_id = 'base_link'
            t_base_obj.child_frame_id = child_name
            
            # These hardcoded offsets are for manual calibration to align the TF with the object's real-world position.
            t_base_obj.transform.translation.x = float(p_base[0]) + 0.65
            t_base_obj.transform.translation.y = float(p_base[1]) + 0.6
            t_base_obj.transform.translation.z = float(p_base[2]) - 2.2

            # Use an identity quaternion (0,0,0,1) for orientation to keep the frame upright.
            t_base_obj.transform.rotation.x = 0.0
            t_base_obj.transform.rotation.y = 0.0
            t_base_obj.transform.rotation.z = 0.0
            t_base_obj.transform.rotation.w = 1.0

            # Broadcast the final transform.
            self.tf_broadcaster.sendTransform(t_base_obj)

        if SHOW_IMAGE:
            try:
                cv2.imshow('fruits_tf_view', display_img)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warning(f"OpenCV display error: {e}")


def main(args=None):
    '''
    Purpose:
    ---
    The main entry point for the ROS2 node. Initializes rclpy, creates an instance
    of the FruitsTF node, spins the node to process callbacks, and handles shutdown.

    Input Arguments:
    ---
    `args` :  [ list, optional ]
        Command-line arguments passed to the script. Defaults to None.

    Returns:
    ---
    None

    Example call:
    ---
    Called automatically when the script is executed.
    '''
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF node")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
