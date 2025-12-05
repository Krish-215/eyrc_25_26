#!/usr/bin/env python3
'''
# Team ID:          <Your Team ID>
# Theme:            <Your Theme Name>
# Author List:      <Your Name(s)>
# Filename:         arm_servo_node.py
# Functions:        __init__, quaternion_to_angular_error, control_loop, main
# Global variables: None
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformException
import math
import time
from tf_transformations import quaternion_multiply, quaternion_inverse

class ArmServoNode(Node):
    '''
    A ROS2 node for servoing a robot arm's end-effector to a series of predefined waypoints
    using Twist commands. It uses a TF listener to get the current pose and publishes
    velocity commands to achieve the desired pose.
    '''
    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the node, publisher, TF listener, waypoints, and control parameters.
        Creates a timer to run the main control loop.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        node = ArmServoNode()
        '''
        super().__init__('arm_servo_node')

        # self.pub: Publisher for sending velocity commands to the arm servoing interface.
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # self.tf_buffer: Stores received TF transforms for a period of time.
        self.tf_buffer = Buffer()
        # self.tf_listener: Receives TF transforms and uses the buffer to look them up.
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # self.waypoints: A list of target poses (x, y, z, qx, qy, qz, qw) in the 'base_link' frame.
        self.waypoints = [
            # WP1
            (-0.214, -0.532, 0.557, 0.707, 0.028, 0.034, 0.707),
            # Lift intermediate
            # Mid intermediate
            (0.0, -0.109, 0.465, 1.566, -0.008, 1.572, 0.776),
            # Approach WP2
            (0.0, -0.109, 0.445, 0.029, 0.997, 0.045, 0.033),
            # WP4
            (0.12, 0.255, 0.415, 0.029, 0.997, 0.045, 0.033),
            # Intermediate between WP4 → WP5
            # WP5 final
            (-0.159, 0.501, 0.415, 0.029, 0.997, 0.045, 0.033),
            # WP3
            (-0.806, 0.010, 0.182, -0.684, 0.726, 0.05, 0.008),
        ]

        # self.current_wp: Index of the current target waypoint in the self.waypoints list.
        self.current_wp = 0
        # self.waiting: Flag to indicate if the arm is pausing at a reached waypoint.
        self.waiting = False
        self.wait_start_time = None
        # self.stuck_counter: Counter to detect if the arm is stuck and not making progress.
        self.stuck_counter = 0
        self.last_pos_error = None
        self.last_orient_error = None

        # --- Control Parameters ---
        self.tolerance_pos = 0.12     # tolerance_pos: Position error tolerance in meters to consider a waypoint reached.
        self.tolerance_ang = 0.12     # tolerance_ang: Angular error tolerance in radians.
        self.kp_pos = 0.5             # kp_pos: Proportional gain for linear velocity control.
        self.kp_ang = 0.6             # kp_ang: Proportional gain for angular velocity control.
        self.max_vel = 0.08           # max_vel: Maximum linear velocity magnitude.
        self.max_omega = 0.4          # max_omega: Maximum angular velocity magnitude.
        self.min_vel = 0.002          # min_vel: Minimum linear velocity to overcome stiction.
        self.min_omega = 0.015        # min_omega: Minimum angular velocity to prevent oscillation near target.

        # self.timer: Timer to call the control_loop function at a fixed rate (20 Hz).
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f"🤖 Loaded {len(self.waypoints)} waypoints (including intermediates)")

    def quaternion_to_angular_error(self, q_current, q_target):
        '''
        Purpose:
        ---
        Calculates the angular error between two quaternions, returning it as a
        rotation vector (axis-angle representation) and the error magnitude.

        Input Arguments:
        ---
        `q_current` :  [ list or tuple ]
            The current orientation as a quaternion [x, y, z, w].
        `q_target` :  [ list or tuple ]
            The target orientation as a quaternion [x, y, z, w].

        Returns:
        ---
        `wx, wy, wz` :  [ float, float, float ]
            The components of the angular velocity vector to correct the error.
        `error_mag` :  [ float ]
            The magnitude of the angular error in radians.

        Example call:
        ---
        wx, wy, wz, err_rad = self.quaternion_to_angular_error([0,0,0,1], [0,0,0.707,0.707])
        '''
        def normalize_quat(q):
            norm = math.sqrt(sum([x**2 for x in q]))
            # Avoid division by zero
            return [x / norm for x in q] if norm > 0 else q

        q_curr_norm = normalize_quat(q_current)
        q_targ_norm = normalize_quat(q_target)

        # Calculate the error quaternion that rotates from current to target.
        q_curr_inv = quaternion_inverse(q_curr_norm)
        q_error = quaternion_multiply(q_targ_norm, q_curr_inv)

        # The angle of rotation is 2 * acos(w_component). Clamp the argument to handle floating point inaccuracies.
        error_mag = 2.0 * math.acos(min(1.0, abs(q_error[3])))

        # If the angle is negligible, there is no rotational error.
        if error_mag < 0.001:
            return 0.0, 0.0, 0.0, error_mag

        # The axis of rotation is the vector part of the quaternion.
        sin_half = math.sqrt(q_error[0]**2 + q_error[1]**2 + q_error[2]**2)
        if sin_half < 0.001:
            return 0.0, 0.0, 0.0, error_mag

        # The rotation vector is axis * angle.
        scale = error_mag / sin_half
        wx = q_error[0] * scale
        wy = q_error[1] * scale
        wz = q_error[2] * scale

        return wx, wy, wz, error_mag

    def control_loop(self):
        '''
        Purpose:
        ---
        This is the main control loop that runs periodically. It gets the current
        end-effector pose, calculates errors to the target waypoint, computes a
        Twist command using a P-controller, and publishes it. It also handles
        waypoint transitions, waiting periods, and stuck detection.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        This function is called automatically by the rclpy.Timer.
        '''
        # Stop if all waypoints have been processed.
        if self.current_wp >= len(self.waypoints):
            self.pub.publish(Twist()) # Publish zero velocity
            if not hasattr(self, 'finished_logged'):
                self.get_logger().info("✅ All waypoints reached. Servo stopped.")
                self.finished_logged = True
            return

        # Handle the waiting period after reaching a waypoint.
        if self.waiting:
            if time.time() - self.wait_start_time >= 1.0:
                self.waiting = False
                self.current_wp += 1
                self.stuck_counter = 0 # Reset stuck counter for new waypoint
                self.last_pos_error = None
                self.last_orient_error = None
                if self.current_wp < len(self.waypoints):
                    self.get_logger().info(f"➡️ Moving to WP{self.current_wp+1}")
            else:
                self.pub.publish(Twist()) # Publish zero velocity while waiting
                return

        try:
            # Lookup the transform from the base frame to the end-effector frame.
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", rclpy.time.Time()
            )

            # Extract current position and orientation.
            cur_x = trans.transform.translation.x
            cur_y = trans.transform.translation.y
            cur_z = trans.transform.translation.z
            q = trans.transform.rotation
            cur_quat = [q.x, q.y, q.z, q.w]

            # Get target position and orientation from the waypoints list.
            tx, ty, tz, qx, qy, qz, qw = self.waypoints[self.current_wp]
            target_quat = [qx, qy, qz, qw]

            # Calculate position error (Euclidean distance).
            dx, dy, dz = tx - cur_x, ty - cur_y, tz - cur_z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            # Calculate orientation error.
            wx, wy, wz, ang_error = self.quaternion_to_angular_error(cur_quat, target_quat)

            # Check if the arm is stuck by monitoring the change in error over time.
            if self.last_pos_error is not None and self.last_orient_error is not None:
                pos_change = abs(dist - self.last_pos_error)
                orient_change = abs(ang_error - self.last_orient_error)
                # If error isn't decreasing, increment stuck counter.
                if pos_change < 0.0005 and orient_change < 0.001:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = max(0, self.stuck_counter - 1) # Decrease if moving
                
                # If stuck for too long (e.g., 60 cycles * 0.05s/cycle = 3s), skip to the next waypoint.
                if self.stuck_counter > 60:
                    self.get_logger().warn(f"⚠️ Stuck at WP{self.current_wp+1}. Skipping to next.")
                    self.pub.publish(Twist())
                    self.wait_start_time = time.time()
                    self.waiting = True
                    self.stuck_counter = 0
                    return

            # Store current errors for the next cycle's stuck detection.
            self.last_pos_error = dist
            self.last_orient_error = ang_error

            twist = Twist()

            # --- Proportional Position Control with dynamic scaling ---
            if dist > self.tolerance_pos:
                # Calculate the unit vector for the direction of movement.
                ux, uy, uz = dx / dist, dy / dist, dz / dist
                # Calculate velocity based on proportional gain and error.
                v = self.kp_pos * dist
                # Slow down when close to the target for a smoother approach.
                if dist > 0.2:
                    v = min(v, self.max_vel)
                else:
                    v = min(v, self.max_vel * 0.5)
                # Ensure a minimum velocity to overcome stiction.
                v = max(v, self.min_vel)
                twist.linear.x = ux * v
                twist.linear.y = uy * v
                twist.linear.z = uz * v

            # --- Proportional Orientation Control ---
            if ang_error > self.tolerance_ang:
                # Calculate angular velocity, clamping to the maximum.
                twist.angular.x = max(-self.max_omega, min(self.kp_ang * wx, self.max_omega))
                twist.angular.y = max(-self.max_omega, min(self.kp_ang * wy, self.max_omega))
                twist.angular.z = max(-self.max_omega, min(self.kp_ang * wz, self.max_omega))
                # Apply a deadband to prevent small oscillations near the target orientation.
                if abs(twist.angular.x) < self.min_omega: twist.angular.x = 0.0
                if abs(twist.angular.y) < self.min_omega: twist.angular.y = 0.0
                if abs(twist.angular.z) < self.min_omega: twist.angular.z = 0.0

            # --- Waypoint Completion Check ---
            if dist <= self.tolerance_pos and ang_error <= self.tolerance_ang:
                wp_labels = ["WP1","Lift","Mid","Approach","WP4","Intermediate","WP5","WP3"]
                wp_label = wp_labels[self.current_wp] if self.current_wp < len(wp_labels) else f"WP{self.current_wp+1}"
                self.get_logger().info(f"✅ Reached {wp_label} | pos_err={dist:.4f}m, orient_err={math.degrees(ang_error):.2f}°")
                self.pub.publish(Twist()) # Stop motion
                self.wait_start_time = time.time()
                self.waiting = True
                self.stuck_counter = 0
                return

            # Publish the calculated velocity command.
            self.pub.publish(twist)

        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")

def main(args=None):
    '''
    Purpose:
    ---
    The main entry point for the ROS2 node. Initializes rclpy, creates an instance
    of the ArmServoNode, spins the node to process callbacks, and handles shutdown.

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
    node = ArmServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()