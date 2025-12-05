#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import euler_from_quaternion

class EEPoseMonitor(Node):
    def __init__(self):
        super().__init__('ee_pose_monitor')

        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to update pose at 10 Hz
        self.timer = self.create_timer(0.1, self.monitor_pose)
        self.get_logger().info("EE Pose Monitor node started. Monitoring 'ee_link' w.r.t 'base_link'.")

    def monitor_pose(self):
        try:
            # Get transform from base_link to ee_link
            trans = self.tf_buffer.lookup_transform(
                'base_link',  # target frame
                'ee_link',    # source frame
                rclpy.time.Time()
            )

            # Position
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            # Orientation (quaternion)
            qx = trans.transform.rotation.x
            qy = trans.transform.rotation.y
            qz = trans.transform.rotation.z
            qw = trans.transform.rotation.w

            # Convert to Euler angles (roll, pitch, yaw)
            roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])

            self.get_logger().info(
                f"Position → x: {x:.3f}, y: {y:.3f}, z: {z:.3f} | "
                f"Orientation → roll: {roll:.3f}, pitch: {pitch:.3f}, yaw: {yaw:.3f}"
            )

        except (TransformException, LookupException, ConnectivityException, ExtrapolationException) as ex:
            self.get_logger().warn(f"TF not ready: {ex}")

def main(args=None):
    rclpy.init(args=args)
    node = EEPoseMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
