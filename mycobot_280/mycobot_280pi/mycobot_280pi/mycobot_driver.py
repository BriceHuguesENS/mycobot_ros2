# ROS 2, time, and math libraries
import rclpy
import time
import numpy as np
from rclpy.node import Node

# myCobot Python API
import pymycobot
from pymycobot import MyCobot280

# ROS 2 message types
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

# Version checking
from packaging import version
import math

# Required pymycobot version
MIN_REQUIRE_VERSION = '3.6.1'

# Check that the pymycobot version is compatible
current_version = pymycobot.__version__
print('Current pymycobot library version: {}'.format(current_version))
if version.parse(current_version) < version.parse(MIN_REQUIRE_VERSION):
    raise RuntimeError(f'The version of pymycobot must be >= {MIN_REQUIRE_VERSION}. Current version: {current_version}')
else:
    print('pymycobot version is valid!')

class MyCobotDriver(Node):
    def __init__(self):
        super().__init__("mycobot_driver")

        # Declare and retrieve parameters for serial connection
        self.declare_parameter('port', '/dev/ttyAMA0')
        self.declare_parameter('baud', 1000000)

        port = self.get_parameter("port").get_parameter_value().string_value
        baud = self.get_parameter("baud").get_parameter_value().integer_value
        self.get_logger().info(f"Connecting to myCobot on port: {port} at baudrate: {baud}")

        # Initialize the MyCobot280 robot
        self.mc = MyCobot280(port, str(baud), debug=False)
        time.sleep(0.05)
        self.mc.set_fresh_mode(1)  # Refresh communication mode
        time.sleep(0.05)
        self.mc.focus_all_servos()

        # Create publishers:
        # - joint_pub publishes current joint angles
        # - marker_pub publishes a 3D marker (e.g. the end-effector position)
        self.joint_pub = self.create_publisher(JointState, "joint_states", 1)
        self.marker_pub = self.create_publisher(Marker, "visualization_marker", 1)

        # Prepare joint state message
        self.joint_state_msg = JointState()
        self.joint_state_msg.name = [
            "joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3",
            "joint5_to_joint4", "joint6_to_joint5", "joint6output_to_joint6"
        ]
        self.joint_state_msg.velocity = [0.0]  # Optional: velocity not used here
        self.joint_state_msg.effort = []       # Optional: effort not measured

        # Subscribe to incoming joint target commands
        self.subscription = self.create_subscription(JointState, '/joint_targets', self.trajectory_callback, 1)

        # Initialize visualization marker (a green sphere on the end-effector)
        self.marker_msg = Marker()
        self.marker_msg.header.frame_id = "/joint1"
        self.marker_msg.ns = "my_namespace"
        self.marker_msg.type = Marker.SPHERE
        self.marker_msg.action = Marker.ADD
        self.marker_msg.scale.x = 0.04
        self.marker_msg.scale.y = 0.04
        self.marker_msg.scale.z = 0.04
        self.marker_msg.color.a = 1.0
        self.marker_msg.color.g = 1.0

        # Publishes initial joint states and end-effector marker
        now = self.get_clock().now().to_msg()

        # Read joint angles (in radians) from the robot
        angles = self.mc.get_radians()
        self.joint_state_msg.header.stamp = now
        self.joint_state_msg.position = angles
        self.joint_pub.publish(self.joint_state_msg)
        self.get_logger().info("Initial position published")
        # Read Cartesian coordinates of the end-effector
        coords = self.mc.get_coords()
        self.marker_msg.header.stamp = now

        # Adjust coordinate axes to ROS convention (swap x/y, invert x)
        self.marker_msg.pose.position.x = coords[1] / 1000 * -1
        self.marker_msg.pose.position.y = coords[0] / 1000
        self.marker_msg.pose.position.z = coords[2] / 1000

        # Publish visualization marker
        self.marker_pub.publish(self.marker_msg)

        self.destroy_publisher(self.joint_pub)
        self.destroy_publisher(self.marker_pub)
        # Set up periodic timer callback (10 Hz)
        #self.timer = self.create_timer(1/10.0, self.timer_callback)
        
    def timer_callback(self):
        """Publishes joint states and end-effector marker"""
        now = self.get_clock().now().to_msg()

        # Read joint angles (in radians) from the robot
        angles = self.mc.get_radians()
        #angles = [0.0,0.0,0.0,0.0,0.0,0.0]

        self.joint_state_msg.header.stamp = now
        self.joint_state_msg.position = angles
        self.joint_pub.publish(self.joint_state_msg)

        # Read Cartesian coordinates of the end-effector
        coords = self.mc.get_coords()
        #coords = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.marker_msg.header.stamp = now

        # Adjust coordinate axes to ROS convention (swap x/y, invert x)
        self.marker_msg.pose.position.x = coords[1] / 1000 * -1
        self.marker_msg.pose.position.y = coords[0] / 1000
        self.marker_msg.pose.position.z = coords[2] / 1000

        # Publish visualization marker
        self.marker_pub.publish(self.marker_msg)

    def trajectory_callback(self, msg):
        """Callback for receiving new joint target positions."""

        # Send new joint positions to the robot (blocking call)

        #start_time = self.get_clock().now()
        self.mc.send_radians(list(msg.position), round(msg.velocity[0]))
        # end_time = self.get_clock().now()
        # elapsed_time = (end_time - start_time).nanoseconds / 1e6  # en millisecondes
        

        #self.get_logger().info(f'Send to motors : {elapsed_time:.2f} ms')
        #self.get_logger().info("Joint command sent to the robot")

    def destroy_node(self):
        # Custom cleanup before shutdown
        self.get_logger().info("Releasing all servos before shutdown.")
        try:
            self.mc.release_all_servos()
        except Exception as e:
            self.get_logger().warn(f"Failed to release servos: {e}")
        return super().destroy_node()




def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)
    node = MyCobotDriver()

    try:
        # Keep the node alive and responsive
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Stopping [mycobot_driver] node.")
    finally:
        # Clean shutdown: stop the node and release resources
        node.get_logger().info("Shutting down [mycobot_driver] node and ROS.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
