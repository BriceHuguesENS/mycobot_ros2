# ROS 2 Python API
import rclpy
import os
from rclpy.node import Node

# ROS 2 message types
from sensor_msgs.msg import JointState

# Numerical library
import numpy as np

class AutoMove(Node):
    def __init__(self):
        super().__init__('auto_move_trajectory')

        # Create a publisher for joint target positions (to be sent to the driver)
        self.target_pub = self.create_publisher(JointState, 'joint_targets', 1)

        # Load the joint trajectory from a NumPy file located in the same directory as this script
        npy_path = os.path.join(os.path.dirname(__file__), 'q_trajectory.npy')
        self.q_trajectory = np.load(npy_path)

        # Log basic info about the loaded trajectory
        self.get_logger().info(f"Trajectory loaded from {npy_path}, shape: {self.q_trajectory.shape}")
        self.get_logger().info(f"Trajectory content:\n{self.q_trajectory}")

        # Define a fixed speed (velocity) for each joint [rad/s]
        self.speed = [30.0] * 6

        # Start at the first index of the trajectory
        self.index = 0

        # Downsample trajectory: take 1 point every 2 (i.e., reduce by factor 2)
        self.q_trajectory = self.q_trajectory[::2]
        self.total_steps = self.q_trajectory.shape[0]

        print(f"Total steps in trajectory: {self.total_steps}")

        # Create a timer to send one joint target every 0.1 second (10 Hz)
        self.timer = self.create_timer(1.0 / 10, self.timer_callback)

        # Store the start time to manage timing logic
        self.start_time = self.get_clock().now()

    def timer_callback(self):
        """Called periodically to send the next joint target in the trajectory."""

        # Compute elapsed time since the last 'start_time' reset
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # Continue sending trajectory points until all have been sent
        if self.index < self.total_steps:

            # Get the current joint configuration from the trajectory
            q = self.q_trajectory[self.index]

            # Create a JointState message with joint names, positions and speed
            msg = JointState()
            msg.name = [
                "joint2_to_joint1",
                "joint3_to_joint2",
                "joint4_to_joint3",
                "joint5_to_joint4",
                "joint6_to_joint5",
                "joint6output_to_joint6",
            ]
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = q.tolist()
            msg.velocity = self.speed

            # Regular sending (after the first 2 points)
            if self.index > 1:
                self.target_pub.publish(msg)
                self.index += 1

            # Initial delay before sending the first two points (wait ~6s)
            if (self.index == 0 or self.index == 1) and elapsed > 6:
                self.target_pub.publish(msg)
                self.index += 1
                # Reset timer reference point
                self.start_time = self.get_clock().now()

        else:
            # Trajectory is complete, stop the timer and log completion
            self.get_logger().info("Trajectory finished.")
            self.timer.cancel()


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)
    node = AutoMove()

    try:
        # Keep the node alive and responsive
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Stopping [auto_move_trajectory] node.")
    except ROSInterruptException:
        node.get_logger().info("ROS interrupt exception. Shutting down.")
    finally:
        # Clean shutdown: stop the node and release resources
        node.get_logger().info("Shutting down [auto_move_trajectory] node and ROS.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
