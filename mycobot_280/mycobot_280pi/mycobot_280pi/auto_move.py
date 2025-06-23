import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class AutoMove(Node):
    def __init__(self):
        super().__init__('auto_move')

        # Publisher to send target joint states for the robot driver node
        self.target_pub = self.create_publisher(JointState, 'target_joint_states', 10)

        # Define the initial and target joint positions in radians
        self.initial_pos = [0, 0, 0, 0, 0, 0]
        self.target_pos = [-0.00, -0.78, 0.78, 1.59, -1.59, -0.00]

        # Timer to control the publishing sequence
        self.step = 0
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz timer

        # Record time when node starts
        self.start_time = self.get_clock().now()

    def timer_callback(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

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

        if self.step == 0:
            # Publish initial position
            msg.position = self.initial_pos
            self.target_pub.publish(msg)
            self.get_logger().info("Published initial position")
            if elapsed > 3.0:
                self.step = 1

        elif self.step == 1:
            # Publish target position after 3 seconds
            msg.position = self.target_pos
            self.target_pub.publish(msg)
            self.get_logger().info("Published target position")
            # Once target is sent, stop timer (optional)
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = AutoMove()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
