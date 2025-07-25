# Import ROS 2 Python client library and utilities
import rclpy
from rclpy.node import Node
import os
import sympy as sp  # symbolic math library (unused here but imported)
from sympy import lambdify
from scipy.spatial.transform import Rotation as R  # for rotation conversions
import math
import numpy as np  # numerical library for array/matrix math

# Import ROS 2 message types for joint states and poses
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose


# Define the robot's Denavit-Hartenberg parameters for kinematics
d = [131.22, 0, 0, 63.4, 75.05, 45.6]             # Link offsets along Z axis
a = [0, -110.4, -96, 0, 0, 0]                     # Link lengths along X axis
alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]    # Link twists around X axis
offset = [0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0]  # Joint angle offsets (to align frames)


def compute_A(a, alpha, d, theta):
    """
    Compute the homogeneous transformation matrix A_i for a single joint i
    using the Denavit-Hartenberg parameters and joint angle theta.
    Returns a 4x4 numpy array representing the transform from frame i-1 to i.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])


def compute_fk(joint_angles):
    """
    Compute the forward kinematics of the robot given the current joint angles.
    Multiplies the chain of transformations to get the end-effector pose relative to base frame.
    Returns a 4x4 homogeneous transformation matrix of the end-effector.
    """
    T = np.eye(4)  # initialize transformation as identity
    for i in range(6):
        theta = joint_angles[i] + offset[i]  # apply joint angle offset
        A = compute_A(a[i], alpha[i], d[i], theta)  # transform for joint i
        T = T @ A  # chain transform
    return T


def compute_jacobian(joint_angles):
    """
    Compute the geometric Jacobian matrix for the robot at given joint angles.
    Returns a 6x6 Jacobian matrix composed of:
     - Top 3 rows: linear velocity components (partial derivatives of EE position)
     - Bottom 3 rows: angular velocity components (rotation axes)
    """
    T = np.eye(4)
    Ts = [T]  # store transformations from base to each joint

    # Compute transformation matrices for all joints
    for i in range(6):
        theta = joint_angles[i] + offset[i]
        A = compute_A(a[i], alpha[i], d[i], theta)
        T = T @ A
        Ts.append(T)

    o_n = Ts[6][:3, 3]  # End-effector position (from last transform)
    Jp = []  # Linear velocity part of Jacobian
    Jo = []  # Angular velocity part of Jacobian

    for i in range(6):
        z_i = Ts[i][:3, 2]  # z-axis of the i-th joint frame (rotation axis)
        o_i = Ts[i][:3, 3]  # origin of the i-th joint frame
        Jp_i = np.cross(z_i, o_n - o_i)  # linear velocity Jacobian component
        Jp.append(Jp_i)
        Jo.append(z_i)  # angular velocity Jacobian component

    # Stack linear and angular parts vertically to form full Jacobian
    J = np.vstack((np.array(Jp).T, np.array(Jo).T))
    return J


class EndEffectorController(Node):
    """
    ROS 2 Node to control the robot end-effector by converting
    desired end-effector pose commands into joint commands using
    Jacobian-based inverse kinematics.
    """

    def __init__(self):
        super().__init__('end_effector_controller')

        self.deltaT = 0.1    # Control loop time step (s)
        self.lmb = 1e-2      # Damping factor for Jacobian pseudo-inverse

        # Publisher for joint target positions to be sent to the robot driver
        self.target_pub = self.create_publisher(JointState, 'joint_targets', 1)

        # Subscriber to receive desired end-effector pose commands
        self.target_sub = self.create_subscription(
            Pose, '/end_effector_target', self.trajectory_callback, 1)

        # Subscriber to receive current joint states from robot
        self.state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 1)

        self.index = 0       # trajectory index (if using trajectories)
        self.speed = 40.0     # joint velocity for publishing (dummy constant speed)

        self.start_time = self.get_clock().now()
        self.old_q_d = None  # stores previous joint positions for control

        # Predefine joint names for the JointState message
        self.joint_msg = JointState()
        self.joint_msg.name = [
            "joint2_to_joint1",
            "joint3_to_joint2",
            "joint4_to_joint3",
            "joint5_to_joint4",
            "joint6_to_joint5",
            "joint6output_to_joint6",
        ]
        self.joint_msg.velocity = [self.speed] * 6  # constant velocity values

    def joint_state_callback(self, joint_msg):
        """
        Callback to update current joint positions from feedback topic.
        Stores the joint positions and then unsubscribes to avoid further updates.
        """
        self.old_q_d = list(joint_msg.position)
        self.destroy_subscription(self.state_sub)

    def trajectory_callback(self, pose_msg):
        """
        Callback triggered on receiving a desired end-effector pose.
        Computes joint commands by inverse kinematics and publishes them.
        """
        # Compute current forward kinematics from latest joint angles
        T_c = compute_fk(list(self.old_q_d))

        pos_c = T_c[:3, 3]  # Current end-effector position

        R_c = R.from_matrix(T_c[:3, :3])  # Current end-effector orientation as rotation object

        # Desired position from the Pose message
        pos_d = np.array([
            pose_msg.position.x,
            pose_msg.position.y,
            pose_msg.position.z
        ])

        # Desired orientation as a quaternion, converted to rotation vector
        R_d = R.from_quat([
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w
        ])
        ori_d = R_d.as_rotvec()

        # Compute position error vector
        err_pos = pos_d - pos_c

        # Compute orientation error as rotation vector
        R_err = R_d * R_c.inv()  # relative rotation from current to desired
        err_ori = R_err.as_rotvec()

        # Log some error norms for debugging
        self.get_logger().info(f"err_pos: {np.linalg.norm(err_pos):.3f}")
        signed_error_y = pos_d[1] - pos_c[1]
        self.get_logger().info(f"signed error along Y: {signed_error_y:.3f}")

        # Proportional control gains for position and orientation
        Kp_pos = 10
        Kp_ori = 1.5

        # Desired twist vector (linear + angular velocities)
        X_dot = np.concatenate([
            Kp_pos * err_pos,
            Kp_ori * err_ori])

        # Compute the Jacobian at current joint positions
        J_num = np.array(compute_jacobian(self.old_q_d))

        # Compute damped pseudo-inverse of Jacobian for stability
        J_pinv = np.linalg.pinv(J_num + self.lmb*np.eye(6))

        # Calculate joint velocities update (gradient descent step)
        dq = J_pinv @ X_dot

        # Update joint commands by integrating velocities
        self.q_d = self.old_q_d + self.deltaT * dq

        # Prepare and publish the joint target message
        self.joint_msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_msg.position = self.q_d.tolist()
        self.joint_msg.velocity = [self.speed] * 6

        self.target_pub.publish(self.joint_msg)
        self.index += 1
        self.old_q_d = self.q_d


def main(args=None):
    """
    Main function: Initialize ROS 2 node and spin to process callbacks until shutdown.
    """
    rclpy.init(args=args)
    node = EndEffectorController()

    try:
        rclpy.spin(node)  # Keep node alive to process incoming messages
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Stopping [end_effector_controller] node.")
    finally:
        node.get_logger().info("Shutting down [end_effector_controller] node and ROS.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
