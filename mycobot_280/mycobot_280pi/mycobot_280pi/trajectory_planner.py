# Import symbolic math (SymPy) and numerical libraries
import sympy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R
from sympy import lambdify

# ROS 2 message types
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

# ROS 2 core libraries
import rclpy
from rclpy.node import Node
import time


class TrajectoryPlanner(Node):
    def __init__(self):
        super().__init__('trajectory_planner')

        # Publisher to send the end-effector target pose
        self.target_pub = self.create_publisher(Pose, 'end_effector_target', 1)
        # Subscriber to receive joint states
        self.state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        # Define symbolic joint variables for FK computation
        self.q = sp.symbols('q1:7')

        # Robot-specific Denavit-Hartenberg parameters
        self.d_num = [131.22, 0, 0, 63.4, 75.05, 45.6]
        self.a_num = [0, -110.4, -96, 0, 0, 0]
        self.alpha = [sp.pi/2, 0, 0, sp.pi/2, -sp.pi/2, 0]
        self.offset = [0, -sp.pi/2, 0, -sp.pi/2, sp.pi/2, 0]

        # Flags and buffers
        self.q_ref = None                     # To store the initial joint positions
        self.joint_state_received = False     # Whether joint state has been received
        self.index = 0                        # Index for publishing trajectory points
        self.trajectory = None                # Placeholder for trajectory

        self.start_time = self.get_clock().now()

        # Start a timer to regularly publish trajectory points (every 0.1s = 10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def joint_state_callback(self, msg):
        """
        Callback triggered on first reception of joint states.
        Computes the trajectory based on current pose.
        """
        if not self.joint_state_received:
            self.q_ref = np.array(msg.position)
            self.joint_state_received = True
            self.get_logger().info("Initial joint state received. Computing trajectory...")
            self.destroy_subscription(self.state_sub)  # No need to listen anymore
            self.compute_trajectory()

    def compute_trajectory(self):
        """
        Computes a circular trajectory in the workspace for the end-effector,
        starting from its current pose (computed via forward kinematics).
        """

        # --- Forward Kinematics (DH method) ---
        T = sp.eye(4)
        for i in range(6):
            theta = self.q[i] + self.offset[i]
            A = sp.Matrix([
                [sp.cos(theta), -sp.sin(theta)*sp.cos(self.alpha[i]),  sp.sin(theta)*sp.sin(self.alpha[i]), self.a_num[i]*sp.cos(theta)],
                [sp.sin(theta),  sp.cos(theta)*sp.cos(self.alpha[i]), -sp.cos(theta)*sp.sin(self.alpha[i]), self.a_num[i]*sp.sin(theta)],
                [0,              sp.sin(self.alpha[i]),               sp.cos(self.alpha[i]),              self.d_num[i]],
                [0,              0,                                   0,                                 1]
            ])
            T = T * A

        # Evaluate FK at initial joint positions
        T_num = T.evalf(subs={self.q[i]: self.q_ref[i] for i in range(6)})
        T_init = np.array(T_num).astype(np.float64)
        pos_init = T_init[:3, 3]
        R_init = R.from_matrix(T_init[:3, :3])
        theta_init = R_init.as_rotvec()

        # --- Circular Position Trajectory ---
        radius = 50.0  # mm
        n_points = 100
        theta = np.linspace(0, 2 * np.pi, n_points + 1)

        # Circle direction vector: define center so pos_init lies on circle
        direction = np.array([1.0, 0.0, 0.0])  # x-axis unit vector
        center = pos_init + radius * direction

        # Generate circular path in XY plane (Z remains constant)
        x = center[0] - radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = np.full_like(x, pos_init[2])
        pos_traj = np.vstack((x, y, z)).T  # shape: (N, 3)

        # --- Orientation Trajectory (constant in this example) ---
        angle = np.deg2rad(0)       # Total rotation angle = 0 (no orientation change)
        axis = np.array([1, 0, 0])  # Arbitrary axis
        angles = np.linspace(0, angle, n_points + 1)
        R_traj = [R_init * R.from_rotvec(a * axis) for a in angles]
        theta_traj = [r.as_rotvec() for r in R_traj]

        # --- Full 6D Trajectory: position + orientation ---
        self.raw_traj = np.hstack((pos_traj, theta_traj))

        # Downsample and pad the trajectory to smooth it at the start
        self.downsampled_traj = self.raw_traj[::1].tolist()
        self.downsampled_traj.insert(0, self.raw_traj[0].tolist())
        self.downsampled_traj.insert(0, self.raw_traj[0].tolist())
        self.downsampled_traj.insert(0, self.raw_traj[0].tolist())
        self.downsampled_traj.append(self.raw_traj[-1].tolist())
        self.downsampled_traj = np.array(self.downsampled_traj)

        self.total_steps = self.downsampled_traj.shape[0]

        self.get_logger().info(f"Trajectory computed with {self.total_steps} steps.")
        self.get_logger().info(f"Trajectory content: \n{self.downsampled_traj} ")

    def timer_callback(self):
        """
        Periodically publishes the next pose in the trajectory.
        Stops when the end of the trajectory is reached.
        """
        if not self.joint_state_received or self.downsampled_traj is None:
            self.get_logger().info("Waiting for initial joint state and trajectory...")
            return

        if self.index < self.total_steps:
            X = self.downsampled_traj[self.index]

            # Create and publish Pose message
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = X[0:3]

            quat = R.from_rotvec(X[3:6]).as_quat()
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            self.target_pub.publish(pose)
            self.index += 1

        elif self.index == self.total_steps:
            self.get_logger().info("End-effector trajectory finished.")
            self.index += 1  # Prevents further publishing
            self.timer.cancel()


def main(args=None):
    """
    Main entry point. Initializes the node and spins until interrupted.
    """
    rclpy.init(args=args)
    node = TrajectoryPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
