import sympy as sp
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from sympy import lambdify


def main():

    # --- Denavit-Hartenberg (DH) parameters of the robot (in mm) ---
    d_num = [131.22, 0, 0, 63.4, 75.05, 45.6]  # Link offsets along z-axis
    a_num = [0, -110.4, -96, 0, 0, 0]          # Link lengths along x-axis
    alpha = [sp.pi/2, 0, 0, sp.pi/2, -sp.pi/2, 0]  # Link twists
    offset = [0, -sp.pi/2, 0, -sp.pi/2, sp.pi/2, 0]  # Joint angle offsets

    # --- Define symbolic joint variables q1 to q6 ---
    q = sp.symbols('q1:7')  # Generates (q1, q2, ..., q6)

    # --- Compute symbolic forward kinematics: list of transformation matrices ---
    T = sp.eye(4)       # Start with identity matrix
    T_list = []         # List to store each transformation up to joint i
    for i in range(6):
        theta = q[i] + offset[i]  # Apply joint offset
        # Standard DH homogeneous transformation matrix
        A = sp.Matrix([
            [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha[i]),  sp.sin(theta)*sp.sin(alpha[i]), a_num[i]*sp.cos(theta)],
            [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha[i]), -sp.cos(theta)*sp.sin(alpha[i]), a_num[i]*sp.sin(theta)],
            [0,             sp.sin(alpha[i]),                sp.cos(alpha[i]),                d_num[i]],
            [0,             0,                              0,                              1]
        ])
        T = T * A           # Accumulate transformations
        T_list.append(T)    # Save transformation up to joint i

    # --- Compute end-effector position (symbolically) ---
    P6 = T_list[-1] @ sp.Matrix([0, 0, 0, 1])  # Position of the tool tip in base frame

    # --- Geometric Jacobian (symbolic) ---

    # Linear velocity components (partial derivatives of position wrt joint angles)
    Jp1 = sp.Matrix([0, 0, 1]).cross(sp.Matrix(T_list[5][:3, 3]))  # Base joint axis is fixed (z0)
    Jp2 = sp.Matrix(T_list[0][:3, 2]).cross(T_list[5][:3, 3] - T_list[0][:3, 3])
    Jp3 = sp.Matrix(T_list[1][:3, 2]).cross(T_list[5][:3, 3] - T_list[1][:3, 3])
    Jp4 = sp.Matrix(T_list[2][:3, 2]).cross(T_list[5][:3, 3] - T_list[2][:3, 3])
    Jp5 = sp.Matrix(T_list[3][:3, 2]).cross(T_list[5][:3, 3] - T_list[3][:3, 3])
    Jp6 = sp.Matrix(T_list[4][:3, 2]).cross(T_list[5][:3, 3] - T_list[4][:3, 3])

    # Angular velocity components (joint axes)
    Jo1 = sp.Matrix([0, 0, 1])
    Jo2 = sp.Matrix(T_list[0][:3, 2])
    Jo3 = sp.Matrix(T_list[1][:3, 2])
    Jo4 = sp.Matrix(T_list[2][:3, 2])
    Jo5 = sp.Matrix(T_list[3][:3, 2])
    Jo6 = sp.Matrix(T_list[4][:3, 2])

    # Stack linear and angular parts to get full Jacobian
    Jp = sp.Matrix.hstack(Jp1, Jp2, Jp3, Jp4, Jp5, Jp6)
    Jo = sp.Matrix.hstack(Jo1, Jo2, Jo3, Jo4, Jo5, Jo6)
    J = Jp.col_join(Jo)

    # --- Initial joint configuration (radians) ---
    q_ref = np.array([ 2.37939736e-06, -1.44626527e+00, -9.22147454e-01,  7.97579321e-01, -1.19893945e-05, -1.30407455e-05]) 
    q_ref_subs = {q[i]: q_ref[i] for i in range(6)}  # Substitution dict for symbolic expressions

    # --- Forward kinematics for initial pose ---
    T_init = np.array(T_list[-1].evalf(subs=q_ref_subs)).astype(np.float64).reshape((4,4))
    pos_init = T_init[:3, 3]                       # Initial position
    R_init = R.from_matrix(T_init[:3, :3])         # Initial orientation as Rotation object
    theta_init = R_init.as_rotvec()                # Rotation vector representation

    # --- Target pose: translation only (100mm backward in x) ---
    pos_target = pos_init + np.array([-100, 0, 0])  # Desired new position 

    # --- Linear interpolation of position waypoints ---
    n_points = int(np.linalg.norm(pos_target - pos_init))  # Number of waypoints (1 mm resolution)
    pos_traj = np.linspace(pos_init, pos_target, n_points + 1)  # Include final point

    # --- Orientation trajectory: here no rotation applied ---
    angle = 0
    axis = np.array([1, 0, 0])  # Arbitrary axis, not used since angle = 0

    # Rotation interpolation (returns identity at each step since angle = 0)
    angles = np.linspace(0, angle, n_points+1)
    R_traj = [R_init * R.from_rotvec(a*axis) for a in angles]
    theta_traj = [R.as_rotvec() for R in R_traj]

    # --- Combine position and orientation into 6D pose vectors ---
    trajectory = np.hstack((pos_traj, theta_traj))
    print(trajectory)

    # --- Convert symbolic FK and Jacobian to fast numerical functions ---
    fk_func = lambdify(q, T_list[-1], modules='numpy')
    jac_func = lambdify(q, J, modules='numpy')

    # --- Control parameters ---
    alpha = 0.01        # Gradient step size
    tol_pos = 0.5       # Position error tolerance (mm)
    tol_theta = 0.5     # Orientation error tolerance (rad)
    max_iter = 100      # Max iterations per waypoint
    lmb = 1e-2          # Damping factor for pseudo-inverse

    q_c = q_ref.copy()  # Current joint config
    q_traj = []         # List to store trajectory in joint space

    # --- Cartesian inverse kinematics loop ---
    for X_d in trajectory:  # For each desired pose in the trajectory
        for _ in range(max_iter):
            T_c = np.array(fk_func(*q_c)).astype(np.float64).reshape((4,4))  # Current FK
            pos_c = T_c[:3, 3]  # Current position
            R_c = T_c[:3, :3]   # Current rotation
            theta_c = R.from_matrix(R_c).as_rotvec()  # Rotation vector
            X_c = np.hstack((pos_c, theta_c))         # Full current 6D pose

            err = X_d - X_c            # Pose error
            err_pos = err[:len(pos_c)]  # Position error
            err_theta = err[len(pos_c):]  # Orientation error

            # Check convergence
            if np.linalg.norm(err_pos) < tol_pos and np.linalg.norm(err_theta) < tol_theta:
                break

            # Evaluate Jacobian and compute damped pseudo-inverse
            J_num = np.array(jac_func(*q_c)).astype(np.float64)
            J_pinv = np.linalg.pinv(J_num + lmb*np.eye(6))  # Damped pseudo-inverse

            dq = J_pinv @ err       # Compute joint update
            q_c += alpha * dq       # Gradient descent step

        q_traj.append(q_c.copy())  # Save joint configuration
        print(f"Waypoint reached: pos={pos_c}, rotvec={theta_c}")
    
    # --- Final trajectory adjustments: add pen-up waypoints before and after ---
    q_traj = np.array(q_traj)
    q_traj = np.vstack([
        [2.39848704e-08, -1.22722971e+00, -1.05608828e+00,  7.12488919e-01, -6.21767914e-07, -4.40515907e-05],
        q_traj,
        [2.17406457e-05, -8.83055743e-01, -2.34339494e+00,  1.65564885e+00, -7.33503017e-05, -1.39782069e-04]
    ])

    # --- Output and save ---
    print(q_traj)
    print("Trajectory computation done.")
    print("FINAL ERROR: " + str(np.linalg.norm(err_pos)))  # Final position error (last point)

    # Save to file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'q_trajectory.npy')
    np.save(file_path, np.array(q_traj))
    print("Trajectory saved.")

if __name__ == '__main__':
    main()
