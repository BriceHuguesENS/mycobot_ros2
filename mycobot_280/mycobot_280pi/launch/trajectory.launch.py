# Standard library
import os

# ROS 2 Python API for locating package resources
from ament_index_python import get_package_share_directory

# ROS 2 launch API: Node and parameter support
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# ROS 2 launch core
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():
    # List to hold all launch elements
    res = []

    # Declare a launch argument for the robot model (URDF or XACRO file)
    model_launch_arg = DeclareLaunchArgument(
        "model",
        default_value=os.path.join(
            get_package_share_directory("mycobot_description"),
            "urdf/mycobot_280_pi/mycobot_280_pi.urdf"
        )
    )
    res.append(model_launch_arg)

    # Declare a launch argument for the RViz configuration file
    rvizconfig_launch_arg = DeclareLaunchArgument(
        "rvizconfig",
        default_value=os.path.join(
            get_package_share_directory("mycobot_280pi"),
            "config/mycobot_pi.rviz"
        )
    )
    res.append(rvizconfig_launch_arg)

    # Generate the robot_description parameter by processing the XACRO/URDF file
    robot_description = ParameterValue(
        Command(['xacro ', LaunchConfiguration('model')]),
        value_type=str
    )

    # Launch the robot_state_publisher node with the robot_description parameter
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{'robot_description': robot_description}]
    )
    res.append(robot_state_publisher_node)

    # Launch RViz with the given configuration
    # rviz_node = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="screen",
    #     arguments=['-d', LaunchConfiguration("rvizconfig")]
    # )
    # res.append(rviz_node)

    # Launch the auto_move node that precomputed and publishes end-effector trajectory
    trajectory_planner_node = Node(
        package="mycobot_280pi",
        executable="trajectory_planner",
        name="trajectory_planner",
        output="screen"
    )
    res.append(trajectory_planner_node)

    # Launch the auto_move node that precomputed and publishes joint trajectory
    end_effector_controller_node = Node(
        package="mycobot_280pi",
        executable="end_effector_controller",
        name="end_effector_controller",
        output="screen"
    )
    res.append(end_effector_controller_node)

        # Launch the custom non-blocking driver for the MyCobot 280 Pi
    mycobot_driver_node = Node(
        package="mycobot_280pi",
        executable="mycobot_driver",
        name="mycobot_driver",
        output="screen"
    )
    res.append(mycobot_driver_node)

    # Return the complete launch description
    return LaunchDescription(res)
