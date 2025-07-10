import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
from datetime import datetime

class DualJointStateLogger(Node):
    def __init__(self):
        super().__init__('joint_logger')

        # Subscribers to real joint states and target joint states
        self.subscription_real = self.create_subscription(
            JointState,
            '/joint_states',
            self.callback_real,
            10)  # QoS history depth

        self.subscription_target = self.create_subscription(
            JointState,
            '/target_joint_states',
            self.callback_target,
            10)  # QoS history depth

        # Start time reference for elapsed time calculation
        self.start_time = self.get_clock().now()

        # Create CSV file with timestamp in filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f"joint_states_dual_{timestamp}.csv"
        self.csv_file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.csv_file)

        # Variables to store joint positions and names
        self.real_positions = None
        self.target_positions = None
        self.joint_names = None

        self.header_written = False

        # For frequency control: minimum time between recorded samples (in seconds)
        self.min_record_interval = 0.01  # e.g., 10 Hz max recording frequency
        self.last_record_time = 0.0

    def write_header(self, names):
        # Columns: elapsed time, real joints, then target joints
        header = ['time (s)']
        header += [f'real_{name}' for name in names]
        header += [f'target_{name}' for name in names]
        self.writer.writerow(header)
        self.header_written = True

    def callback_real(self, msg: JointState):
        self.real_positions = msg.position
        if not self.joint_names and msg.name:
            self.joint_names = msg.name
        self.try_write_row()

    def callback_target(self, msg: JointState):
        self.target_positions = msg.position
        if not self.joint_names and msg.name:
            self.joint_names = msg.name
        self.try_write_row()

    def try_write_row(self):
        if self.real_positions is not None and self.target_positions is not None and self.joint_names:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

            if elapsed - self.last_record_time >= self.min_record_interval:
                if not self.header_written:
                    self.write_header(self.joint_names)

                row = [elapsed] + list(self.real_positions) + list(self.target_positions)
                self.writer.writerow(row)
                self.last_record_time = elapsed


    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f"Data saved to: {self.filename}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DualJointStateLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
