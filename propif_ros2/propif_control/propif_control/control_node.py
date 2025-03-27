import rclpy
from rclpy.node import Node
import numpy as np
from enum import Enum
import torch

# Curobo imports
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState as CuroboJointState
from curobo.types.base import TensorDeviceType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType

# ROS message/service imports
from propif_msgs.msg import PlaneInfo
from propif_msgs.srv import ExecuteJointCommand, GetRobotState
from std_msgs.msg import String


class ControllerState(Enum):
    IDLE = 0
    PLANNING = 1
    EXECUTION = 2


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        # Load config, change this path to ${Path_to_project}/Propif/configs/pandaconfig.yaml
        self.curobo_robot_config = "/home/steve/UCL_RAI/ProPIF/configs/pandaconfig.yaml"
        self.control_frequency = 100

        # Internal states
        self.state = ControllerState.IDLE
        self.detected_planes = []
        self.current_plane = None
        self.path = None
        self.path_index = 0
        self.tensor_args = TensorDeviceType()

        # Service clients
        self.joint_command_client = self.create_client(ExecuteJointCommand, 'execute_joint_command')
        while not self.joint_command_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for joint command service...')

        self.state_client = self.create_client(GetRobotState, 'get_robot_state')
        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for robot state service...')

        self.latest_robot_state = None
        self.create_timer(0.1, self.request_robot_state)

        # Robot model + planner setup
        self.setup_robot_model()
        self.setup_motion_planner()

        # ROS subscriptions/publishers
        self.create_subscription(PlaneInfo, '/detected_planes', self.plane_callback, 10)
        self.status_publisher = self.create_publisher(String, '/control_status', 10)

        # Control loop
        self.create_timer(1.0 / self.control_frequency, self.control_loop)

        # One-time init sequence
        self.init_timer = self.create_timer(1.0, self.initialization_sequence_wrapper)

        self.get_logger().info('Control node initialized')

    def request_robot_state(self):
        request = GetRobotState.Request()
        future = self.state_client.call_async(request)
        future.add_done_callback(self.robot_state_response_callback)

    def robot_state_response_callback(self, future):
        try:
            result = future.result()
            if result and result.success:
                self.latest_robot_state = result
        except Exception as e:
            self.get_logger().error(f"Robot state error: {e}")

    def get_robot_state(self):
        return self.latest_robot_state

    def get_initial_robot_state(self, timeout_sec=5.0):
        start = self.get_clock().now().nanoseconds / 1e9
        while self.latest_robot_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.get_clock().now().nanoseconds / 1e9) - start > timeout_sec:
                break
        return self.latest_robot_state

    def setup_robot_model(self):
        try:
            init_state = self.get_initial_robot_state()
            if not init_state:
                raise RuntimeError("Failed to get initial state")
            self.num_joints = init_state.num_joints
            self.home_joint_angles = list(init_state.joint_positions)
            self.joint_names = [f"joint{i+1}" for i in range(self.num_joints)]
            self.get_logger().info('Robot model initialized')
        except Exception as e:
            self.get_logger().error(f'Robot model error: {e}')
            raise

    def setup_motion_planner(self):
        try:
            # Create world config with flower box
            cuboid = Cuboid(
                name="flower_box",
                dims=[0.77, 0.77, 0.32],
                pose=[1.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]
            )
            world_config = WorldConfig(cuboid=[cuboid])
            
            # Use your custom configuration file directly
            config = MotionGenConfig.load_from_robot_config(
                self.curobo_robot_config,  # Your custom YAML file path
                world_config,
                self.tensor_args,
                interpolation_dt=0.01,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                use_cuda_graph=False,
                self_collision_check=True,
                num_ik_seeds=50,            # More IK attempts
                num_trajopt_seeds=10,       # More trajectory optimization seeds
                evaluate_interpolated_trajectory=True
            )
            
            self.motion_planner = MotionGen(config)
            self.motion_planner.warmup(enable_graph=False)
            self.get_logger().info('Motion planner initialized with custom configuration')
        except Exception as e:
            self.get_logger().error(f'Motion planner error: {e}')
            self.motion_planner = None

    def initialization_sequence_wrapper(self):
        self.initialization_sequence()
        self.init_timer.cancel()

    def initialization_sequence(self):
        try:
            q_des = self.home_joint_angles
            self.execute_joint_command(q_des, [0.0] * self.num_joints)
            self.get_logger().info('Robot moved to home')
        except Exception as e:
            self.get_logger().error(f'Initialization error: {e}')
        self.state = ControllerState.IDLE
        self.publish_status("Initialization complete")

    def plane_callback(self, msg):
        if msg:
            if msg.object_idx not in [p.object_idx for p in self.detected_planes]:
                self.detected_planes.append(msg)
                self.get_logger().info(f'Plane detected: {msg.object_idx}')
            if self.state == ControllerState.IDLE and self.detected_planes:
                self.current_plane = self.detected_planes[0]
                self.state = ControllerState.PLANNING
                self.publish_status("Planning path to plane")

    def control_loop(self):
        robot_state = self.get_robot_state()
        if not robot_state:
            return
        q_current = np.array(robot_state.joint_positions)
        if self.state == ControllerState.IDLE:
            self.hold_position(q_current)
        elif self.state == ControllerState.PLANNING:
            if self.plan_path_to_target():
                self.state = ControllerState.EXECUTION
                self.path_index = 0
                self.publish_status("Executing path")
            else:
                self.state = ControllerState.IDLE
                self.publish_status("Planning failed")
        elif self.state == ControllerState.EXECUTION:
            if self.path_index < len(self.path):
                q_des = self.path[self.path_index]
                self.execute_joint_command(q_des, [0.0] * self.num_joints)
                if self.is_waypoint_reached(q_current, q_des):
                    self.path_index += 1
            else:
                self.publish_status("Target reached")
                if self.detected_planes:
                    self.detected_planes.pop(0)
                self.current_plane = self.detected_planes[0] if self.detected_planes else None
                self.state = ControllerState.PLANNING if self.current_plane else ControllerState.IDLE
                self.publish_status("Next plane planning" if self.current_plane else "All planes processed")

    def plan_path_to_target(self):
        if not self.current_plane or not self.motion_planner:
            return False
        robot_state = self.get_robot_state()
        self.get_logger().info(f"Robot state Now: {robot_state}")
        if not robot_state:
            return False
        current_q = np.array(robot_state.joint_positions)
        
        # Add joint limits validation
        lower_limits = np.array(robot_state.joint_limits_lower)
        upper_limits = np.array(robot_state.joint_limits_upper)
        safety_margin = 0.01  # Small margin to stay away from limits
        
        # Check if any joint is at or beyond limits
        for i, (pos, lower, upper) in enumerate(zip(current_q, lower_limits, upper_limits)):
            if pos <= lower + safety_margin or pos >= upper - safety_margin:
                self.get_logger().warn(f"Joint {i+1} at position {pos:.4f} is too close to limits [{lower:.4f}, {upper:.4f}]")
                # Try to move joint slightly away from limit
                if pos <= lower + safety_margin:
                    current_q[i] = lower + 2*safety_margin
                else:
                    current_q[i] = upper - 2*safety_margin
                self.get_logger().info(f"Adjusted joint {i+1} to {current_q[i]:.4f}")
        
        normal = np.array([
            self.current_plane.normal.x,
            self.current_plane.normal.y,
            self.current_plane.normal.z
        ])
        target_pos = [
            self.current_plane.centroid.x + 0.2 * normal[0],
            self.current_plane.centroid.y + 0.2 * normal[1],
            self.current_plane.centroid.z + 0.2 * normal[2],
        ]
        self.get_logger().info(f"Target pos: {target_pos}")

        start_state = CuroboJointState.from_position(
            self.np_to_torch_tensor(current_q.reshape(1, -1)),
            joint_names=self.joint_names
        )
        R = self.compute_orientation_matrix(normal)
        target_quat = self.rotation_to_quaternion(R)
        goal_pose = CuroboPose.from_list([
            target_pos[0], target_pos[1], target_pos[2],
            target_quat[0], target_quat[1], target_quat[2], target_quat[3]
        ])

        self.get_logger().info(f"Start state: {start_state}")
        self.get_logger().info(f"Goal pose: {goal_pose}")
        
        result = self.motion_planner.plan_single(
            start_state, goal_pose,
            MotionGenPlanConfig(max_attempts=100)
        )

        if result.success:
            traj = result.get_interpolated_plan()
            self.path = self.torch_to_np(traj.position)
            self.get_logger().info(f'Path with {len(self.path)} waypoints')
            return True
        else:
            self.get_logger().warn(f'Motion planning failed: {result.status}')
            return False

    def compute_orientation_matrix(self, normal):
        try:
            z = normal / np.linalg.norm(normal)
            ref = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(z, ref)) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            x = np.cross(ref, z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            return np.column_stack((x, y, z))
        except Exception as e:
            self.get_logger().error(f'Orientation error: {e}')
            return np.eye(3)

    def rotation_to_quaternion(self, R):
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        return [qw / norm, qx / norm, qy / norm, qz / norm]

    def np_to_torch_tensor(self, np_array):
        if not isinstance(np_array, np.ndarray):
            np_array = np.array(np_array)
        return torch.tensor(np_array, **self.tensor_args.as_torch_dict())

    def torch_to_np(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    def execute_joint_command(self, q_des, qd_des):
        request = ExecuteJointCommand.Request()
        request.position = q_des.tolist() if isinstance(q_des, np.ndarray) else list(q_des)
        request.velocity = list(qd_des)
        self.joint_command_client.call_async(request)

    def hold_position(self, q_current):
        self.execute_joint_command(q_current, [0.0] * self.num_joints)

    def is_waypoint_reached(self, q_current, q_desired, tol=0.05):
        return np.all(np.abs(q_current - q_desired) < tol)

    def publish_status(self, text):
        msg = String()
        msg.data = f"{self.state.name}: {text}"
        self.status_publisher.publish(msg)
        self.get_logger().info(msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
