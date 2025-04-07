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
from propif_msgs.srv import ExecuteJointTrajectory, GetRobotState
from propif_msgs.msg import PlaneInfo
from std_msgs.msg import String

class ControllerState(Enum):
    DETECTION = 0
    EXECUTION = 1
    RETURN_HOME = 2
    IDLE = 3

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        # Load configuration
        self.curobo_robot_config = "/home/steve/UCL_RAI/ProPIF/configs/pandaconfig.yaml"
        self.control_frequency = 100

        # Initialize state variables
        self.state = ControllerState.DETECTION
        self.detected_planes = []
        self.current_plane = None
        self.last_detection_time = None
        self.detection_start_time = None
        self.execution_index = 0
        self.execution_wait_start = None
        self.tensor_args = TensorDeviceType()

        # Directly initialize parameters
        self.flower_position = [1.0, 0.0, 0.05]
        self.swing_amplitude = 0.3     # Swing amplitude (m)
        self.swing_period = 10.0        # Swing period (s)
        self.detection_timeout = 20.0   # Patient detection timeout (s)

        # Trajectory Service Client
        self.trajectory_client = self.create_client(ExecuteJointTrajectory, 'execute_joint_trajectory')
        while not self.trajectory_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for trajectory service...')

        self.state_client = self.create_client(GetRobotState, 'get_robot_state')
        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for robot state service...')

        # Add trajectory execution status tracking
        self.trajectory_executing = False
        
        # Create a subscriber for trajectory status
        self.create_subscription(
            String, 
            'trajectory_status', 
            self.trajectory_status_callback, 
            10
        )

        self.latest_robot_state = None
        self.create_timer(0.1, self.request_robot_state)

        self.setup_robot_model()
        self.setup_motion_planner()

        # Subscribe to PlaneInfo messages
        self.create_subscription(PlaneInfo, '/detected_planes', self.plane_callback, 10)
        # Publish status messages
        self.status_publisher = self.create_publisher(String, '/control_status', 10)

        # Initialization sequence
        self.init_timer = self.create_timer(1.0, self.initialization_sequence_wrapper)

        # Control loop timer
        self.create_timer(1.0 / self.control_frequency, self.control_loop)

        self.get_logger().info('Control node initialized, starting in DETECTION phase')

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
            
            # Use actual joint names from Panda robot
            self.joint_names = [f"panda_joint{i+1}" for i in range(self.num_joints)]
            self.get_logger().info(f'Using joint names: {self.joint_names}')
            self.get_logger().info('Robot model initialized')
        except Exception as e:
            self.get_logger().error(f'Robot model error: {e}')
            raise

    def setup_motion_planner(self):
        try:
            # create a cuboid object for the flower model
            cuboid = Cuboid(
                name="flower_box",
                dims=[0.77, 0.77, 0.32],
                pose=[1.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]
            )
            world_config = WorldConfig(cuboid=[cuboid])
            config = MotionGenConfig.load_from_robot_config(
                self.curobo_robot_config,
                world_config,
                self.tensor_args,
                interpolation_dt=0.01,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                use_cuda_graph=False,
                self_collision_check=True,
                num_ik_seeds=50,
                num_trajopt_seeds=10,
                evaluate_interpolated_trajectory=True
            )
            self.motion_planner = MotionGen(config)
            self.motion_planner.warmup(enable_graph=False)

            # Print initial end-effector pose using forward kinematics
            home_state = CuroboJointState.from_position(
                self.np_to_torch_tensor(np.array(self.home_joint_angles).reshape(1, -1)),
                joint_names=self.joint_names
            )
            # In setup_motion_planner:
            self.get_logger().info(f'Joint names used: {self.joint_names}')
            self.get_logger().info(f'Home joint angles: {self.home_joint_angles}')
            # Get the pose directly from motion planner's FK method
            state = self.motion_planner.compute_kinematics(home_state)
            pose = state.ee_pose.position.detach().cpu().numpy()
            
            self.get_logger().info(f'Home position end-effector pose (x, y, z): {pose[0, :3]}')
            
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
            self.send_trajectory(np.array([q_des]), 1.0)
            self.get_logger().info('Robot moved to home')
        except Exception as e:
            self.get_logger().error(f'Initialization error: {e}')
        self.publish_status("Initialization complete, starting DETECTION phase")

    def plane_callback(self, msg):
        if msg:
            if msg.object_idx not in [p.object_idx for p in self.detected_planes]:
                self.detected_planes.append(msg)
                self.get_logger().info(f'Plane detected: {msg.object_idx}')
            # Refresh the last detection time
            self.last_detection_time = self.get_clock().now().nanoseconds / 1e9

    def trajectory_status_callback(self, msg):
        """Callback function for trajectory status messages"""
        if msg.data == "Trajectory execution finished":
            self.get_logger().info("Trajectory execution completed, ready to move to next point")
            self.trajectory_executing = False
            
            # If in detection phase, update to the next target point
            if self.state == ControllerState.DETECTION:
                self.waypoint_idx = (self.waypoint_idx + 1) % len(self.waypoints)

    def control_loop(self):
        robot_state = self.get_robot_state()
        if not robot_state:
            return
        current_q = np.array(robot_state.joint_positions)
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.state == ControllerState.DETECTION:
            if not hasattr(self, 'waypoint_idx'):
                # Initialize detection phase fixed points
                self.detection_start_time = current_time
                self.last_plan_time = 0
                self.waypoint_idx = 0
                self.planning_in_progress = False
                self.trajectory_executing = False

                # Three fixed points: left, center, right
                self.waypoints = [
                    [0.45, 0 - self.swing_amplitude, 0.83],  # Left point
                    [0.4, 0, 0.83],                         # Center point
                    [0.45, 0 + self.swing_amplitude, 0.83],  # Right point
                ]
                
                self.get_logger().info(f'Detection phase fixed points configured: {self.waypoints}')
            
            # Check if new trajectory planning is needed (if not currently planning or executing)
            if not self.planning_in_progress and not self.trajectory_executing and current_time - self.last_plan_time >= 0.5:
                self.last_plan_time = current_time
                
                # Get current target point
                target_pos = self.waypoints[self.waypoint_idx]
                
                # Calculate direction toward the flower
                look_at_point = self.flower_position
                direction = np.array(look_at_point) - np.array(target_pos)
                norm = np.linalg.norm(direction)
                if norm < 1e-3:
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction = direction / norm
                
                # Create rotation matrix and quaternion
                R = self.compute_orientation_matrix(direction)
                target_quat = self.rotation_to_quaternion(R)
                
                goal_pose = CuroboPose.from_list([
                    target_pos[0], target_pos[1], target_pos[2],
                    target_quat[0], target_quat[1], target_quat[2], target_quat[3]
                ])
                
                start_state = CuroboJointState.from_position(
                    self.np_to_torch_tensor(current_q.reshape(1, -1)),
                    joint_names=self.joint_names
                )
                
                self.planning_in_progress = True
                self.get_logger().info(f'Planning to fixed point {self.waypoint_idx+1}/3: {target_pos}')
                
                result = self.motion_planner.plan_single(
                    start_state, goal_pose,
                    MotionGenPlanConfig(max_attempts=10)
                )
                
                self.planning_in_progress = False
                
                if result.success:
                    traj = result.get_interpolated_plan()
                    trajectory = self.torch_to_np(traj.position)
                    self.send_trajectory(trajectory, 0.05)
                    self.trajectory_executing = True  # Mark trajectory as executing
                    self.get_logger().info(f'Successfully planned to point {self.waypoint_idx+1}/3, target: {target_pos}')
                    # Do not update waypoint_idx here - will be updated in trajectory_status_callback
                else:
                    self.get_logger().warn(f'Planning to point {self.waypoint_idx+1}/3 failed: {result.status}')
                    # Try next point
                    self.waypoint_idx = (self.waypoint_idx + 1) % len(self.waypoints)
            
            # Detection timeout check
            if self.last_detection_time and (current_time - self.last_detection_time) > self.detection_timeout:
                self.get_logger().info("Detection timeout reached, switching to EXECUTION phase")
                self.state = ControllerState.EXECUTION
                self.execution_index = 0
                self.execution_wait_start = None

        elif self.state == ControllerState.EXECUTION:
            if self.execution_index < len(self.detected_planes):
                if self.execution_wait_start is None:
                    self.current_plane = self.detected_planes[self.execution_index]
                    success = self.plan_path_to_target()
                    if success:
                        self.execution_wait_start = current_time
                    else:
                        self.get_logger().warn("Planning failed for target, skipping")
                        self.execution_index += 1
                else:
                    # Interact with the target plane for 2 seconds
                    if (current_time - self.execution_wait_start) >= 2.0:
                        self.execution_wait_start = None
                        self.execution_index += 1
            else:
                self.get_logger().info("All targets executed, switching to RETURN_HOME phase")
                self.state = ControllerState.RETURN_HOME

        elif self.state == ControllerState.RETURN_HOME:
            home_q = np.array(self.home_joint_angles)
            traj = np.array([home_q])
            self.send_trajectory(traj, 1.0)
            self.get_logger().info("Returning home, task complete")
            self.state = ControllerState.IDLE
            self.publish_status("Returned to home, task complete")

        elif self.state == ControllerState.IDLE:
            self.hold_position(current_q)

    def plan_path_to_target(self):
        if not self.current_plane or not self.motion_planner:
            return False
        robot_state = self.get_robot_state()
        self.get_logger().info(f"Robot state Now: {robot_state}")
        if not robot_state:
            return False
        current_q = np.array(robot_state.joint_positions)
        lower_limits = np.array(robot_state.joint_limits_lower)
        upper_limits = np.array(robot_state.joint_limits_upper)
        safety_margin = 0.01
        for i, (pos, lower, upper) in enumerate(zip(current_q, lower_limits, upper_limits)):
            if pos <= lower + safety_margin or pos >= upper - safety_margin:
                self.get_logger().warn(f"Joint {i+1} at position {pos:.4f} is too close to limits [{lower:.4f}, {upper:.4f}]")
                if pos <= lower + safety_margin:
                    current_q[i] = lower + 2 * safety_margin
                else:
                    current_q[i] = upper - 2 * safety_margin
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
            trajectory = self.torch_to_np(traj.position)
            self.get_logger().info(f'Path with {len(trajectory)} waypoints')
            self.send_trajectory(trajectory, 0.01)
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

    def send_trajectory(self, trajectory, time_step):
        """Send trajectory to execution service"""
        flattened_traj = trajectory.flatten().tolist()
        request = ExecuteJointTrajectory.Request()
        request.trajectory = flattened_traj
        request.num_waypoints = trajectory.shape[0]
        request.time_step = time_step
        future = self.trajectory_client.call_async(request)
        future.add_done_callback(self.trajectory_response_callback)

    def trajectory_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                pass
            else:
                self.get_logger().warn("Trajectory execution failed")
        except Exception as e:
            self.get_logger().error(f"Trajectory service call failed: {e}")

    def hold_position(self, q_current):
        self.send_trajectory(np.array([q_current]), 0.1)

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
