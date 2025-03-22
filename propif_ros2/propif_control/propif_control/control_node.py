import rclpy
from rclpy.node import Node
import numpy as np
import time
from enum import Enum

import torch
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState as CuroboJointState
from curobo.types.base import TensorDeviceType

from propif_msgs.msg import PlaneInfo
from propif_msgs.srv import ExecuteJointCommand, GetRobotState, ComputeIK, ComputeFK
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

class ControllerState(Enum):
    IDLE = 0
    PLANNING = 1
    EXECUTION = 2
    APPROACH = 3
    INTERACTION = 4
    RETURN = 5

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.declare_parameter('robot_config', 'pandaconfig.json')
        self.declare_parameter('control_frequency', 100.0)
        self.declare_parameter('approach_distance', 0.1)
        self.declare_parameter('interaction_duration', 2.0)
        self.declare_parameter('curobo_robot_config', 'franka.yml')
        
        self.robot_config = self.get_parameter('robot_config').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.interaction_duration = self.get_parameter('interaction_duration').value
        self.curobo_robot_config = self.get_parameter('curobo_robot_config').value
        
        self.state = ControllerState.IDLE
        self.detected_planes = []
        self.current_plane = None
        self.path = None
        self.path_index = 0
        self.tensor_args = TensorDeviceType()

        # 异步更新机器人状态
        self.latest_robot_state = None
        self.robot_state_timer = self.create_timer(0.1, self.request_robot_state)
        
        self.setup_service_clients()
        self.setup_robot_model()
        self.setup_motion_planner()
        
        self.plane_subscription = self.create_subscription(
            PlaneInfo, '/detected_planes', self.plane_callback, 10)
        self.status_publisher = self.create_publisher(String, '/control_status', 10)
        self.path_visualization_publisher = self.create_publisher(
            MarkerArray, '/planned_path', 10)
        
        control_period = 1.0 / self.control_frequency
        self.control_timer = self.create_timer(control_period, self.control_loop)
        self.init_timer = self.create_timer(1.0, self.initialization_sequence_wrapper)
        
        self.get_logger().info('Control node initialized')
    
    def setup_service_clients(self):
        self.joint_command_client = self.create_client(ExecuteJointCommand, 'execute_joint_command')
        while not self.joint_command_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for joint command service...')
        self.get_logger().info('Joint command service available!')
        
        self.state_client = self.create_client(GetRobotState, 'get_robot_state')
        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for robot state service...')
        self.get_logger().info('Robot state service available!')
        
        self.compute_ik_client = self.create_client(ComputeIK, 'compute_ik')
        while not self.compute_ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for compute IK service...')
        self.get_logger().info('Compute IK service available!')
        
        self.compute_fk_client = self.create_client(ComputeFK, 'compute_fk')
        while not self.compute_fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for compute FK service...')
        self.get_logger().info('Compute FK service available!')
    
    def request_robot_state(self):
        request = GetRobotState.Request()
        future = self.state_client.call_async(request)
        future.add_done_callback(self.robot_state_response_callback)
    
    def robot_state_response_callback(self, future):
        try:
            result = future.result()
            if result and result.success:
                self.latest_robot_state = result
            else:
                self.get_logger().warn("Robot state service returned failure")
                self.latest_robot_state = None
        except Exception as e:
            self.get_logger().error(f"Robot state response error: {str(e)}")
            self.latest_robot_state = None

    def get_robot_state(self):
        return self.latest_robot_state

    def get_initial_robot_state(self, timeout_sec=5.0):
        start_time = self.get_clock().now().nanoseconds / 1e9
        while self.latest_robot_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - start_time > timeout_sec:
                break
        return self.latest_robot_state

    def setup_robot_model(self):
        try:
            initial_state = self.get_initial_robot_state()
            if not initial_state:
                raise RuntimeError("Unable to get robot state for initialization")
            # 使用同步获得的状态更新 latest_robot_state
            self.latest_robot_state = initial_state
            self.num_joints = initial_state.num_joints
            self.joint_limits_lower = list(initial_state.joint_limits_lower)
            self.joint_limits_upper = list(initial_state.joint_limits_upper)
            self.joint_vel_limits = list(initial_state.joint_vel_limits)
            self.home_joint_angles = list(initial_state.joint_positions)
            self.current_joint_angles = self.home_joint_angles.copy()
            joint_names = [f"joint{i+1}" for i in range(self.num_joints)]
            self.joint_names = joint_names
            self.controlled_frame_name = "panda_link8"
            self.kp_pos = 100.0
            self.kp_ori = 10.0
            self.kp = 1000.0
            self.kd = 100.0
            self.get_logger().info('Robot model initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize robot model: {str(e)}')
            raise
    
    def setup_motion_planner(self):
        try:
            world_config = {
                "cuboid": {
                    "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, -0.1, 1, 0, 0, 0]},
                }
            }
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.curobo_robot_config,
                world_config,
                interpolation_dt=0.01,
            )
            self.motion_planner = MotionGen(motion_gen_config)
            self.motion_planner.warmup()
            self.get_logger().info('Curobo motion planner initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize motion planner: {str(e)}')
            self.motion_planner = None
    
    def execute_joint_command(self, q_des, qd_des):
        try:
            request = ExecuteJointCommand.Request()
            request.position = q_des.tolist() if isinstance(q_des, np.ndarray) else list(q_des)
            request.velocity = qd_des.tolist() if isinstance(qd_des, np.ndarray) else list(qd_des)
            future = self.joint_command_client.call_async(request)
            return True
        except Exception as e:
            self.get_logger().error(f'Joint command error: {str(e)}')
            return False
    
    def initialization_sequence_wrapper(self):
        self.initialization_sequence()
        self.init_timer.cancel()

    def initialization_sequence(self):
        self.get_logger().info('Starting initialization sequence')
        try:
            q_des = self.home_joint_angles
            qd_des = np.zeros(self.num_joints)
            self.execute_joint_command(q_des, qd_des)
            self.get_logger().info('Robot moved to home position')
        except Exception as e:
            self.get_logger().error(f'Initialization failed: {str(e)}')
        self.state = ControllerState.IDLE
        self.publish_status("Initialization complete")
    
    def plane_callback(self, msg):
        if msg:
            if msg.object_idx not in [p.object_idx for p in self.detected_planes]:
                self.detected_planes.append(msg)
                self.get_logger().info(f'New plane detected: {msg.object_idx}')
            if self.state == ControllerState.IDLE and self.detected_planes:
                self.current_plane = self.detected_planes[0]
                self.state = ControllerState.PLANNING
                self.publish_status("Planning path to detected plane")
    
    def control_loop(self):
        try:
            robot_state = self.get_robot_state()
            if not robot_state:
                return
            q_current = np.array(robot_state.joint_positions)
            qd_current = np.array(robot_state.joint_velocities)
            if self.state == ControllerState.IDLE:
                self.hold_position(q_current, qd_current)
            elif self.state == ControllerState.PLANNING:
                success = self.plan_path_to_target()
                if success:
                    self.state = ControllerState.EXECUTION
                    self.path_index = 0
                    self.publish_status("Executing planned path")
                else:
                    self.state = ControllerState.IDLE
                    self.publish_status("Planning failed")
            elif self.state == ControllerState.EXECUTION:
                if self.path_index < len(self.path):
                    q_des = self.path[self.path_index]
                    qd_des = np.zeros(self.num_joints)
                    self.execute_joint_command(q_des, qd_des)
                    if self.is_waypoint_reached(q_current, q_des):
                        self.path_index += 1
                else:
                    self.state = ControllerState.APPROACH
                    self.publish_status("Approaching target")
            elif self.state == ControllerState.APPROACH:
                success = self.approach_target()
                if success:
                    self.state = ControllerState.INTERACTION
                    self.interaction_start_time = self.get_clock().now().seconds_nanoseconds()[0]
                    self.publish_status("Interacting with plant")
                else:
                    self.state = ControllerState.IDLE
                    self.publish_status("Approach failed")
            elif self.state == ControllerState.INTERACTION:
                current_time = self.get_clock().now().seconds_nanoseconds()[0]
                if current_time - self.interaction_start_time >= self.interaction_duration:
                    self.state = ControllerState.RETURN
                    self.publish_status("Returning to home")
                else:
                    self.perform_interaction(q_current, qd_current)
            elif self.state == ControllerState.RETURN:
                success = self.return_to_home()
                if success:
                    if self.detected_planes:
                        self.detected_planes.pop(0)
                        if self.detected_planes:
                            self.current_plane = self.detected_planes[0]
                            self.state = ControllerState.PLANNING
                            self.publish_status("Planning for next plane")
                        else:
                            self.state = ControllerState.IDLE
                            self.current_plane = None
                            self.publish_status("All planes processed")
                    else:
                        self.state = ControllerState.IDLE
                        self.current_plane = None
                        self.publish_status("Return complete")
                else:
                    self.state = ControllerState.IDLE
                    self.publish_status("Return failed")
        except Exception as e:
            self.get_logger().error(f'Control loop error: {str(e)}')
            self.emergency_stop()
            self.state = ControllerState.IDLE
    
    def np_to_torch_tensor(self, np_array):
        if not isinstance(np_array, np.ndarray):
            np_array = np.array(np_array)
        return torch.tensor(np_array, **self.tensor_args.as_torch_dict())

    def torch_to_np(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def plan_path_to_target(self):
        if not self.current_plane:
            return False
        try:
            self.get_logger().info('Curobo!!!')
            robot_state = self.get_robot_state()
            if not robot_state:
                return False
            current_q = np.array(robot_state.joint_positions)
            normal = np.array([
                self.current_plane.normal.x,
                self.current_plane.normal.y,
                self.current_plane.normal.z
            ])
            target_position = [
                self.current_plane.centroid.x + normal[0] * 0.2,
                self.current_plane.centroid.y + normal[1] * 0.2,
                self.current_plane.centroid.z + normal[2] * 0.2,
            ]
            self.get_logger().info(f"Target position: {target_position}")

            if self.motion_planner:
                start_state = CuroboJointState.from_position(
                    self.np_to_torch_tensor(current_q.reshape(1, -1)),
                    joint_names=self.joint_names
                )
                rotation_matrix = self.compute_orientation_matrix(normal)
                target_quaternion = self.rotation_to_quaternion(rotation_matrix)
                goal_pose = CuroboPose.from_list([
                    target_position[0], target_position[1], target_position[2], 
                    target_quaternion[0], target_quaternion[1], target_quaternion[2], target_quaternion[3]
                ])
                result = self.motion_planner.plan_single(
                    start_state, 
                    goal_pose,
                    MotionGenPlanConfig(max_attempts=10)
                )
                if result.success:
                    traj = result.get_interpolated_plan()
                    self.path = self.torch_to_np(traj.position)
                    self.get_logger().info(f'Path planned with {len(self.path)} waypoints')
                    return True
            return self.plan_simple_path()
        except Exception as e:
            self.get_logger().error(f'Path planning error: {str(e)}')
            return self.plan_simple_path()
    
    def plan_simple_path(self):
        self.get_logger().info('Planning simple path')
        if not self.current_plane:
            return False
        try:
            robot_state = self.get_robot_state()
            if not robot_state:
                return False
            q_current = np.array(robot_state.joint_positions)
            target_pos = np.array([
                self.current_plane.centroid.x,
                self.current_plane.centroid.y,
                self.current_plane.centroid.z + self.approach_distance
            ])
            normal = np.array([
                self.current_plane.normal.x,
                self.current_plane.normal.y,
                self.current_plane.normal.z
            ])
            num_waypoints = 30
            path = []
            goal_q = self.solve_ik(target_pos, normal, q_current)
            if goal_q is None:
                self.get_logger().warn('IK failed for target')
                return False
            for i in range(num_waypoints):
                alpha = i / (num_waypoints - 1)
                q_interp = (1 - alpha) * q_current + alpha * goal_q
                path.append(q_interp)
            self.path = path
            self.get_logger().info(f'Simple path planned with {len(path)} waypoints')
            return True
        except Exception as e:
            self.get_logger().error(f'Simple planning error: {str(e)}')
            return False
    
    def solve_ik(self, target_pos, normal, seed):
        try:
            time_step = 0.01
            target_rot = self.compute_orientation_matrix(normal)
            request = ComputeIK.Request()
            request.seed = seed.tolist() if isinstance(seed, np.ndarray) else list(seed)
            request.target_position = target_pos.tolist() if isinstance(target_pos, np.ndarray) else list(target_pos)
            target_quat = self.rotation_to_quaternion(target_rot)
            request.target_orientation = target_quat
            request.time_step = time_step
            future = self.compute_ik_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.result() is not None and future.result().success:
                return np.array(future.result().joint_angles)
            else:
                self.get_logger().warn('ComputeIK service failed')
                return None
        except Exception as e:
            self.get_logger().error(f'ComputeIK call error: {str(e)}')
            return None
    
    def call_compute_fk(self, joint_positions, controlled_frame_name):
        try:
            request = ComputeFK.Request()
            request.joint_positions = joint_positions.tolist() if isinstance(joint_positions, np.ndarray) else list(joint_positions)
            request.controlled_frame_name = controlled_frame_name
            future = self.compute_fk_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.result() is not None and future.result().success:
                return np.array(future.result().position), np.array(future.result().orientation)
            else:
                self.get_logger().warn('ComputeFK service failed')
                return None, False
        except Exception as e:
            self.get_logger().error(f'ComputeFK call error: {str(e)}')
            return None, False
    
    def compute_orientation_matrix(self, normal):
        try:
            z_axis = np.array(normal, dtype=np.float32)
            z_axis = z_axis / np.linalg.norm(z_axis)
            reference = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(z_axis, reference)) > 0.9:
                reference = np.array([1.0, 0.0, 0.0])
            x_axis = np.cross(reference, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            return rotation_matrix
        except Exception as e:
            self.get_logger().error(f'Orientation matrix error: {str(e)}')
            return np.eye(3)
    
    def rotation_to_quaternion(self, R):
        try:
            trace = R[0,0] + R[1,1] + R[2,2]
            if trace > 0:
                S = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * S
                qx = (R[2,1] - R[1,2]) / S
                qy = (R[0,2] - R[2,0]) / S
                qz = (R[1,0] - R[0,1]) / S
            elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                qw = (R[2,1] - R[1,2]) / S
                qx = 0.25 * S
                qy = (R[0,1] + R[1,0]) / S
                qz = (R[0,2] + R[2,0]) / S
            elif R[1,1] > R[2,2]:
                S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                qw = (R[0,2] - R[2,0]) / S
                qx = (R[0,1] + R[1,0]) / S
                qy = 0.25 * S
                qz = (R[1,2] + R[2,1]) / S
            else:
                S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                qw = (R[1,0] - R[0,1]) / S
                qx = (R[0,2] + R[2,0]) / S
                qy = (R[1,2] + R[2,1]) / S
                qz = 0.25 * S
            norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
            return [qw/norm, qx/norm, qy/norm, qz/norm]
        except Exception as e:
            self.get_logger().error(f'Quaternion conversion error: {str(e)}')
            return [1.0, 0.0, 0.0, 0.0]
    
    def approach_target(self):
        try:
            if not self.current_plane:
                return False
            robot_state = self.get_robot_state()
            if not robot_state:
                return False
            q_current = np.array(robot_state.joint_positions)
            current_pos, success = self.call_compute_fk(q_current, self.controlled_frame_name)
            if not success:
                return False
            target_pos = np.array([
                self.current_plane.centroid.x,
                self.current_plane.centroid.y,
                self.current_plane.centroid.z
            ])
            normal = np.array([
                self.current_plane.normal.x,
                self.current_plane.normal.y,
                self.current_plane.normal.z
            ])
            num_steps = 15
            approach_path = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                interp_pos = (1 - alpha) * current_pos + alpha * target_pos
                q_des = self.solve_ik(interp_pos, normal, q_current)
                if q_des is not None:
                    approach_path.append(q_des)
                    q_current = q_des
            for q_des in approach_path:
                qd_des = np.zeros(self.num_joints)
                self.execute_joint_command(q_des, qd_des)
                time.sleep(0.2)
            return True
        except Exception as e:
            self.get_logger().error(f'Approach error: {str(e)}')
            return False
    
    def perform_interaction(self, q_current, qd_current):
        self.hold_position(q_current, qd_current)
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        remaining_time = self.interaction_duration - (current_time - self.interaction_start_time)
        if int(remaining_time) != int(remaining_time + 0.1):
            self.get_logger().info(f'Interaction: {remaining_time:.1f}s remaining')
    
    def return_to_home(self):
        try:
            robot_state = self.get_robot_state()
            if not robot_state:
                return False
            q_current = np.array(robot_state.joint_positions)
            q_home = np.array(self.home_joint_angles)
            num_steps = 30
            return_path = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                q_interp = (1 - alpha) * q_current + alpha * q_home
                return_path.append(q_interp)
            for q_des in return_path:
                qd_des = np.zeros(self.num_joints)
                self.execute_joint_command(q_des, qd_des)
                time.sleep(0.05)
            return True
        except Exception as e:
            self.get_logger().error(f'Return error: {str(e)}')
            return False
    
    def hold_position(self, q_current, qd_current):
        q_des = q_current
        qd_des = np.zeros(self.num_joints)
        self.execute_joint_command(q_des, qd_des)
    
    def is_waypoint_reached(self, q_current, q_desired, tolerance=0.05):
        return np.all(np.abs(q_current - q_desired) < tolerance)
    
    def emergency_stop(self):
        try:
            robot_state = self.get_robot_state()
            if robot_state:
                request = ExecuteJointCommand.Request()
                request.position = robot_state.joint_positions
                request.velocity = [0.0] * self.num_joints
                self.joint_command_client.call_async(request)
            self.get_logger().warn('Emergency stop activated')
        except Exception as e:
            self.get_logger().error(f'Emergency stop error: {str(e)}')
    
    def publish_status(self, status_text):
        msg = String()
        msg.data = f"{self.state.name}: {status_text}"
        self.status_publisher.publish(msg)
        self.get_logger().info(msg.data)

def main(args=None):
    rclpy.init(args=args)
    control_node = ControlNode()
    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
