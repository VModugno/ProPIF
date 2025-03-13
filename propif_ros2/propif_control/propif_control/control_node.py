#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
import os
from enum import Enum
from threading import Lock

# Add torch for Curobo
import torch

# ROS2 messages
from propif_msgs.msg import PlaneInfo
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

# Simulation and Control package
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, CartesianDiffKin

# Curobo for motion planning
try:
    # Updated curobo imports based on documentation
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import JointState as CuroboJointState
    from curobo.types.base import TensorDeviceType
    CUROBO_AVAILABLE = True
except ImportError:
    print("Curobo not found. Path planning will be limited.")

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
        
        # Declare parameters
        self.declare_parameter('use_simulation', True)  # Use PyBullet simulation or real robot
        self.declare_parameter('robot_config', 'pandaconfig.json')  # Robot configuration file
        self.declare_parameter('control_frequency', 100.0)  # Control loop frequency
        self.declare_parameter('approach_distance', 0.1)  # Distance to approach plane
        self.declare_parameter('interaction_duration', 2.0)  # Duration to interact with plant
        self.declare_parameter('curobo_robot_config', 'franka.yml')  # Curobo robot configuration
        
        # Get parameters
        self.use_simulation = self.get_parameter('use_simulation').value
        self.robot_config = self.get_parameter('robot_config').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.interaction_duration = self.get_parameter('interaction_duration').value
        self.curobo_robot_config = self.get_parameter('curobo_robot_config').value
        
        # Initialize control state
        self.state = ControllerState.IDLE
        self.detected_planes = []  # Store detected planes
        self.current_plane = None  # Current target plane
        self.path = None  # Planned path
        self.path_index = 0  # Current index in the path
        self.mutex = Lock()  # Mutex for thread safety
        
        # Curobo tensor device setup
        if CUROBO_AVAILABLE:
            self.tensor_args = TensorDeviceType()
        
        # Initialize simulation and control components
        self.setup_robot_control()
        
        # Initialize Curobo motion planner if available
        if CUROBO_AVAILABLE:
            self.setup_motion_planner()
            
        # ROS subscribers and publishers
        self.plane_subscription = self.create_subscription(
            PlaneInfo, 
            '/detected_planes', 
            self.plane_callback, 
            10)
        
        self.status_publisher = self.create_publisher(
            String, 
            '/control_status', 
            10)
        
        self.path_visualization_publisher = self.create_publisher(
            MarkerArray,
            '/planned_path',
            10)
            
        # Create timer for control loop
        control_period = 1.0 / self.control_frequency  # Convert frequency to period
        self.control_timer = self.create_timer(control_period, self.control_loop)
        
        # Create a one-shot timer for initialization sequence
        self.init_timer = self.create_timer(1.0, self.initialization_sequence_wrapper)
        
        self.get_logger().info('Control node initialized')
    
    def initialization_sequence_wrapper(self):
        """Wrapper for one-time execution of initialization sequence"""
        self.initialization_sequence()
        self.init_timer.cancel()

    def setup_robot_control(self):
        """Initialize robot control components"""
        try:
            # config directory
            config_dir = "/home/steve/UCL_RAI/ProPIF" 
            
            self.get_logger().info(f'Looking for robot config at: {os.path.join(config_dir, self.robot_config)}')
            
            # Initialize simulation interface
            self.sim = pb.SimInterface(
                self.robot_config, 
                conf_file_path_ext=config_dir,
                use_gui=False
            )
            
            # Get active joint names
            ext_names = self.sim.getNameActiveJoints()
            ext_names = np.expand_dims(np.array(ext_names), axis=0)
            source_names = ["pybullet"]
            
            # Create dynamic model
            self.dyn_model = PinWrapper(
                self.robot_config, 
                "pybullet", 
                ext_names, 
                source_names, 
                False, 
                0,
                config_dir
            )
            
            # Get robot configuration
            self.num_joints = self.dyn_model.getNumberofActuatedJoints()
            self.controlled_frame_name = "panda_link8"  # End-effector frame
            self.joint_limits_lower, self.joint_limits_upper = self.sim.GetBotJointsLimit()
            self.joint_vel_limits = self.sim.GetBotJointsVelLimit()
            
            # Initial joint configuration
            self.home_joint_angles = self.sim.GetInitMotorAngles()
            self.current_joint_angles = self.home_joint_angles.copy()
            
            # Controller configuration
            self.cmd = MotorCommands()
            
            # Controller gains
            # High-level Cartesian controller gains
            self.kp_pos = 100.0  # Position gain
            self.kp_ori = 10.0   # Orientation gain
            
            # Low-level joint PD controller gains
            self.kp = 1000.0  # P gain
            self.kd = 100.0   # D gain
            
            self.get_logger().info('Robot control initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize robot control: {str(e)}')
            raise
    
    def setup_motion_planner(self):
        """Initialize Curobo motion planner according to documentation"""
        try:
            # Create a simple world configuration
            world_config = {
                "cuboid": {
                    "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, -0.1, 1, 0, 0, 0]},
                }
            }
            
            # Initialize motion generation configuration 
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.curobo_robot_config,  # Robot config file (franka.yml)
                world_config,              # World collision objects
                interpolation_dt=0.01,     # Interpolation time step
            )
            
            # Create the motion generator
            self.motion_planner = MotionGen(motion_gen_config)
            
            # Warm up the planner (initialize CUDA kernels)
            self.motion_planner.warmup()
            
            # Get joint names from sim for curobo
            joint_names = self.sim.getNameActiveJoints()
            self.joint_names = joint_names
            
            self.get_logger().info('Curobo motion planner initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Curobo motion planner: {str(e)}')
            self.motion_planner = None
    
    def np_to_torch_tensor(self, np_array):
        """Convert numpy array to torch tensor with proper device/dtype"""
        if not CUROBO_AVAILABLE:
            return np_array
        
        # Ensure array is numpy array (not list)
        if not isinstance(np_array, np.ndarray):
            np_array = np.array(np_array)
        
        # Convert to torch tensor with device/dtype from tensor_args
        tensor = torch.tensor(np_array, **self.tensor_args.as_torch_dict())
        return tensor

    def torch_to_np(self, tensor):
        """Convert torch tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def rotation_to_quaternion(self, R):
        """Convert rotation matrix to quaternion - [w,x,y,z] format for Curobo"""
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
                
            # Normalize
            norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
            qw /= norm
            qx /= norm
            qy /= norm
            qz /= norm
            
            # Return in [w,x,y,z] format for Curobo
            return [qw, qx, qy, qz]
            
        except Exception as e:
            self.get_logger().error(f'Error converting rotation to quaternion: {str(e)}')
            return [1.0, 0.0, 0.0, 0.0]  # Identity quaternion [w,x,y,z]
    
    def plan_path_to_target(self):
        """Plan a collision-free path to the target using Curobo"""
        if not CUROBO_AVAILABLE or not self.motion_planner:
            self.get_logger().warn('Curobo not available, using simple joint interpolation')
            return self.plan_simple_path()
            
        if not self.current_plane:
            self.get_logger().error('No target plane available for planning')
            return False
            
        try:
            # Get current robot state
            current_q = self.sim.GetMotorAngles(0)
            
            # Convert to torch tensor for Curobo
            start_state = CuroboJointState.from_position(
                self.np_to_torch_tensor(current_q.reshape(1, -1)),  # Shape [1, num_joints]
                joint_names=self.joint_names
            )
            
            # Define target pose from plane information
            target_position = [
                self.current_plane.centroid.x,
                self.current_plane.centroid.y,
                self.current_plane.centroid.z + self.approach_distance  # Stay at approach distance
            ]
            
            # Compute orientation to align z-axis with plane normal
            normal = [
                self.current_plane.normal.x,
                self.current_plane.normal.y,
                self.current_plane.normal.z
            ]
            
            # Calculate target orientation using our helper function
            rotation_matrix = self.compute_orientation_matrix(normal)
            
            # Convert rotation matrix to quaternion for Curobo [w,x,y,z] format
            target_quaternion = self.rotation_to_quaternion(rotation_matrix)
            
            # Create Curobo Pose object - format is [x, y, z, qw, qx, qy, qz]
            goal_pose = CuroboPose.from_list([
                target_position[0], target_position[1], target_position[2], 
                target_quaternion[0], target_quaternion[1], target_quaternion[2], target_quaternion[3]
            ])
            
            # Plan path using Curobo
            result = self.motion_planner.plan_single(
                start_state, 
                goal_pose,
                MotionGenPlanConfig(max_attempts=10)
            )
            
            if result.success:
                # Get interpolated trajectory
                traj = result.get_interpolated_plan()
                
                # Convert to numpy for our control system
                self.path = self.torch_to_np(traj.position)
                
                self.visualize_path(self.path)
                self.get_logger().info(f'Path planned successfully with {len(self.path)} waypoints')
                return True
            else:
                self.get_logger().warn('Curobo planning failed')
                return self.plan_simple_path()  # Fallback to simple planning
                
        except Exception as e:
            self.get_logger().error(f'Path planning error: {str(e)}')
            return self.plan_simple_path()  # Fallback on error
    def compute_orientation_matrix(self, normal):
        """Compute rotation matrix to align end-effector z-axis with plane normal"""
        try:
            # Normalize the normal vector
            z_axis = np.array(normal, dtype=np.float32)
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # Choose a reference vector that's not parallel to z_axis
            reference = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(z_axis, reference)) > 0.9:
                reference = np.array([1.0, 0.0, 0.0])
                
            # Construct coordinate frame
            x_axis = np.cross(reference, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Form rotation matrix
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            return rotation_matrix
            
        except Exception as e:
            self.get_logger().error(f'Error computing orientation matrix: {str(e)}')
            return np.eye(3)  # Return identity on error
    
    def plane_callback(self, msg):
        """Callback for receiving detected planes"""
        with self.mutex:
            # Add new plane to the list if not already present
            plane_exists = False
            for i, plane in enumerate(self.detected_planes):
                if plane.object_idx == msg.object_idx:
                    # Update existing plane information
                    self.detected_planes[i] = msg
                    plane_exists = True
                    break
            
            if not plane_exists:
                self.detected_planes.append(msg)
                self.get_logger().info(f'New plane detected: {msg.object_idx}')
            
            # If idle, switch to planning state for the first plane
            if self.state == ControllerState.IDLE and self.detected_planes:
                self.current_plane = self.detected_planes[0]
                self.state = ControllerState.PLANNING
                self.publish_status("Planning path to detected plane")
    
    def initialization_sequence(self):
        """Initial sequence to move the robot to home position"""
        self.get_logger().info('Starting initialization sequence')
        # Move to home position
        try:
            q_des = self.home_joint_angles
            qd_des = np.zeros(self.num_joints)
            
            # Execute joint command to move to home position
            self.execute_joint_command(q_des, qd_des, duration=3.0)
            self.get_logger().info('Robot moved to home position')
            
        except Exception as e:
            self.get_logger().error(f'Initialization sequence failed: {str(e)}')
        
        # Switch to IDLE state after initialization
        self.state = ControllerState.IDLE
        self.publish_status("Initialization complete, waiting for targets")
    
    def control_loop(self):
        """Main control loop executed at the specified frequency"""
        try:
            # Get current robot state
            q_current = self.sim.GetMotorAngles(0)
            qd_current = self.sim.GetMotorVelocities(0)
            
            # State machine for robot control
            if self.state == ControllerState.IDLE:
                # In idle state, hold current position
                self.hold_position(q_current, qd_current)
                
            elif self.state == ControllerState.PLANNING:
                # Plan path to target
                with self.mutex:
                    success = self.plan_path_to_target()
                    
                if success:
                    self.state = ControllerState.EXECUTION
                    self.path_index = 0
                    self.publish_status("Executing planned path")
                else:
                    self.get_logger().error('Path planning failed')
                    self.state = ControllerState.IDLE
                    self.publish_status("Planning failed, returning to IDLE")
                
            elif self.state == ControllerState.EXECUTION:
                # Execute current path
                if self.path_index < len(self.path):
                    # Get desired position from path
                    q_des = self.path[self.path_index]
                    qd_des = np.zeros(self.num_joints)  # Zero velocity for waypoints
                    
                    # Execute joint command
                    self.execute_joint_command(q_des, qd_des)
                    
                    # Check if waypoint is reached
                    if self.is_waypoint_reached(q_current, q_des):
                        self.path_index += 1
                else:
                    # Path execution complete, move to approach
                    self.state = ControllerState.APPROACH
                    self.publish_status("Path execution complete, approaching target")
                
            elif self.state == ControllerState.APPROACH:
                # Approach the plane carefully
                success = self.approach_target()
                
                if success:
                    self.state = ControllerState.INTERACTION
                    self.interaction_start_time = self.get_clock().now().seconds_nanoseconds()[0]
                    self.publish_status("Interacting with plant")
                else:
                    self.get_logger().warn('Approach failed, returning to IDLE')
                    self.state = ControllerState.IDLE
                    self.publish_status("Approach failed, returning to IDLE")
                
            elif self.state == ControllerState.INTERACTION:
                # Perform interaction with the plane
                current_time = self.get_clock().now().seconds_nanoseconds()[0]
                
                # Check if interaction time has elapsed
                if current_time - self.interaction_start_time >= self.interaction_duration:
                    self.state = ControllerState.RETURN
                    self.publish_status("Interaction complete, returning to home")
                else:
                    # Execute interaction behavior
                    self.perform_interaction(q_current, qd_current)
                
            elif self.state == ControllerState.RETURN:
                # Return to home position
                success = self.return_to_home()
                
                if success:
                    # Clear the current target and move to the next plane if available
                    with self.mutex:
                        if self.detected_planes:
                            # Remove the processed plane
                            self.detected_planes.pop(0)
                            
                            # Check if there are more planes to process
                            if self.detected_planes:
                                self.current_plane = self.detected_planes[0]
                                self.state = ControllerState.PLANNING
                                self.publish_status("Planning path to next detected plane")
                            else:
                                self.state = ControllerState.IDLE
                                self.current_plane = None
                                self.publish_status("All planes processed, waiting for new targets")
                        else:
                            self.state = ControllerState.IDLE
                            self.current_plane = None
                            self.publish_status("Return complete, waiting for targets")
                else:
                    self.get_logger().warn('Return to home failed')
                    self.state = ControllerState.IDLE
                    self.publish_status("Return failed, manual intervention required")
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {str(e)}')
            self.emergency_stop()
            self.state = ControllerState.IDLE
    
    def plan_simple_path(self):
        """Simple path planning using CartesianDiffKin from simulation_and_control"""
        if not self.current_plane:
            return False
            
        try:
            # Get current robot state
            q_current = self.sim.GetMotorAngles(0)
            
            # Target position from plane information (with approach distance)
            target_pos = np.array([
                self.current_plane.centroid.x,
                self.current_plane.centroid.y,
                self.current_plane.centroid.z + self.approach_distance
            ])
            
            # Get normal vector for orientation
            normal = np.array([
                self.current_plane.normal.x,
                self.current_plane.normal.y,
                self.current_plane.normal.z
            ])
            
            # Create simple path by linear interpolation in joint space
            num_waypoints = 30
            path = []
            current_q = q_current
            
            # Use IK to find goal configuration
            goal_q = self.solve_ik(target_pos, normal, current_q)
            if goal_q is None:
                self.get_logger().warn('IK failed for target position')
                return False
                
            # Generate path by interpolation
            for i in range(num_waypoints):
                alpha = i / (num_waypoints - 1)
                q_interp = (1 - alpha) * current_q + alpha * goal_q
                path.append(q_interp)
                
            self.path = path
            self.get_logger().info(f'Simple path planned with {len(path)} waypoints')
            return True
                
        except Exception as e:
            self.get_logger().error(f'Simple path planning error: {str(e)}')
            return False
    
    def solve_ik(self, target_pos, normal, seed):
        """Solve IK using CartesianDiffKin"""
        try:
            time_step = 0.01
            pd_d = np.zeros(3)  # Zero velocity
            
            # Compute target orientation
            target_rot = self.compute_orientation_matrix(normal)
            
            # Use CartesianDiffKin to solve IK
            q_des, qd_des = CartesianDiffKin(
                self.dyn_model,
                self.controlled_frame_name, 
                seed,                             # Current joint angles
                target_pos,                       # Target position
                pd_d,                             # Target velocity (zero)
                target_rot,                       # Target orientation
                np.zeros((3,3)),                  # Target angular velocity (zero)
                time_step,                        # Time step
                "pos",                            # Control mode
                self.kp_pos,                      # Position gain
                self.kp_ori,                      # Orientation gain
                np.array(self.joint_vel_limits)   # Joint velocity limits
            )
            
            return q_des
            
        except Exception as e:
            self.get_logger().error(f'IK solving error: {str(e)}')
            return None
    
    def execute_joint_command(self, q_des, qd_des, duration=None):
        """Execute a joint command using feedback linearization control"""
        q_mes = self.sim.GetMotorAngles(0)
        qd_mes = self.sim.GetMotorVelocities(0)
        
        # Apply feedback linearization control
        tau_cmd = feedback_lin_ctrl(
            self.dyn_model, 
            q_mes, 
            qd_mes, 
            q_des, 
            qd_des, 
            self.kp, 
            self.kd
        )
        
        # Set and send control command
        self.cmd.SetControlCmd(tau_cmd, ["torque"]*self.num_joints)
        self.sim.Step(self.cmd, "torque")
        
        # If visualizer is enabled, update the model
        if self.dyn_model.visualizer:
            self.dyn_model.DisplayModel(q_mes)
    
    def is_waypoint_reached(self, q_current, q_desired, tolerance=0.05):
        """Check if the current joint configuration is close to the desired one"""
        return np.all(np.abs(q_current - q_desired) < tolerance)
    
    def approach_target(self):
        """Approach the target plane carefully"""
        try:
            if not self.current_plane:
                return False
                
            # Get current position
            q_current = self.sim.GetMotorAngles(0)
            current_pos, current_rot = self.dyn_model.ComputeFK(q_current, self.controlled_frame_name)
            
            # Target position at the plane
            target_pos = np.array([
                self.current_plane.centroid.x,
                self.current_plane.centroid.y,
                self.current_plane.centroid.z
            ])
            
            # Normal for orientation
            normal = np.array([
                self.current_plane.normal.x,
                self.current_plane.normal.y,
                self.current_plane.normal.z
            ])
            
            # Generate slow approach path
            num_steps = 15
            approach_path = []
            
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                interp_pos = (1 - alpha) * current_pos + alpha * target_pos
                
                # Get joint configuration for this position
                q_des = self.solve_ik(interp_pos, normal, q_current)
                if q_des is not None:
                    approach_path.append(q_des)
                    q_current = q_des
            
            # Execute approach path slowly
            for q_des in approach_path:
                qd_des = np.zeros(self.num_joints)
                self.execute_joint_command(q_des, qd_des)
                time.sleep(0.2)  # Slow down approach
                
                # Check for any issues during approach
                if self.check_force_limits():
                    self.get_logger().warn('Force limit exceeded during approach')
                    return False
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Approach error: {str(e)}')
            return False
    
    def perform_interaction(self, q_current, qd_current):
        """Perform interaction with the plant - hold position still for interaction_duration"""
        try:
            # Simply hold the current position
            self.hold_position(q_current, qd_current)
            
            # Log the remaining interaction time
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            remaining_time = self.interaction_duration - (current_time - self.interaction_start_time)
            
            # Log remaining time occasionally (every ~1 second)
            if int(remaining_time) != int(remaining_time + 0.1):  # Only log when the integer second changes
                self.get_logger().info(f'Interaction: holding position for {remaining_time:.1f} more seconds')
                
        except Exception as e:
            self.get_logger().error(f'Interaction error: {str(e)}')
            self.hold_position(q_current, qd_current)
    
    def return_to_home(self):
        """Return to the home position"""
        try:
            q_current = self.sim.GetMotorAngles(0)
            q_home = self.home_joint_angles
            
            # Create interpolated path to home
            num_steps = 30
            return_path = []
            
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                q_interp = (1 - alpha) * q_current + alpha * q_home
                return_path.append(q_interp)
            
            # Execute return path
            for q_des in return_path:
                qd_des = np.zeros(self.num_joints)
                self.execute_joint_command(q_des, qd_des)
                time.sleep(0.05)  # Smoother motion
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Return to home error: {str(e)}')
            return False
    
    def hold_position(self, q_current, qd_current):
        """Hold current position"""
        q_des = q_current
        qd_des = np.zeros(self.num_joints)
        self.execute_joint_command(q_des, qd_des)
    
    def emergency_stop(self):
        """Stop the robot immediately"""
        try:
            # Set zero torque command
            zero_tau = np.zeros(self.num_joints)
            self.cmd.SetControlCmd(zero_tau, ["torque"]*self.num_joints)
            self.sim.Step(self.cmd, "torque")
            self.get_logger().warn('Emergency stop activated')
            
        except Exception as e:
            self.get_logger().error(f'Emergency stop error: {str(e)}')
    
    def check_force_limits(self):
        """Check if force/torque limits are exceeded"""
        # This is a placeholder - would be implemented with real force sensors
        return False
    
    def visualize_path(self, path):
        """Visualize the planned path using MarkerArray"""
        try:
            from visualization_msgs.msg import Marker, MarkerArray
            from geometry_msgs.msg import Point
            from std_msgs.msg import ColorRGBA
            
            marker_array = MarkerArray()
            
            # Create markers for each waypoint
            for i, q in enumerate(path):
                # Compute forward kinematics to get the end-effector position
                pos, _ = self.dyn_model.ComputeFK(q, self.controlled_frame_name)
                
                # Create a sphere marker
                marker = Marker()
                marker.header.frame_id = "world"  # Adjust to your frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "path_waypoints"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # Set position
                marker.pose.position.x = pos[0]
                marker.pose.position.y = pos[1]
                marker.pose.position.z = pos[2]
                marker.pose.orientation.w = 1.0
                
                # Set scale
                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                
                # Set color (gradient from green to blue)
                color = ColorRGBA()
                color.a = 1.0
                color.r = 0.0
                color.g = 1.0 - float(i) / len(path)
                color.b = float(i) / len(path)
                marker.color = color
                
                marker_array.markers.append(marker)
            
            # Create line strip connecting waypoints
            line_strip = Marker()
            line_strip.header.frame_id = "world"  # Adjust to your frame
            line_strip.header.stamp = self.get_clock().now().to_msg()
            line_strip.ns = "path_line"
            line_strip.id = len(path)
            line_strip.type = Marker.LINE_STRIP
            line_strip.action = Marker.ADD
            
            # Set scale
            line_strip.scale.x = 0.01
            
            # Set color
            line_strip.color.a = 1.0
            line_strip.color.r = 1.0
            line_strip.color.g = 1.0
            line_strip.color.b = 0.0
            
            # Add points
            for q in path:
                pos, _ = self.dyn_model.ComputeFK(q, self.controlled_frame_name)
                point = Point()
                point.x = pos[0]
                point.y = pos[1]
                point.z = pos[2]
                line_strip.points.append(point)
            
            marker_array.markers.append(line_strip)
            
            # Publish the marker array
            self.path_visualization_publisher.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f'Path visualization error: {str(e)}')
    
    def publish_status(self, status_text):
        """Publish controller status"""
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
        # Clean up
        control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()