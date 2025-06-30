import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
# Curobo imports
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState as CuroboJointState
from curobo.types.base import TensorDeviceType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType

cartesian_flag = True
regulation_flag = True



class curobo_planner:
    def __init__(self, sim, dyn_model):
        self.sim = sim
        self.dyn_model = dyn_model
        self.tensor_args = TensorDeviceType()

        self.curobo_robot_config = "/home/chenzhe/ros_envs/propif_env/src/ProPIF/configs/pandaconfig.yaml"

        # Flower position for orientation calculation
        self.flower_position = [1.0, 0.0, 0.05]

        self.setup_robot_model()
        self.setup_motion_planner()




    def setup_robot_model(self):
        try:
            init_joint_angles = self.sim.GetInitMotorAngles()
            self.num_joints = len(init_joint_angles)
            self.home_joint_angles = list(init_joint_angles)

            self.joint_names = [f"panda_joint{i+1}" for i in range(self.num_joints)]
            print(f'Using joint names: {self.joint_names}')
            print(f'Home joint angles: {self.home_joint_angles}')
            print('Robot model initialized')
        except Exception as e:
            print(f'Robot model error: {e}')
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
            # Get the pose directly from motion planner's FK method
            state = self.motion_planner.compute_kinematics(home_state)
            pose = state.ee_pose.position.detach().cpu().numpy()
            print(f'Home position end-effector pose (x, y, z): {pose[0, :3]}')
            print('Motion planner initialized with custom configuration')
        except Exception as e:
            print(f'Motion planner error: {e}')
            self.motion_planner = None

    def get_current_joint_state(self):
        """Get current joint state from simulation"""
        current_q = self.sim.GetMotorAngles(0)
        return np.array(current_q)
    

    def plan_path_to_target(self,target_pos, target_normal=None):
        # if not self.current_plane or not self.motion_planner:
        #     return False
        # robot_state = self.get_robot_state()
        # self.get_logger().info(f"Robot state Now: {robot_state}")
        # if not robot_state:
        #     return False
        if not self.motion_planner:
            print("Motion planner not initialized")
            return None, False
        
        current_q = self.get_current_joint_state()
        lower_limits, upper_limits = self.sim.GetBotJointsLimit()
        safety_margin = 0.01
        for i, (pos, lower, upper) in enumerate(zip(current_q, lower_limits, upper_limits)):
            if pos <= lower + safety_margin or pos >= upper - safety_margin:
                print(f"Joint {i+1} at position {pos:.4f} is too close to limits [{lower:.4f}, {upper:.4f}]")
                if pos <= lower + safety_margin:
                    current_q[i] = lower + 2 * safety_margin
                else:
                    current_q[i] = upper - 2 * safety_margin
                print(f"Adjusted joint {i+1} to {current_q[i]:.4f}")
        # normal = np.array([
        #     self.current_plane.normal.x,
        #     self.current_plane.normal.y,
        #     self.current_plane.normal.z
        # ])
        #target_pos = [
        #    self.current_plane.centroid.x + 0.2 * normal[0],
        #    self.current_plane.centroid.y + 0.2 * normal[1],
        #    self.current_plane.centroid.z + 0.2 * normal[2],
        #]
        print(f"Target pos: {target_pos}")

        # create start state
        start_state = CuroboJointState.from_position(
            self.np_to_torch_tensor(current_q.reshape(1, -1)),
            joint_names=self.joint_names
        )

        # Compute orientation
        if target_normal is not None:
            normal = np.array(target_normal)
        else:
            direction = np.array(self.flower_position) - np.array(target_pos)
            norm = np.linalg.norm(direction)
            if norm < 1e-3:
                normal = np.array([1.0, 0.0, 0.0])
            else:
                normal = direction / norm
                
        R = self.compute_orientation_matrix(normal)
        target_quat = self.rotation_to_quaternion(R)

        # create goal pose
        goal_pose = CuroboPose.from_list([
            target_pos[0], target_pos[1], target_pos[2],
            target_quat[0], target_quat[1], target_quat[2], target_quat[3]
        ])

        print(f"Start state: {start_state}")
        print(f"Goal pose: {goal_pose}")

        # Plan motion
        result = self.motion_planner.plan_single(
            start_state, goal_pose,
            MotionGenPlanConfig(max_attempts=100)
        )
        if result.success:
            traj = result.get_interpolated_plan()
            trajectory = self.torch_to_np(traj.position)
            print(f'Successfully planned path with {len(trajectory)} waypoints')
            return trajectory, True
        else:
            print(f'Motion planning failed: {result.status}')
            return None, False
    
    def plan_detection_waypoints(self, swing_amplitude=0.3):
        """Plan path to detection waypoints (similar to control_node detection phase)"""
        waypoints = [
            [0.45, 0 - swing_amplitude, 0.83],  # Left point
            [0.4, 0, 0.83],                     # Center point
            [0.45, 0 + swing_amplitude, 0.83], # Right point
        ]

        trajectories = []

        for i,waypoint in enumerate(waypoints):
            print(f"Planning to detection waypoint {i+1}/3: {waypoint}")

            look_at_point = self.flower_position
            direction = np.array(look_at_point) - np.array(waypoint)
            norm = np.linalg.norm(direction)
            if norm < 1e-3:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm

            trajectory, success = self.plan_path_to_target(waypoint, direction)

            if success:
                trajectories.append(trajectory)
                print(f"Successfully planned to waypoint {i+1}")
            else:
                print(f"Failed to plan to waypoint {i+1}")

        return trajectories
    
    def compute_orientation_matrix(self, normal):
        """Compute orientation matrix from normal vector"""
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
            print(f'Orientation error: {e}')
            return np.eye(3)
        
    def rotation_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
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
        """Convert numpy array to torch tensor"""
        if not isinstance(np_array, np.ndarray):
            np_array = np.array(np_array)

        return torch.tensor(np_array, **self.tensor_args.as_torch_dict())
            
    def torch_to_np(self, tensor):
        """Convert torch tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def execute_trajectory(self, trajectory, time_step=0.01):
        """Execute a planned trajectory in simulation"""
        if trajectory is None:
            print("No trajectory to execute")
            return False
            
        print(f"Executing trajectory with {len(trajectory)} waypoints")
        
        # Initialize control
        cmd = MotorCommands()
        kp = 1000
        kd = 100
        
        for i, q_des in enumerate(trajectory):
            # Get current state
            q_mes = self.sim.GetMotorAngles(0)
            qd_mes = self.sim.GetMotorVelocities(0)
            qd_des = np.zeros(self.num_joints)  # Zero desired velocity
            
            # Control command
            tau_cmd = feedback_lin_ctrl(self.dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"]*self.num_joints)
            self.sim.Step(cmd, "torque")

            # Visualization update
            if self.dyn_model.visualizer: 
                for index in range(len(self.sim.bot)):
                    q = self.sim.GetMotorAngles(index)
                    self.dyn_model.DisplayModel(q)
            
            # Check for exit
            keys = self.sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')
            if qKey in keys and keys[qKey] and self.sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Execution interrupted by user")
                return False
                
            time.sleep(time_step)
            
            if i % 10 == 0:  # Print progress every 10 waypoints
                print(f"Executing waypoint {i+1}/{len(trajectory)}")
        
        print("Trajectory execution completed")
        return True
    
def main():

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = "/home/chenzhe/ros_envs/propif_env/src/ProPIF/"
    
    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext = root_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos,init_R = dyn_model.ComputeFK(init_joint_angles,controlled_frame_name)
    # print init joint
    print(f"Initial joint angles: {init_joint_angles}")
    
    # check joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")


    joint_vel_limits = sim.GetBotJointsVelLimit()
    
    print(f"joint vel limits: {joint_vel_limits}")
    
    # Initialize curobo planner
    print("Initializing Curobo planner...")
    try:
        planner = curobo_planner(sim, dyn_model)
        print("Curobo planner initialized successfully!")
        
        # Test 1: Plan to a simple target position
        print("\n=== Test 1: Planning to simple target ===")
        target_pos = [0.5, 0.2, 0.8]
        trajectory, success = planner.plan_path_to_target(target_pos)
        
        if success:
            print("Planning successful! Executing trajectory...")
            planner.execute_trajectory(trajectory, time_step=0.05)
        else:
            print("Planning failed!")
        
        # Test 2: Plan detection waypoints
        print("\n=== Test 2: Planning detection waypoints ===")
        detection_trajectories = planner.plan_detection_waypoints(swing_amplitude=0.3)
        
        for i, traj in enumerate(detection_trajectories):
            if traj is not None:
                print(f"Executing detection trajectory {i+1}...")
                planner.execute_trajectory(traj, time_step=0.05)
                time.sleep(1.0)  # Pause between trajectories
        
        # Test 3: Return to home
        print("\n=== Test 3: Returning to home ===")
        home_trajectory, success = planner.plan_path_to_target(init_cartesian_pos)
        if success:
            planner.execute_trajectory(home_trajectory, time_step=0.05)
        
    except Exception as e:
        print(f"Error initializing or testing curobo planner: {e}")
        import traceback
        traceback.print_exc()

    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
      
    # data collection loop
    while True:
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qd_des = np.zeros(num_joints)
       
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_mes, qd_des, 1000, 100)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*num_joints)  # Set the torque command
        sim.Step(cmd, "torque")  # Simulation step with torque command

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)): # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        time.sleep(0.01)  # Slow down the loop for better visualization
    
   

if __name__ == '__main__':
    main()