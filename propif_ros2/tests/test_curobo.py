import numpy as np
import time
import os
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
    def __init__(self):
        self.setup_robot_model()
        self.setup_motion_planner()




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
            # Get the pose directly from motion planner's FK method
            state = self.motion_planner.compute_kinematics(home_state)
            pose = state.ee_pose.position.detach().cpu().numpy()
        except Exception as e:
            self.get_logger().error(f'Motion planner error: {e}')
            self.motion_planner = None



    def plan_path_to_target(self,target_pos):
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
        #target_pos = [
        #    self.current_plane.centroid.x + 0.2 * normal[0],
        #    self.current_plane.centroid.y + 0.2 * normal[1],
        #    self.current_plane.centroid.z + 0.2 * normal[2],
        #]
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
            print(f'Motion planning failed: {result.status}')
            return False
    

def main():

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # added this line to manage the fact that the file is in tests folder
    name_current_directory = "tests"
    # remove current directory name from cur_dir
    root_dir = root_dir.replace(name_current_directory, "")
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
    
    # for regulation
    q_des =  init_joint_angles
    qd_des_clip = np.zeros(num_joints)
    
    # Sinusoidal reference for cartesian trajectory tracking
    # Specify different amplitude values for each joint
    amplitudes = [0, 0.1, 0]  # Example amplitudes for 4 joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4]  # Example frequencies for 4 joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency,init_cartesian_pos)  # Initialize the reference
    
    #check = ref.check_sinusoidal_feasibility(sim)  # Check the feasibility of the reference trajectory
    #if not check:
    #    raise ValueError("Sinusoidal reference trajectory is not feasible. Please adjust the amplitude or frequency.")
    
    #simulation_time = sim.GetTimeSinceReset()
    time_step = sim.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    
    # P conttroller high level
    kp_pos = 100 # position 
    kp_ori = 0   # orientation
    
    # PD controller gains low level (feedbacklingain)
    kp = 1000
    kd = 100

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    

    
    # data collection loop
    while True:
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude

        if not regulation_flag:
        
            p_d, pd_d = ref.get_values(current_time)  # Desired position and velocity
            
            # inverse differential kinematics
            ori_des = None
            ori_d_des = None
            q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, p_d, pd_d, ori_des, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque command
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
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des_clip)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        print("current time in seconds",current_time)

    
    num_joints = len(q_mes)
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i+1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i+1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
   
    
    
     
    
    

if __name__ == '__main__':
    main()