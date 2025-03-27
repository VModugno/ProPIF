#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os
from cv_bridge import CvBridge
import cv2

from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from propif_msgs.srv import ExecuteJointCommand, GetRobotState

from simulation_and_control import pb  # PinWrapper etc. removed

import tf2_ros

class SimulationNode(Node):
    def __init__(self):
        super().__init__('simulation_node')
        
        self.declare_parameter('robot_config', 'pandaconfig.json')
        self.declare_parameter('flower_model', 'flower.obj')
        self.declare_parameter('simulation_rate', 100.0)
        self.declare_parameter('gui_enabled', True)
        self.declare_parameter('camera_enabled', True)
        
        self.robot_config = self.get_parameter('robot_config').value
        self.flower_model = self.get_parameter('flower_model').value 
        self.simulation_rate = self.get_parameter('simulation_rate').value
        self.gui_enabled = self.get_parameter('gui_enabled').value
        self.camera_enabled = self.get_parameter('camera_enabled').value
        
        self.load_robot()
        self.load_flower()
        
        if self.camera_enabled:
            self.setup_camera()
            
        self.setup_publishers()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.execute_command_service = self.create_service(
            ExecuteJointCommand, 'execute_joint_command', self.handle_joint_command
        )
        self.robot_state_service = self.create_service(
            GetRobotState, 'get_robot_state', self.handle_get_state
        )
        
        # IK/FK services and related code removed

        self.get_logger().info('Robot control services initialized')
        
        sim_period = 1.0 / self.simulation_rate
        self.sim_timer = self.create_timer(sim_period, self.simulation_loop)
                
        self.get_logger().info('Simulation node initialized')

    def load_robot(self):
        try:
            config_dir = "/home/steve/UCL_RAI/ProPIF"
            self.get_logger().info(f'Loading robot config from: {os.path.join(config_dir, self.robot_config)}')
            self.sim_interface = pb.SimInterface(
                self.robot_config, 
                conf_file_path_ext=config_dir,
                use_gui=self.gui_enabled
            )
            self.robot_id = self.sim_interface.bot[0].bot_pybullet
            self.initial_joint_positions = self.sim_interface.bot[0].init_joint_angles
            self.num_joints = self.sim_interface.bot[0].num_motors
            if not hasattr(self, 'robot_id') or self.robot_id is None:
                raise ValueError("Robot ID not properly initialized")
            if not hasattr(self, 'num_joints') or self.num_joints <= 0:
                raise ValueError(f"Invalid number of joints: {self.num_joints}")
            self.get_logger().info(f'Robot loaded successfully with {self.num_joints} joints')
        except Exception as e:
            self.get_logger().error(f'Failed to load robot: {str(e)}')
            raise

    def load_flower(self):
        try:
            config_dir = "/home/steve/UCL_RAI/ProPIF"
            flower_path = os.path.join(config_dir, "models", "objects", self.flower_model)
            p_client = self.sim_interface.pybullet_client
            flower_position = [1.0, 0.0, 0.05]
            flower_orientation = p_client.getQuaternionFromEuler([1.57, 0, -1.37])
            scale = 0.5
            visual_shape_id = p_client.createVisualShape(
                shapeType=p_client.GEOM_MESH,
                fileName=flower_path,
                meshScale=[scale, scale, scale]
            )
            collision_shape_id = p_client.createCollisionShape(
                shapeType=p_client.GEOM_BOX,
                halfExtents=[0.172, 0.385, 0.16]
            )
            self.flower_id = p_client.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=flower_position,
                baseOrientation=flower_orientation
            )
            texture_path = os.path.join(config_dir, "models", "objects", "texture.png")
            if os.path.exists(texture_path):
                texture_id = p_client.loadTexture(texture_path)
                p_client.changeVisualShape(self.flower_id, -1, textureUniqueId=texture_id)
            self.get_logger().info('Flower model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load flower model: {str(e)}')

    def setup_camera(self):
        try:
            p_client = self.sim_interface.pybullet_client
            self.camera_width = 640
            self.camera_height = 480
            self.camera_fov = 60
            self.camera_aspect = float(self.camera_width) / float(self.camera_height)
            self.near_plane = 0.01
            self.far_plane = 10.0
            self.camera_position = [0, 0, 0]
            self.camera_target = [1, 0, 0]
            self.camera_up = [0, 0, 1]
            self.view_matrix = p_client.computeViewMatrix(
                cameraEyePosition=self.camera_position,
                cameraTargetPosition=self.camera_target,
                cameraUpVector=self.camera_up
            )
            self.projection_matrix = p_client.computeProjectionMatrixFOV(
                fov=self.camera_fov,
                aspect=self.camera_aspect,
                nearVal=self.near_plane,
                farVal=self.far_plane
            )
            self.camera_info_msg = CameraInfo()
            self.camera_info_msg.height = self.camera_height
            self.camera_info_msg.width = self.camera_width
            fx = fy = self.camera_width / (2 * np.tan(np.radians(self.camera_fov / 2)))
            cx = self.camera_width / 2
            cy = self.camera_height / 2
            self.camera_info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            self.camera_info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            self.get_logger().info('Camera setup complete')
        except Exception as e:
            self.get_logger().error(f'Failed to setup camera: {str(e)}')

    def setup_publishers(self):
        try:
            if self.camera_enabled:
                self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
                self.depth_pub = self.create_publisher(Image, '/camera/depth/image_rect_raw', 10)
                self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
                self.bridge = CvBridge()
            self.status_pub = self.create_publisher(String, '/simulation/status', 10)
            self.get_logger().info('Publishers initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to setup publishers: {str(e)}')

    def simulation_loop(self):
        try:
            p_client = self.sim_interface.pybullet_client
            for joint_idx in range(self.num_joints):
                target_pos = self.initial_joint_positions[joint_idx]
                p_client.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=p_client.POSITION_CONTROL,
                    targetPosition=target_pos,
                    positionGain=0.3,
                    velocityGain=1.0,
                    force=500
                )
            p_client.stepSimulation()
            if self.camera_enabled:
                self.update_camera_position()
                self.publish_camera_data()
            self.publish_tf_transforms()
            status_msg = String()
            status_msg.data = "Simulation running"
            self.status_pub.publish(status_msg)
        except Exception as e:
            self.get_logger().error(f'Error in simulation loop: {str(e)}')

    def update_camera_position(self):
        try:
            p_client = self.sim_interface.pybullet_client
            end_effector_link = 7
            link_state = p_client.getLinkState(self.robot_id, end_effector_link)
            link_position = link_state[0]
            link_orientation = link_state[1]
            offset = [0.0, 0.0, 0.05]
            offset_world = p_client.multiplyTransforms(
                link_position, link_orientation, offset, [0, 0, 0, 1]
            )[0]
            self.camera_position = offset_world
            forward = [0.0, 0.0, 0.15]
            target_world = p_client.multiplyTransforms(
                link_position, link_orientation, forward, [0, 0, 0, 1]
            )[0]
            self.camera_target = target_world
            self.camera_up = [0, 0, 1]
            self.view_matrix = p_client.computeViewMatrix(
                cameraEyePosition=self.camera_position,
                cameraTargetPosition=self.camera_target,
                cameraUpVector=self.camera_up
            )
        except Exception as e:
            self.get_logger().error(f'Failed to update camera position: {str(e)}')

    def publish_camera_data(self):
        try:
            p_client = self.sim_interface.pybullet_client
            img_data = p_client.getCameraImage(
                width=self.camera_width,
                height=self.camera_height,
                viewMatrix=self.view_matrix,
                projectionMatrix=self.projection_matrix,
                renderer=p_client.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_array = np.array(img_data[2], dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
            rgb_array = rgb_array[:, :, :3]
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            depth_buffer = np.array(img_data[3], dtype=np.float32)
            depth = self.far_plane * self.near_plane / (
                self.far_plane - (self.far_plane - self.near_plane) * depth_buffer
            )
            now = self.get_clock().now().to_msg()
            rgb_msg = self.bridge.cv2_to_imgmsg(bgr_array, encoding='bgr8')
            rgb_msg.header.stamp = now
            rgb_msg.header.frame_id = 'camera_link'
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = 'camera_link'
            self.camera_info_msg.header.stamp = now
            self.camera_info_msg.header.frame_id = 'camera_link'
            self.rgb_pub.publish(rgb_msg)
            self.depth_pub.publish(depth_msg)
            self.camera_info_pub.publish(self.camera_info_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish camera data: {str(e)}')

    def publish_tf_transforms(self):
        try:
            p_client = self.sim_interface.pybullet_client
            now = self.get_clock().now().to_msg()
            world_to_base = TransformStamped()
            world_to_base.header.stamp = now
            world_to_base.header.frame_id = 'world'
            world_to_base.child_frame_id = 'base_link'
            world_to_base.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(world_to_base)
            for i in range(p_client.getNumJoints(self.robot_id)):
                link_state = p_client.getLinkState(self.robot_id, i)
                link_name = p_client.getJointInfo(self.robot_id, i)[12].decode('utf-8')
                if not link_name:
                    continue
                transform = TransformStamped()
                transform.header.stamp = now
                transform.header.frame_id = 'world'
                transform.child_frame_id = link_name
                position = link_state[0]
                orientation = link_state[1]
                transform.transform.translation.x = position[0]
                transform.transform.translation.y = position[1]
                transform.transform.translation.z = position[2]
                transform.transform.rotation.x = orientation[0]
                transform.transform.rotation.y = orientation[1]
                transform.transform.rotation.z = orientation[2]
                transform.transform.rotation.w = orientation[3]
                self.tf_broadcaster.sendTransform(transform)
            if self.camera_enabled:
                self.publish_camera_transform(now)
        except Exception as e:
            self.get_logger().error(f'Failed to publish TF transforms: {str(e)}')
    
    def publish_camera_transform(self, timestamp):
        camera_transform = TransformStamped()
        camera_transform.header.stamp = timestamp
        camera_transform.header.frame_id = 'world'
        camera_transform.child_frame_id = 'camera_link'
        camera_transform.transform.translation.x = self.camera_position[0]
        camera_transform.transform.translation.y = self.camera_position[1]
        camera_transform.transform.translation.z = self.camera_position[2]
        forward = np.array(self.camera_target) - np.array(self.camera_position)
        forward = forward / np.linalg.norm(forward)
        up = np.array(self.camera_up)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        rot_matrix = np.column_stack((right, up, -forward))
        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rot_matrix)
        camera_transform.transform.rotation.x = qx
        camera_transform.transform.rotation.y = qy
        camera_transform.transform.rotation.z = qz
        camera_transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(camera_transform)
    
    def rotation_matrix_to_quaternion(self, R):
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
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        return qx/norm, qy/norm, qz/norm, qw/norm

    def handle_joint_command(self, request, response):
        try:
            p_client = self.sim_interface.pybullet_client
            self.get_logger().debug('Received joint command')
            for i, pos in enumerate(request.position):
                if i < self.num_joints:
                    p_client.setJointMotorControl2(
                        bodyUniqueId=self.robot_id,
                        jointIndex=i,
                        controlMode=p_client.POSITION_CONTROL,
                        targetPosition=pos,
                        targetVelocity=request.velocity[i] if i < len(request.velocity) else 0,
                        positionGain=0.5,
                        velocityGain=1.0,
                        force=500
                    )
            p_client.stepSimulation()
            response.success = True
        except Exception as e:
            self.get_logger().error(f'Joint command error: {str(e)}')
            response.success = False
        return response

    def handle_get_state(self, request, response):
        try:
            p_client = self.sim_interface.pybullet_client
            joint_positions = []
            joint_velocities = []
            joint_torques = []
            for i in range(self.num_joints):
                js = p_client.getJointState(self.robot_id, i)
                joint_positions.append(js[0])
                joint_velocities.append(js[1])
                joint_torques.append(js[3])
            try:
                lower_limits, upper_limits = self.sim_interface.GetBotJointsLimit()
                velocity_limits = self.sim_interface.GetBotJointsVelLimit()
            except Exception as e:
                self.get_logger().warn(f'Failed to get joint limits: {e}, using defaults')
                lower_limits = [-3.14] * self.num_joints
                upper_limits = [3.14] * self.num_joints
                velocity_limits = [10.0] * self.num_joints
            response.joint_positions = joint_positions
            response.joint_velocities = joint_velocities
            response.joint_torques = joint_torques
            response.joint_limits_lower = lower_limits
            response.joint_limits_upper = upper_limits
            response.joint_vel_limits = velocity_limits
            response.num_joints = self.num_joints
            response.success = True
        except Exception as e:
            self.get_logger().error(f'Get robot state error: {str(e)}')
            response.success = False
        return response

def main(args=None):
    rclpy.init(args=args)
    sim_node = SimulationNode()
    try:
        rclpy.spin(sim_node)
    except KeyboardInterrupt:
        pass
    finally:
        sim_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
