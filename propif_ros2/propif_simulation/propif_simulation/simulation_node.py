
import rclpy
from rclpy.node import Node
import numpy as np
import os
from cv_bridge import CvBridge
from threading import Lock
import cv2

# ROS 2 messages
from std_msgs.msg import String
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import Image, CameraInfo

from simulation_and_control import pb  # Reuse your existing simulation interface

# TF2
import tf2_ros

class SimulationNode(Node):
    def __init__(self):
        super().__init__('simulation_node')
        
        # Declare parameters
        self.declare_parameter('robot_config', 'pandaconfig.json')
        self.declare_parameter('flower_model', 'flower.obj')
        self.declare_parameter('simulation_rate', 100.0)  # Hz
        self.declare_parameter('gui_enabled', True)
        self.declare_parameter('camera_enabled', True)
        
        # Get parameters
        self.robot_config = self.get_parameter('robot_config').value
        self.flower_model = self.get_parameter('flower_model').value 
        self.simulation_rate = self.get_parameter('simulation_rate').value
        self.gui_enabled = self.get_parameter('gui_enabled').value
        self.camera_enabled = self.get_parameter('camera_enabled').value
        
        # Initialize simulation components
        self.load_robot()
        self.load_flower()
        self.setup_camera()
        
        # Initialize publishers
        self.setup_publishers()
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Initialize mutex for thread safety
        self.mutex = Lock()
        
        # Create timer for simulation loop
        sim_period = 1.0 / self.simulation_rate
        self.sim_timer = self.create_timer(sim_period, self.simulation_loop)
        
        self.get_logger().info('Simulation node initialized')

    def load_robot(self):
        """Load robot model into simulation"""
        try:
            # Config file path
            config_dir = "/home/steve/UCL_RAI/ProPIF" 
            
            self.get_logger().info(f'Loading robot config from: {os.path.join(config_dir, self.robot_config)}')
            
            # Initialize SimInterface which will load the robot in PyBullet
            self.sim_interface = pb.SimInterface(
                self.robot_config, 
                conf_file_path_ext=config_dir,
                use_gui=True
            )
            
            # Store robot ID for later use
            self.robot_id = self.sim_interface.bot[0].bot_pybullet
            
            # Get initial joint positions
            self.initial_joint_positions = self.sim_interface.bot[0].init_joint_angles
            
            # Get joint information
            self.num_joints = self.sim_interface.bot[0].num_motors
            
            self.get_logger().info(f'Robot loaded successfully with {self.num_joints} joints')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load robot: {str(e)}')
            raise

    def load_flower(self):
        """Load flower model into simulation and apply texture"""
        try:
            # Find the flower model path
            config_dir = "/home/steve/UCL_RAI/ProPIF"
            flower_path = os.path.join(config_dir, "models", "objects", self.flower_model)
            p_client = self.sim_interface.pybullet_client
            
            self.get_logger().info(f'Loading flower model from: {flower_path}')
            
            # Define position for the flower (in front of the robot along y-axis)
            flower_position = [1.0, 0.0, 0.05]
            flower_orientation = p_client.getQuaternionFromEuler([1.57, 0, -1.57])
            
            # Scale factor for flower model (since it's very small based on the OBJ file)
            scale = 0.4  # Scale up by 100x to convert from mm to reasonable size
            
            # Load flower as visual shape only initially
            visual_shape_id = p_client.createVisualShape(
                shapeType=p_client.GEOM_MESH,
                fileName=flower_path,
                meshScale=[scale, scale, scale]
            )
            
            # Create collision shape for the flower (simplified box collision)
            collision_shape_id = p_client.createCollisionShape(
                shapeType=p_client.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05]  # Small collision box
            )
            
            # Create the flower body
            self.flower_id = p_client.createMultiBody(
                baseMass=0.1,  # Light mass
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=flower_position,
                baseOrientation=flower_orientation
            )
            
            # Load and apply the combined texture
            texture_path = os.path.join(config_dir, "models", "objects", "texture.png")
            if os.path.exists(texture_path):
                try:
                    texture_id = p_client.loadTexture(texture_path)
                    p_client.changeVisualShape(
                        self.flower_id,
                        -1,  # Base link
                        textureUniqueId=texture_id
                    )
                    self.get_logger().info(f'Applied texture from: {texture_path}')
                except Exception as e:
                    self.get_logger().error(f'Failed to apply texture: {str(e)}')
                    self.apply_fallback_coloring()
            else:
                self.get_logger().warn(f'Texture file not found: {texture_path}')
                
            self.get_logger().info('Flower model loaded successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load flower model: {str(e)}')

    def setup_camera(self):
        """Setup virtual camera for sensing"""
        if not self.camera_enabled:
            self.get_logger().info('Camera disabled, skipping camera setup')
            return
            
        try:
            # Set up the camera parameters (similar to RealSense)
            p_client = self.sim_interface.pybullet_client
            self.camera_width = 640
            self.camera_height = 480
            self.camera_fov = 60  # Field of view in degrees
            self.camera_aspect = float(self.camera_width) / float(self.camera_height)
            self.near_plane = 0.01  # Near clipping plane
            self.far_plane = 10.0   # Far clipping plane
            
            # Camera position: at the end effector
            # We'll update this in the simulation loop
            self.camera_position = [0, 0, 0]
            self.camera_target = [1, 0, 0]  # Looking forward along x-axis
            self.camera_up = [0, 0, 1]      # Z-axis up
            
            # Create camera matrices
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
            
            # Create camera info message
            self.camera_info_msg = CameraInfo()
            self.camera_info_msg.height = self.camera_height
            self.camera_info_msg.width = self.camera_width
            
            # Approximate intrinsic matrix (focal lengths and principal point)
            fx = fy = self.camera_width / (2 * np.tan(np.radians(self.camera_fov / 2)))
            cx = self.camera_width / 2
            cy = self.camera_height / 2
            
            # Set camera matrix (K)
            self.camera_info_msg.k = [
                fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0
            ]
            
            # For simplicity, set P matrix same as K with additional zeros
            self.camera_info_msg.p = [
                fx, 0.0, cx, 0.0,
                0.0, fy, cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            
            # Identity rotation matrix for D (no distortion)
            self.camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Set R (rectification matrix) to identity
            self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            
            self.get_logger().info('Camera setup complete')
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup camera: {str(e)}')

    def setup_publishers(self):
        """Initialize ROS publishers"""
        try:
            # Camera data publishers
            if self.camera_enabled:
                self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
                self.depth_pub = self.create_publisher(Image, '/camera/depth/image_rect_raw', 10)
                self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
            
            # Status publisher
            self.status_pub = self.create_publisher(String, '/simulation/status', 10)
            
            # Create image bridge
            self.bridge = CvBridge()
            
            self.get_logger().info('Publishers initialized')
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup publishers: {str(e)}')

    def simulation_loop(self):
        """Main simulation loop"""
        try:
            # Step the simulation
            self.sim_interface.pybullet_client.stepSimulation()
            
            # Update camera position and orientation
            if self.camera_enabled:
                self.update_camera_position()
                self.publish_camera_data()
            
            # Publish TF transforms
            self.publish_tf_transforms()
            
            # Publish simulation status
            status_msg = String()
            status_msg.data = "Simulation running"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in simulation loop: {str(e)}')

    def update_camera_position(self):
        """Update camera position based on end effector location"""
        try:
            p_client = self.sim_interface.pybullet_client
            # Get the end effector link index
            end_effector_link = 7  # For Panda arm, link 7 is last arm link (before hand)
            
            # Get link state
            link_state = p_client.getLinkState(self.robot_id, end_effector_link)
            link_position = link_state[0]  # World position of link
            link_orientation = link_state[1]  # World orientation of link
            
            # Calculate camera position (mounted on end effector)
            offset = [0.0, 0.0, 0.05]
            
            # Transform offset to world frame
            offset_world = p_client.multiplyTransforms(
                link_position, 
                link_orientation,
                offset, 
                [0, 0, 0, 1]
            )[0]
            
            self.camera_position = offset_world
            
            forward = [0.0, 0.0, 0.15]  # Look 15cm forward along z-axis of end effector
            
            target_world = p_client.multiplyTransforms(
                link_position,
                link_orientation,
                forward,
                [0, 0, 0, 1]
            )[0]
            
            self.camera_target = target_world
            
            up = [0, 0, 1]
            
            self.camera_up = up
            
            # Update view matrix
            self.view_matrix = p_client.computeViewMatrix(
                cameraEyePosition=self.camera_position,
                cameraTargetPosition=self.camera_target,
                cameraUpVector=self.camera_up
            )
            
        except Exception as e:
            self.get_logger().error(f'Failed to update camera position: {str(e)}')

    def publish_camera_data(self):
        """Publish simulated camera data"""
        if not self.camera_enabled:
            return
            
        try:
            p_client = self.sim_interface.pybullet_client
            # Capture RGB and depth images from the camera
            img_data = p_client.getCameraImage(
                width=self.camera_width,
                height=self.camera_height,
                viewMatrix=self.view_matrix,
                projectionMatrix=self.projection_matrix,
                renderer=p_client.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Extract RGB image
            rgb_array = np.array(img_data[2], dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
            rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
            
            # Convert RGB to BGR for OpenCV compatibility
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # Extract depth image
            depth_buffer = np.array(img_data[3], dtype=np.float32)
            far = self.far_plane
            near = self.near_plane
            
            # Convert depth buffer to real depth
            depth = far * near / (far - (far - near) * depth_buffer)
            
            # Create ROS messages
            rgb_msg = self.bridge.cv2_to_imgmsg(bgr_array, encoding='bgr8')
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
            
            # Set message headers
            now = self.get_clock().now().to_msg()
            rgb_msg.header.stamp = now
            rgb_msg.header.frame_id = 'camera_link'
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = 'camera_link'
            
            # Update camera info timestamp
            self.camera_info_msg.header.stamp = now
            self.camera_info_msg.header.frame_id = 'camera_link'
            
            # Publish messages
            self.rgb_pub.publish(rgb_msg)
            self.depth_pub.publish(depth_msg)
            self.camera_info_pub.publish(self.camera_info_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish camera data: {str(e)}')

    def publish_tf_transforms(self):
        """Publish TF transforms for robot links"""
        try:
            p_client = self.sim_interface.pybullet_client
            # Create TF broadcaster if not already created
            if not hasattr(self, 'tf_broadcaster'):
                self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            
            # Get current time
            now = self.get_clock().now().to_msg()
            
            # Publish world -> base transform
            world_to_base = TransformStamped()
            world_to_base.header.stamp = now
            world_to_base.header.frame_id = 'world'
            world_to_base.child_frame_id = 'base_link'
            world_to_base.transform.translation.x = 0.0
            world_to_base.transform.translation.y = 0.0
            world_to_base.transform.translation.z = 0.0
            world_to_base.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(world_to_base)
            
            # Publish transforms for all robot links
            for i in range(p_client.getNumJoints(self.robot_id)):
                link_state = p_client.getLinkState(self.robot_id, i)
                link_name = p_client.getJointInfo(self.robot_id, i)[12].decode('utf-8')
                
                if not link_name:
                    continue
                    
                # Get link position and orientation
                position = link_state[0]
                orientation = link_state[1]
                
                # Create transform message
                transform = TransformStamped()
                transform.header.stamp = now
                transform.header.frame_id = 'world'
                transform.child_frame_id = link_name
                
                # Set translation
                transform.transform.translation.x = position[0]
                transform.transform.translation.y = position[1]
                transform.transform.translation.z = position[2]
                
                # Set rotation
                transform.transform.rotation.x = orientation[0]
                transform.transform.rotation.y = orientation[1]
                transform.transform.rotation.z = orientation[2]
                transform.transform.rotation.w = orientation[3]
                
                # Send transform
                self.tf_broadcaster.sendTransform(transform)
                
            # Publish camera link transform
            if self.camera_enabled:
                # Create camera transform from the camera position and orientation
                camera_transform = TransformStamped()
                camera_transform.header.stamp = now
                camera_transform.header.frame_id = 'world'
                camera_transform.child_frame_id = 'camera_link'
                
                # Set translation from camera position
                camera_transform.transform.translation.x = self.camera_position[0]
                camera_transform.transform.translation.y = self.camera_position[1]
                camera_transform.transform.translation.z = self.camera_position[2]
                
                # Calculate rotation from camera orientation vectors
                # This is a simplified way to get a quaternion from the camera orientation
                # For better accuracy, use a proper orientation calculation method
                forward = np.array(self.camera_target) - np.array(self.camera_position)
                forward = forward / np.linalg.norm(forward)
                
                up = np.array(self.camera_up)
                up = up / np.linalg.norm(up)
                
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                
                # Recompute up to ensure orthogonality
                up = np.cross(right, forward)
                
                # Create rotation matrix
                rot_matrix = np.column_stack((right, up, -forward))  # Camera looks along -z axis
                
                # Convert rotation matrix to quaternion
                trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]
                
                if trace > 0:
                    S = np.sqrt(trace + 1.0) * 2
                    qw = 0.25 * S
                    qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
                    qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
                    qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
                elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                    S = np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
                    qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
                    qx = 0.25 * S
                    qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
                    qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
                elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                    S = np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
                    qw = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
                    qx = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
                    qy = 0.25 * S
                    qz = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
                else:
                    S = np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
                    qw = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
                    qx = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
                    qy = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
                    qz = 0.25 * S
                
                # Normalize quaternion
                qnorm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                qw /= qnorm
                qx /= qnorm
                qy /= qnorm
                qz /= qnorm
                
                # Set rotation
                camera_transform.transform.rotation.x = qx
                camera_transform.transform.rotation.y = qy
                camera_transform.transform.rotation.z = qz
                camera_transform.transform.rotation.w = qw
                
                # Send transform
                self.tf_broadcaster.sendTransform(camera_transform)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish TF transforms: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    sim_node = SimulationNode()
    
    try:
        rclpy.spin(sim_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        sim_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()