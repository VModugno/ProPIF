import rclpy
from rclpy.node import Node
import cv2
import os
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image, CameraInfo
from propif_msgs.msg import PlaneInfo
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Vector3, Quaternion
from cv_bridge import CvBridge
from ultralytics import YOLOWorld as YOLO
from std_srvs.srv import Trigger
import time
import torch
import tf2_ros

# local classes
from propif_perception.roi import Rois
from propif_perception.feat_manager import FeatureManager
from propif_perception.camera_loc_manager import CameraLocManager

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        self.classes = ['flower', 'leaf', 'tree', 'plant', '']
        self.debug_windows = False
        self.use_sfm_reconstruction = False
        self.reconstruction_image_count = 10
        self.frame_id = 'map'
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize detected planes list
        self.detected_planes = []
        
        # Create CvBridge to convert ROS images to OpenCV
        self.bridge = CvBridge()
        
        # Initialize camera parameters
        self.camera_intrinsics = None
        self.rotation_matrix = None
        self.translation_vector = None
        self.cam_loc_manager = None  # Only used if SFM is needed
        
        # Initialize SFM-related state
        self.sfm_initialized = False
        self.collected_images_count = 0
        self.collection_mode = self.use_sfm_reconstruction  # SFM or not
        
        # If using SFM, set up directories for reconstruction
        if self.use_sfm_reconstruction:
            self.setup_sfm_directories()
        
        # Setup TF2 listener for Plan A
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Create subscribers
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        
        # Create publishers
        self.processed_image_pub = self.create_publisher(Image, '/processed_image', 10)
        self.plane_info_pub = self.create_publisher(PlaneInfo, '/detected_planes', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)
        
        # Store latest images
        self.latest_color = None
        self.latest_depth = None
        
        # Initialize YOLO model (or any other model) + feature manager
        self.load_yolo_model()
        self.featMan = FeatureManager(self.device, len(self.classes))
        
        # Create a timer for processing images (10Hz)
        self.timer = self.create_timer(0.1, self.process_images)
        
        # If not using SFM, we rely on direct camera pose (Plan A), so set a separate timer
        if not self.use_sfm_reconstruction:
            self.camera_pose_timer = self.create_timer(0.1, self.update_camera_pose)
        
        # If using SFM, optionally provide a service to toggle collection mode
        if self.use_sfm_reconstruction:
            self.create_service(Trigger, 'toggle_collection_mode', self.toggle_collection_mode_callback)
        
        # Log info
        self.get_logger().info('Perception node initialized')
        if self.use_sfm_reconstruction:
            self.get_logger().info('Using SFM reconstruction mode (Plan B)')
            self.get_logger().info(f'Collect {self.reconstruction_image_count} images for reconstruction')
        else:
            self.get_logger().info('Using direct camera pose mode (Plan A)')
    
    def setup_sfm_directories(self):
        """Setup directories for SFM reconstruction"""
        os.makedirs('.cache/', exist_ok=True)
        os.makedirs('.cache/outputs/', exist_ok=True)
        os.makedirs('.cache/mapping/', exist_ok=True)
        os.makedirs('query/', exist_ok=True)
        
        # Clear any previous mapping images
        for f in Path('.cache/mapping/').glob('*.*'):
            f.unlink()
        
        self.get_logger().info('SFM directories initialized')
    
    def load_yolo_model(self):
        """Initialize YOLO model for object detection"""
        try:
            model = YOLO('yolov8s-worldv2.pt')
            model.set_classes(self.classes)
            model.save("custom_yolov8s.pt")
            self.yolo_model = YOLO('custom_yolov8s.pt')
            self.yolo_model.model.to(self.device)
            self.get_logger().info('YOLO model initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize YOLO model: {str(e)}')
    
    def toggle_collection_mode_callback(self, request, response):
        """Service callback to toggle between collection and detection modes"""
        self.collection_mode = not self.collection_mode
        mode = "collection" if self.collection_mode else "detection"
        self.get_logger().info(f'Switched to {mode} mode')
        
        # If switching to detection mode and we have enough images, perform reconstruction
        if not self.collection_mode and self.collected_images_count >= self.reconstruction_image_count and not self.sfm_initialized:
            self.get_logger().info('Starting 3D reconstruction...')
            try:
                self.cam_loc_manager.reconstruction_3D()
                self.sfm_initialized = True
                self.get_logger().info('3D reconstruction completed successfully')
            except Exception as e:
                self.get_logger().error(f'Failed to perform 3D reconstruction: {str(e)}')
        
        response.success = True
        response.message = f"Switched to {mode} mode. Images: {self.collected_images_count}/{self.reconstruction_image_count}"
        return response
    
    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        if self.camera_intrinsics is None:
            self.get_logger().info('Received camera info')
            
            # Extract camera intrinsics matrix
            fx = msg.k[0]
            fy = msg.k[4]
            cx = msg.k[2]
            cy = msg.k[5]
            self.camera_intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            # For Plan B (SFM), still need CameraLocManager
            if self.use_sfm_reconstruction:
                self.cam_loc_manager = CameraLocManager(
                    msg.height, 
                    msg.width,
                    fx, cx, cy, 0.0
                )
            
            self.get_logger().info('Camera intrinsics initialized')
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to 3x3 rotation matrix"""
        # Extract quaternion components
        x, y, z, w = q.x, q.y, q.z, q.w
        
        # Compute rotation matrix elements
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        
        yy = y * y
        yz = y * z
        yw = y * w
        
        zz = z * z
        zw = z * w
        
        # Form the rotation matrix
        rot_matrix = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
        ])
        
        return rot_matrix
    
    def compute_orientation_from_normal(self, normal):
        """Compute quaternion orientation from normal vector"""
        # Normalize the vector
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            # If normal is too small, return identity quaternion
            return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
            
        normal = normal / norm
        
        # Find rotation from [0,0,1] to normal
        z_axis = np.array([0, 0, 1])
        
        # Handle special case when normal is parallel to z-axis
        if np.abs(np.dot(normal, z_axis) - 1.0) < 1e-6:
            return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        elif np.abs(np.dot(normal, z_axis) + 1.0) < 1e-6:
            return Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)
        
        # Compute the rotation axis (cross product)
        axis = np.cross(z_axis, normal)
        axis = axis / np.linalg.norm(axis)
        
        # Compute the rotation angle
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
        
        # Convert axis-angle to quaternion
        sin_half_angle = np.sin(angle/2)
        cos_half_angle = np.cos(angle/2)
        
        quat = Quaternion()
        quat.x = axis[0] * sin_half_angle
        quat.y = axis[1] * sin_half_angle
        quat.z = axis[2] * sin_half_angle
        quat.w = cos_half_angle
        
        return quat
    
    def update_camera_pose(self):
        """Query camera pose from TF system (Plan A only)"""
        if not self.use_sfm_reconstruction:
            try:
                # Query latest camera pose from TF system
                transform = self.tf_buffer.lookup_transform(
                    'world',
                    'camera_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                # Extract rotation and translation
                q = transform.transform.rotation
                t = transform.transform.translation
                
                # Convert quaternion to rotation matrix
                self.rotation_matrix = self.quaternion_to_rotation_matrix(q)
                self.translation_vector = np.array([t.x, t.y, t.z])
                
                # Log success occasionally to avoid spam
                if not hasattr(self, '_last_tf_success') or time.time() - self._last_tf_success > 10.0:
                    self.get_logger().info('Camera pose updated successfully')
                    self._last_tf_success = time.time()
                    
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException, tf2_ros.TimeoutException) as e:
                # Error logging
                if not hasattr(self, '_last_tf_error') or time.time() - self._last_tf_error > 5.0:
                    self.get_logger().warning(f'Failed to get camera pose: {str(e)}')
                    self._last_tf_error = time.time()

    def color_callback(self, msg):
        """Process incoming color images"""
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # If in collection mode for SFM, save images for mapping
            if self.use_sfm_reconstruction and self.collection_mode and self.collected_images_count < self.reconstruction_image_count:
                if self.latest_color is not None:
                    # Save image for SFM mapping
                    timestamp = int(time.time() * 1000)
                    img_path = f'.cache/mapping/image_{self.collected_images_count:03d}_{timestamp}.jpg'
                    cv2.imwrite(img_path, self.latest_color)
                    self.collected_images_count += 1
                    self.get_logger().info(f'Saved mapping image {self.collected_images_count}/{self.reconstruction_image_count}')
                    
                    # If we've collected enough images, switch to detection mode
                    if self.collected_images_count >= self.reconstruction_image_count:
                        self.get_logger().info('Collected enough images for reconstruction. Starting 3D reconstruction...')
                        try:
                            self.cam_loc_manager.reconstruction_3D()
                            self.sfm_initialized = True
                            self.collection_mode = False
                            self.get_logger().info('3D reconstruction completed successfully. Switched to detection mode.')
                        except Exception as e:
                            self.get_logger().error(f'Failed to perform 3D reconstruction: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Failed to process color image: {str(e)}')
    
    def depth_callback(self, msg):
        """Process incoming depth images"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f'Failed to process depth image: {str(e)}')
    
    def get_camera_pose(self):
        """Get current camera pose based on mode (Plan A or B)"""
        if self.use_sfm_reconstruction:
            # For Plan B: Use SFM to get camera pose
            if not self.sfm_initialized:
                return None, None
            
            # Save current image as query for localization
            if self.latest_color is not None:
                cv2.imwrite('query/query.png', self.latest_color)
                
            # Get camera location from SFM
            try:
                cam_loc = self.cam_loc_manager.get_cam_loc()
                return cam_loc.Rotation_matrix, cam_loc.Translation_vector
            except Exception as e:
                self.get_logger().error(f'Error getting camera pose from SFM: {str(e)}')
                return None, None
        else:
            # For Plan A: Use TF/robot state to get camera pose
            return self.rotation_matrix, self.translation_vector
    
    def process_images(self):
        """Main image processing loop"""
        # Skip processing if we're still collecting images or don't have necessary data
        if (self.collection_mode or 
            self.latest_color is None or 
            self.latest_depth is None or 
            self.camera_intrinsics is None):
            self.get_logger().error(f'There is missing data, skipping processing')
            return
        
        # Get camera pose based on current mode
        rotation_matrix, translation_vector = self.get_camera_pose()
        if rotation_matrix is None or translation_vector is None:
            self.get_logger().error(f'There was an error getting camera pose, skipping processing')
            return
            
        try:
            color_image = self.latest_color.copy()
            depth_image = self.latest_depth.copy()
            
            #! Debug: Show camera feed
            cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Camera Feed', 640, 480)
            cv2.imshow('Camera Feed', color_image)
            # Depth
            if self.latest_depth is not None:
                depth_display = cv2.normalize(depth_image, None, 0, 255, 
                                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                
                cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Depth Map', 640, 480)
                cv2.imshow('Depth Map', depth_colormap)
            
            # YOLO object detection
            results = self.yolo_model.predict(color_image, conf=0.1, iou=0.3, max_det=100, agnostic_nms=True)
            
            for result in results:
                # self.get_logger().info(f'Detection classes: {[self.classes[int(cls)] for cls in result.boxes.cls]}')
                processed_img = result.plot()
                rois = self.extract_rois(color_image, result.boxes)
                
                self.processed_image_pub.publish(self.bridge.cv2_to_imgmsg(processed_img, "bgr8"))
                
                cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('YOLO Detection', 640, 480)
                cv2.imshow('YOLO Detection', processed_img)
                
                plane_info_list = self.featMan.process_new_image(
                    color_image, depth_image, rois, self.classes,
                    rotation_matrix, translation_vector, self.camera_intrinsics,
                    self.debug_windows
                )

                # print(f'Plane info list: {plane_info_list} !!!!')
                
                # Publish plane info and visualization markers
                if plane_info_list:
                    self.publish_plane_info(plane_info_list)
                    self.publish_visualization_markers(plane_info_list)
                    
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc to close debug windows
                self.get_logger().info('User pressed Esc, closing debug windows')
                cv2.destroyAllWindows()
                
        except Exception as e:
            self.get_logger().error(f'Error processing images: {str(e)}')
    
    def extract_rois(self, image, boxes):
        """Extract regions of interest from detection boxes"""
        images = []
        x1, y1, x2, y2, cls = [], [], [], [], []
        
        for i, box in enumerate(boxes.xyxy):
            x1_val, y1_val, x2_val, y2_val = map(int, box)
            x1.append(x1_val)
            y1.append(y1_val)
            x2.append(x2_val)
            y2.append(y2_val)
            images.append(image[y1_val:y2_val, x1_val:x2_val])
            cls.append(boxes.cls[i])
        
        return Rois(images, x1, y1, x2, y2, cls)
    
    def publish_plane_info(self, plane_info_list):
        """Publish detected plane information"""
        for plane in plane_info_list:
            msg = PlaneInfo()
            msg.object_idx = plane.object_idx
            
            # Set normal vector
            msg.normal = Vector3()
            msg.normal.x = float(plane.normal[0])
            msg.normal.y = float(plane.normal[1])
            msg.normal.z = float(plane.normal[2])
            
            # Set centroid point
            msg.centroid = Point()
            msg.centroid.x = float(plane.centroid[0])
            msg.centroid.y = float(plane.centroid[1])
            msg.centroid.z = float(plane.centroid[2])
            
            # # Debug: Add 3d points
            # if hasattr(plane, 'threed_object') and plane.threed_object is not None:
            #     # points cloud (world frame)
            #     points = plane.threed_object.get_point_cloud_points()
                
            #     # add limits, only send 300 points
            #     max_points = min(300, len(points))
            #     step = max(1, len(points) // max_points)
                
            #     for i in range(0, len(points), step):
            #         point = Point()
            #         point.x = float(points[i][0])
            #         point.y = float(points[i][1])
            #         point.z = float(points[i][2])
            #         msg.point_cloud.append(point)
                
            #     self.get_logger().info(f"Added {len(msg.point_cloud)} points to PlaneInfo message")
            
            # Publish message
            self.plane_info_pub.publish(msg)
            self.get_logger().debug(f'Published plane info for object {plane.object_idx}')
    
    def publish_visualization_markers(self, plane_info_list):
        """Publish visualization markers for RViz"""
        marker_array = MarkerArray()
        
        for _, plane in enumerate(plane_info_list):
            # Create centroid marker (sphere)
            centroid_marker = Marker()
            centroid_marker.header.frame_id = self.frame_id
            centroid_marker.header.stamp = self.get_clock().now().to_msg()
            centroid_marker.ns = "plane_centroids"
            centroid_marker.id = plane.object_idx
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD
            
            centroid_marker.pose.position.x = float(plane.centroid[0])
            centroid_marker.pose.position.y = float(plane.centroid[1])
            centroid_marker.pose.position.z = float(plane.centroid[2])
            
            centroid_marker.scale.x = 0.05
            centroid_marker.scale.y = 0.05
            centroid_marker.scale.z = 0.05
            
            centroid_marker.color.r = 1.0
            centroid_marker.color.g = 0.0
            centroid_marker.color.b = 0.0
            centroid_marker.color.a = 1.0
            
            # Create normal vector marker (arrow)
            normal_marker = Marker()
            normal_marker.header.frame_id = self.frame_id
            normal_marker.header.stamp = self.get_clock().now().to_msg()
            normal_marker.ns = "plane_normals"
            normal_marker.id = plane.object_idx
            normal_marker.type = Marker.ARROW
            normal_marker.action = Marker.ADD
            
            normal_marker.pose.position.x = float(plane.centroid[0])
            normal_marker.pose.position.y = float(plane.centroid[1])
            normal_marker.pose.position.z = float(plane.centroid[2])
            
            # Set orientation based on normal vector
            orientation = self.compute_orientation_from_normal(plane.normal)
            normal_marker.pose.orientation = orientation
            
            normal_marker.scale.x = 0.2  # arrow length
            normal_marker.scale.y = 0.02  # arrow width
            normal_marker.scale.z = 0.02  # arrow height
            
            normal_marker.color.g = 1.0
            normal_marker.color.a = 1.0
            
            marker_array.markers.append(centroid_marker)
            marker_array.markers.append(normal_marker)
        
        self.markers_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()