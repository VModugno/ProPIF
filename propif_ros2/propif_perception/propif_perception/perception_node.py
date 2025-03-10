#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import os
from pathlib import Path
from sensor_msgs.msg import Image, CameraInfo, JointState
from propif_msgs.msg import PlaneInfo
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Vector3, Pose
from cv_bridge import CvBridge
from ultralytics import YOLOWorld as YOLO
from std_srvs.srv import Trigger
import time
import torch

# local classes
from roi import Rois
from feat_manager import FeatureManager
from camera_loc_manager import CameraLocManager

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        # Declare parameters
        self.declare_parameter('classes', ['flower', 'leaf', 'tree', 'plant'])
        self.declare_parameter('debug_windows', False)
        self.declare_parameter('use_sfm_reconstruction', False)  # Plan B: Use SFM reconstruction instead of direct pose
        self.declare_parameter('reconstruction_image_count', 10)  # Number of images to collect for reconstruction
        
        # Get parameters
        self.classes = self.get_parameter('classes').value
        self.debug_windows = self.get_parameter('debug_windows').value
        self.use_sfm_reconstruction = self.get_parameter('use_sfm_reconstruction').value
        self.reconstruction_image_count = self.get_parameter('reconstruction_image_count').value

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize detected planes list
        self.detected_planes = []
        
        # Create CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Initialize camera location manager and state variables
        self.cam_loc_manager = None
        self.sfm_initialized = False
        self.collected_images_count = 0
        self.collection_mode = self.use_sfm_reconstruction  # Start in collection mode if using SFM
        
        # Create directories for SFM reconstruction if needed
        if self.use_sfm_reconstruction:
            self.setup_sfm_directories()
        
        # Create subscribers
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        
        # For Plan A: Subscribe to robot joint states to get end effector pose
        if not self.use_sfm_reconstruction:
            self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Create publishers
        self.processed_image_pub = self.create_publisher(Image, '/processed_image', 10)
        self.plane_info_pub = self.create_publisher(PlaneInfo, '/detected_planes', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)
        
        # Store latest images
        self.latest_color = None
        self.latest_depth = None
        self.camera_pose = None  # For Plan A
        
        # Initialize YOLO model and feature manager
        self.load_yolo_model()
        self.featMan = FeatureManager(self.device, len(self.classes))
        
        # Create timer for processing images
        self.timer = self.create_timer(0.1, self.process_images)
        
        # Service to toggle collection mode for SFM
        if self.use_sfm_reconstruction:
            self.create_service(Trigger, 'toggle_collection_mode', self.toggle_collection_mode_callback)
        
        self.get_logger().info('Perception node initialized')
        if self.use_sfm_reconstruction:
            self.get_logger().info('Using SFM reconstruction mode (Plan B)')
            self.get_logger().info(f'Collect {self.reconstruction_image_count} images for reconstruction')
        else:
            self.get_logger().info('Using direct camera pose mode (Plan A)')
        
    def setup_sfm_directories(self):
        """Setup directories for SFM reconstruction"""
        # Create cache directories for SFM
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
        response.message = f"Switched to {mode} mode"
        return response
    
    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        if self.cam_loc_manager is None:
            self.get_logger().info('Received camera info')
            self.cam_loc_manager = CameraLocManager(
                msg.height, 
                msg.width,
                msg.k[0],  # fx
                msg.k[2],  # cx
                msg.k[5],  # cy
                0.0  # k
            )
            
            # Only perform reconstruction_3D for Plan A
            if not self.use_sfm_reconstruction:
                self.cam_loc_manager.reconstruction_3D()
                self.sfm_initialized = True
                
            self.get_logger().info('Camera location manager initialized')
    
    def joint_state_callback(self, msg):
        """Process joint states to get end effector pose (Plan A)"""
        # This would typically come from forward kinematics or a tf lookup
        # Placeholder - in a real system you'd compute the camera pose from joint states
        self.camera_pose = Pose()  # This would be populated with actual pose data
    
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
    
    def process_images(self):
        """Main image processing loop"""
        # Skip processing if we're still collecting images or don't have necessary data
        if self.collection_mode or self.latest_color is None or self.latest_depth is None or self.cam_loc_manager is None:
            return
        
        # For Plan B, we need SFM to be initialized
        if self.use_sfm_reconstruction and not self.sfm_initialized:
            return
            
        try:
            # Make copies to avoid concurrency issues
            color_image = self.latest_color.copy()
            depth_image = self.latest_depth.copy()
            
            # For Plan B: Get camera pose using SFM
            if self.use_sfm_reconstruction:
                # Save current image as query for localization
                cv2.imwrite('query/query.png', color_image)
                camera_loc = self.cam_loc_manager.get_cam_loc()
                # Now we have camera_loc.Rotation_matrix and camera_loc.Translation_vector
                
            # Process with YOLO
            results = self.yolo_model.predict(color_image, conf=0.1, iou=0.3, max_det=100)
            
            for result in results:
                processed_img = result.plot()
                rois = self.extract_rois(color_image, result.boxes)
                
                # Process detected ROIs
                plane_info_list = self.featMan.process_new_image(
                    color_image, 
                    depth_image, 
                    rois,
                    self.classes,
                    self.cam_loc_manager,
                    self.debug_windows
                )
                
                # Publish results
                if plane_info_list:
                    self.publish_plane_info(plane_info_list)
                    self.publish_visualization_markers(plane_info_list)
                
                # Publish processed image
                self.processed_image_pub.publish(self.bridge.cv2_to_imgmsg(processed_img, "bgr8"))
                
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
            
            # Publish message
            self.plane_info_pub.publish(msg)
            self.get_logger().debug(f'Published plane info for object {plane.object_idx}')
    
    def publish_visualization_markers(self, plane_info_list):
        """Publish visualization markers for RViz"""
        marker_array = MarkerArray()
        
        for i, plane in enumerate(plane_info_list):
            # Create centroid marker (sphere)
            centroid_marker = Marker()
            centroid_marker.header.frame_id = "camera_link"
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
            normal_marker.header.frame_id = "camera_link"
            normal_marker.header.stamp = self.get_clock().now().to_msg()
            normal_marker.ns = "plane_normals"
            normal_marker.id = plane.object_idx
            normal_marker.type = Marker.ARROW
            normal_marker.action = Marker.ADD
            
            normal_marker.pose.position.x = float(plane.centroid[0])
            normal_marker.pose.position.y = float(plane.centroid[1])
            normal_marker.pose.position.z = float(plane.centroid[2])
            
            # Set orientation based on normal vector
            normal_marker.pose.orientation.w = 1.0
            
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