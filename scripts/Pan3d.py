import rospy
import threading
import numpy as np
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
#import pygame
import queue
import cv2
import message_filters
from dataclasses import dataclass
from collections import deque
from ultralytics import YOLOWorld as YOLO
import torch
import pyrealsense2 as rs
import time
import av  # PyAV for 16-bit depth video
import sys

# local classes
from roi import Rois as roi
from feat_manager import FeatureManager
from camera_loc_manager import CameraLocManager

torch.set_grad_enabled(False)

DEBUGWINDOWSVIDEO = False

@dataclass
class ImageData:
    rgb_image: np.ndarray
    depth_image: np.ndarray
    timestamp: float  # Assuming timestamp is required

class Pan3D:
    def __init__(self, classes, max_length=1000, video_input=False, video_path=None, depth_vid_path=None, start_minute=0, device="cuda"):
        # rospy.init_node('ThreeDPan', anonymous=True)
        self.video_input = video_input
        self.running = True
        # Create CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Initialize camera info
        self.focal_length = 427.03
        self.center_x = 315.78
        self.center_y = 240.92
        self.k = 0.0
        self.cam_width = 640
        self.cam_height = 480
        self.fps = 15
        
        if not self.video_input:
            pipeline = rs.pipeline()
            config = rs.config()
            # Modify camera resolution and fps as needed
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            
            profile = pipeline.start(config)
            color_profile = profile.get_stream(rs.stream.color)
            color_intrin = color_profile.as_video_stream_profile().get_intrinsics()
            
            self.focal_length = (color_intrin.fx + color_intrin.fy) / 2
            self.center_x = color_intrin.ppx
            self.center_y = color_intrin.ppy
            self.k = 0.0
            self.cam_width = color_intrin.width
            self.cam_height = color_intrin.height
        
        # Initialize the camera location manager
        self.cam_loc_manager = CameraLocManager(self.cam_height, self.cam_width, self.focal_length, self.center_x, self.center_y, self.k)
        
        # Initialize subscribers
        if not self.video_input:
            self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
            self.depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
            self.ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
            self.ts.registerCallback(self.image_callback)
        else:
            # Open the video file
            self.video_path = video_path
            self.depth_vid_path = depth_vid_path
            
            # RGB video stream using OpenCV
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError("Error opening video file. Stopping object initialization.")
            
            # Depth video stream using PyAV to preserve 16-bit depth data
            try:
                self.depth_container = av.open(depth_vid_path)
            except av.AVError as e:
                raise ValueError(f"Error opening depth video file: {depth_vid_path}") from e
            self.depth_frames = self.depth_container.decode(video=0)
            
            # HLOC video stream using OpenCV
            self.hloc_cap = cv2.VideoCapture(video_path)
            if not self.hloc_cap.isOpened():
                raise ValueError(f"Error opening hloc video file: {video_path}")

            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = start_minute * 60 * self.video_fps  # 60 seconds per minute
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # End video_input branch
        
        # Initialize publisher
        # self.processed_image_yolow_pub = rospy.Publisher("/processed_image_yolow", Image, queue_size=10)
        # self.processed_image_fastsam_pub = rospy.Publisher("/processed_image_fastSAM", Image, queue_size=10)
        
        # Initialize image deque
        self.images = deque(maxlen=max_length)
        self.hloc_thread = threading.Thread(target=self.Build3DModel)
        self.hloc_thread.daemon = True
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True  # Ensure the thread exits when the main program does

        # Initialize the YOLO model
        self.classes = classes
        self.yolo_model = None

        if device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine.")
    
        self.device = device
        self.LoadingYoloWorldModelWithClasses(self.device)
        
        self.featMan = FeatureManager(self.device, len(classes))
    
        # Initialize Pygame for keyboard handling (if needed)
        # pygame.init()
        # pygame.display.set_mode((100, 100))
        # rospy.loginfo("Keyboard driver initialized. Press 'a' to start accumulating, 'z' to stop accumulating, 'd' to save data.")
    
    def Build3DModel(self):
        print("Building 3D model...")
        if self.video_input:
            image_count = 0
            frame_count = 0
            while image_count < 10:
                ret, frame = self.hloc_cap.read()
                if not ret:
                    print("Video is too short for building 3D model.")
                    break

                if frame_count % 30 == 0:
                    image_path = f'.cache/mapping/reference_{image_count}.png'
                    cv2.imwrite(image_path, frame)
                    image_count += 1
                
                frame_count += 1

            self.hloc_cap.release()
            self.cam_loc_manager.reconstruction_3D()
            print("\033[92m============= Successfully build 3D model =============\033[0m")
            self.featMan.set_model_completed()
        # TODO: handle camera input if necessary
        
    def LoadingYoloWorldModelWithClasses(self, device="cuda"):
        # Initialize a YOLO-World model
        model = YOLO('yolov8s-worldv2.pt')  # or select yolov8m/l-world.pt

        # Define custom classes
        model.set_classes(self.classes)

        # Save the model with the defined offline vocabulary
        model.save("custom_yolov8s.pt")

        # Load the model with the custom classes
        self.yolo_model = YOLO('custom_yolov8s.pt')
        # Move the model to the specified device
        self.yolo_model.model.to(device)

    def image_callback(self, color_msg, depth_msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            # Processing of synchronized images
            image_data = ImageData(rgb_image=color_image, depth_image=depth_image, timestamp=time.time())
            self.images.append(image_data)
        except Exception as e:
            print(f"Error processing images: {e}")
            # rospy.logerr(f"Error processing images: {e}")
    '''
    def keyboard_handler(self):
        running = True
        clock = pygame.time.Clock()  # Create a clock object
        while not rospy.is_shutdown() and self.running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.accumulating = True
                        rospy.loginfo("Started accumulating data.")
                    elif event.key == pygame.K_z:
                        self.accumulating = False
                        rospy.loginfo("Stopped accumulating data.")
                    elif event.key == pygame.K_d:
                        self.SaveData("collected_data")
                        self.ClearData()
                        rospy.loginfo("Data saved.")
                elif event.type == pygame.QUIT:
                    self.running = False

            # This will ensure the loop does not run faster than 100 Hz
            clock.tick(100)  # Maintain 100 Hz frequency
    '''
    def process_images(self):
        # Process images in a loop until ROS shuts down
        while self.running:
            if self.video_input:
                # Read the next RGB frame from the video file
                ret_color, frame = self.cap.read()
                if not ret_color:
                    print("End of video file reached.")
                    break

                # Read the next depth frame using PyAV for 16-bit depth data
                try:
                    depth_frame_av = next(self.depth_frames)
                except StopIteration:
                    print("End of depth video reached.")
                    break
                depth_frame = depth_frame_av.to_ndarray(format="gray16le")
                
                image_data = ImageData(rgb_image=frame, depth_image=depth_frame, timestamp=time.time())
                self.images.append(image_data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.images:  # Check if deque is not empty
                # Pop the oldest image data from the deque
                image_data = self.images.popleft()

                start_time = time.time()
                image_yolow = self.post_process_image(image_data.rgb_image)
                end_time = time.time()
                print(f"Processing time: {end_time - start_time:.2f} seconds")
                if self.video_input:
                    cv2.imshow("Video_yolow", image_yolow)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                try:
                    ros_image_yolow = self.bridge.cv2_to_imgmsg(image_yolow, "bgr8")
                    # ros_image_fast_sam = self.bridge.cv2_to_imgmsg(image_fast_sam, "bgr8")
                    self.processed_image_yolow_pub.publish(ros_image_yolow)
                    # self.processed_image_fastsam_pub.publish(ros_image_fast_sam)
                except Exception as e:
                    # rospy.logerr("Failed to convert or publish image: %s", e)
                    print("Failed to convert or publish image: %s", e)
    
    def post_process_image(self, image):
        """
        Post-process the image using the YOLO model.
        """
        start_time = time.time()
        results = self.yolo_model.predict(image, conf=0.1, iou=0.3, max_det=100, agnostic_nms=True)
        image_yoloW = image.copy()

        for result in results:
            image_yoloW = result.plot()
            rois = self.extract_rois(image, result.boxes)
            self.featMan.process_new_image(image, rois, self.classes, self.cam_loc_manager, DEBUGWINDOWSVIDEO)

        end_time = time.time()
        print(f"Processing time inside post_process_image: {end_time - start_time:.2f} seconds")
        return image_yoloW

    def extract_rois(self, image, boxes):
        """
        Extract regions of interest from the image using absolute box coordinates.
        
        Args:
            image (numpy.ndarray): The original image array.
            boxes (Boxes): The Boxes object containing detection boxes.

        Returns:
            List of ROI object: A list containing ROI objects.
        """
        images = []
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        cls = []
        bounding_boxes = boxes.xyxy
        class_labels = boxes.cls
        for box, cls_cur in zip(bounding_boxes, class_labels):  # Convert to numpy array if not already
            x1_cur, y1_cur, x2_cur, y2_cur = map(int, box)  # Ensure coordinates are integer values
            x1.append(x1_cur)
            y1.append(y1_cur)
            x2.append(x2_cur)
            y2.append(y2_cur)
            images.append(image[y1_cur:y2_cur, x1_cur:x2_cur])
            cls.append(cls_cur)
        rois = roi(images, x1, y1, x2, y2, cls)
        return rois

    def run(self):
        self.processing_thread.start()
        self.hloc_thread.start()
        # self.keyboard_thread.start()
        # Use rospy.spin() to keep your node alive and handle callbacks
        # rospy.spin()
        # pygame.quit()
        # self.keyboard_thread.join()
        self.hloc_thread.join()
        self.processing_thread.join()

    def cleanup(self):
        # Clean up and close the window when done
        cv2.destroyAllWindows()
