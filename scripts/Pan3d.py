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
from ultralytics import YOLO
from ultralytics import FastSAM
import torch
import time  # Import the time module
# local classes

from roi import Rois as roi
from feat_manager import FeatureManager

torch.set_grad_enabled(False)

DEBUGWINDOWSVIDEO = True

@dataclass
class ImageData:
    rgb_image: np.ndarray
    depth_image: np.ndarray
    timestamp: float  # Assuming timestamp is required

class Pan3D:
    def __init__(self,classes,max_length=1000,video_input=False, video_path=None, start_minute=10, device="cuda"):
        # rospy.init_node('ThreeDPan', anonymous=True)
        self.video_input = video_input
        self.running = True
        # Create CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Current state storage
        self.current_color_image = None
        self.current_depth_image = None
        
        # List to hold trajectory data and images
        self.rgb_image_stack = []
        self.depth_image_stack = []
        
        # Initialize subscribers
        if not self.video_input:
            self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
            self.depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
            self.ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
            self.ts.registerCallback(self.image_callback)
            # here i open the video file
        else:
            # Open the video file
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                #rospy.logerr("Error opening video file. Stopping object initialization.")
                raise ValueError("Error opening video file. Stopping object initialization.")
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # rospy.loginfo(f"Video file opened: {self.video_path} ({self.video_length} frames, {self.video_fps} fps)")
            start_frame = start_minute * 60 * self.video_fps  # 60 seconds per minute
            # Set the current frame to the frame at start_minute
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        # Initialize publisher
        # self.processed_image_yolow_pub = rospy.Publisher("/processed_image_yolow", Image, queue_size=10)
        # self.processed_image_fastsam_pub = rospy.Publisher("/processed_image_fastSAM", Image, queue_size=10)
        # Synchronize the subscribers by time
    

        # Initialize keyboard handler thread
        #self.keyboard_thread = threading.Thread(target=self.keyboard_handler)
        self.images = deque(maxlen=max_length)
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True  # Ensure the thread exits when the main program does

        # Initialize the YOLO model
        self.classes = classes
        self.yolo_model = None
        self.fast_sam_model = None

        # check if the device is available
        if device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine.")
    
        self.device = device
        self.LoadingYoloWorldModelWithClasses(self.device)
        self.LoadingFastSamModel(self.device)
        
        self.featMan=FeatureManager(self.device, len(classes))

    
        # Initialize Pygame for keyboard handling
        #pygame.init()
        #pygame.display.set_mode((100, 100))

        #rospy.loginfo("Keyboard driver initialized. Press 'a' to start accumulating, 'z' to stop accumulating, 'd' to save data. ")
    
    def LoadingYoloWorldModelWithClasses(self,device="cuda"):
        

        # Initialize a YOLO-World model
        model = YOLO('yolov8s-world.pt')  # or select yolov8m/l-world.pt

        # Define custom classes
        model.set_classes(self.classes)

        # Save the model with the defined offline vocabulary
        model.save("custom_yolov8s.pt")

        # Load the model with the custom classes
        self.yolo_model = YOLO('custom_yolov8s.pt')
        # move the model the the device
        self.yolo_model.model.to(device)

    def LoadingFastSamModel(self,device="cuda"):
        # Create a FastSAM model
        self.fast_sam_model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt
        # move the model the the device
        self.fast_sam_model.model.to(device)


    def image_callback(self, color_msg, depth_msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

            # Processing of synchronized images here
            image_data = ImageData(rgb_image=color_image, depth_image=depth_image, timestamp=time.time())
            self.images.append(image_data)
        except Exception as e:
            print(f"Error processing images: {e}")
            # rospy.logerr(f"Error processing images: {e}")
    '''
    def keyboard_handler(self):
        running = True
        clock = pygame.time.Clock()  # Create a clock object
        while not rospy.is_shutdown() and  self.running:
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

            # here i create the image from the video file 
            if self.video_input:
                # Read the next frame from the video file
                ret, frame = self.cap.read()
                if not ret:
                    # rospy.loginfo("End of video file reached.")
                    print("End of video file reached.")
                    break
                color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                depth_image = np.zeros_like(color_image)
                image_data = ImageData(rgb_image=color_image, depth_image=depth_image, timestamp=time.time())
                cv2.imshow("Video", frame)
                self.images.append(image_data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.images:  # Check if deque is not empty
                # Pop the oldest image data from the deque
                image_data = self.images.popleft()

                start_time = time.time()
                image_yolow, image_fast_sam = self.post_process_image(image_data.rgb_image)
                image_yolow_rgb = cv2.cvtColor(image_yolow, cv2.COLOR_BGR2RGB)
                image_fast_sam_rgb = cv2.cvtColor(image_fast_sam, cv2.COLOR_BGR2RGB)
                # cv2.imshow("Video_sam", image_fast_sam_rgb)
                # cv2.imshow("Video_yolow", image_yolow_rgb)
                end_time = time.time()
                print(f"Processing time: {end_time - start_time:.2f} seconds")
                if  DEBUGWINDOWSVIDEO and self.video_input:
                    cv2.imshow("Video_sam", image_fast_sam_rgb)
                    cv2.imshow("Video_yolow", image_yolow_rgb)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                try:
                    ros_image_yolow = self.bridge.cv2_to_imgmsg(image_yolow, "bgr8")
                    ros_image_fast_sam = self.bridge.cv2_to_imgmsg(image_fast_sam, "bgr8")
                    self.processed_image_yolow_pub.publish(ros_image_yolow)
                    self.processed_image_fastsam_pub.publish(ros_image_fast_sam)
                except Exception as e:
                    # rospy.logerr("Failed to convert or publish image: %s", e)
                    print("Failed to convert or publish image: %s", e)

    def post_process_image(self, image):
        # Example image processing; for now, just return the same image
        # Add actual image processing logic here
        # Show results
        start_time = time.time()
        results = self.yolo_model.predict(image, conf=0.05, iou=0.4, max_det=50)

        # Plot all YOLO results on the original image
        image_yoloW = image.copy()
        for result in results:
            image_yoloW = result.plot()

        rois = self.extract_rois(image, results[0].boxes)
        # extract all the mask from the results
        # masks = results[0].masks
        # for mask in masks:
        # Run inference on an image
        # TODO this passage can be optimized by resizing the image to all the same size perform the inference and scale the results back to the original size
        # optimization for later or we can parallelize with multiple sam model in GPU
        all_masks = []
        for roi_img in rois.images:
            cur_result = self.fast_sam_model.predict(roi_img, retina_masks=True)
            all_masks.append(cur_result[0].masks)
        
        self.featMan.process_new_image(image, rois, DEBUGWINDOWSVIDEO)
        
        
        end_time = time.time()
        print(f"Processing time inside post_process_image: {end_time - start_time:.2f} seconds")
        # extract mask from results_fs
        
        # Prepare a Prompt Process object
        # everything_results = self.fast_sam_model(image, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        # prompt_process = FastSAMPrompt(image, everything_results, device='gpu')
        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        # ann = prompt_process.box_prompt(bbox=mask)
                  
        image_fastSAM = results[0].plot()
         
        return image_yoloW, image_fastSAM
    
    # TODO this function can be operate in gpu
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
        bounding_boxes = boxes.xyxy.cpu().numpy()
        class_labels = boxes.cls.cpu().numpy()
        for box, cls_cur in zip(bounding_boxes, class_labels):  # Convert to numpy array if not already
            x1_cur, y1_cur, x2_cur, y2_cur = map(int, box)  # Ensure coordinates are integer values
            x1.append(x1_cur)
            y1.append(y1_cur)
            x2.append(x2_cur)
            y2.append(y2_cur)
            images.append(image[y1_cur:y2_cur, x1_cur:x2_cur])
            cls.append(cls_cur)
        rois = roi(images, x1, y1, x2, y2, cls)
        print(f"Extracted {len(rois.images)} ROIs.")
        return rois

    def run(self):
        self.processing_thread.start()
        #self.keyboard_thread.start()

        # Use rospy.spin() to keep your node alive and handle callbacks
        # rospy.spin()
        #pygame.quit()
        #self.keyboard_thread.join()
        self.processing_thread.join()

    def cleanup(self):
        # Clean up and close the window when done
        cv2.destroyAllWindows()