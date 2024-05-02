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
from ultralytics.models.fastsam import FastSAMPrompt
import time  # Import the time module
@dataclass
class ImageData:
    rgb_image: np.ndarray
    depth_image: np.ndarray
    timestamp: float  # Assuming timestamp is required

class Rois:
    """
    A class to handle regions of interest (ROIs) in an image, including methods to map detections within the ROI
    back to the original full image.

    Attributes:
        image (numpy.ndarray): The ROI image extracted from the original image.
        x1 (int): The x-coordinate of the top-left corner of the ROI in the original image.
        y1 (int): The y-coordinate of the top-left corner of the ROI in the original image.
        x2 (int): The x-coordinate of the bottom-right corner of the ROI in the original image.
        y2 (int): The y-coordinate of the bottom-right corner of the ROI in the original image.
    """
    
    def __init__(self, images:list, x1:list, y1:list, x2:list, y2:list):
        """
        Initialize the Roi object with the ROI image and its coordinates in the original image.
        
        Args:
            image (numpy.ndarray): The ROI image.
            x1, y1 (int): Coordinates of the top-left corner of the ROI in the original image.
            x2, y2 (int): Coordinates of the bottom-right corner of the ROI in the original image.
        """
        self.images = images
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def map_point_to_original(self, px, py, roi_index):
        """
        Map a point from ROI coordinates back to the original image coordinates.
        
        Args:
            px, py (int): The coordinates of the point in the ROI.
        
        Returns:
            tuple: The coordinates of the point in the original image.
        """
        original_x = self.x1[roi_index] + px
        original_y = self.y1[roi_index] + py
        return (original_x, original_y)

class Pan3D:
    def __init__(self,classes,max_length=1000):
        rospy.init_node('ThreeDPan', anonymous=True)
        
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
        self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        # Initialize publisher
        self.processed_image_yolow_pub = rospy.Publisher("/processed_image_yolow", Image, queue_size=10)
        self.processed_image_fastsam_pub = rospy.Publisher("/processed_image_fastSAM", Image, queue_size=10)
        # Synchronize the subscribers by time
        self.ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.image_callback)


        # Initialize keyboard handler thread
        #self.keyboard_thread = threading.Thread(target=self.keyboard_handler)

        self.images = deque(maxlen=max_length)
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True  # Ensure the thread exits when the main program does

        # Initialize the YOLO model
        self.classes = classes
        self.yolo_model = None
        self.fast_sam_model = None
        self.LoadingYoloWorldModelWithClasses()
        self.LoadingFastSamModel()

        # Initialize Pygame for keyboard handling
        #pygame.init()
        #pygame.display.set_mode((100, 100))

        rospy.loginfo("Keyboard driver initialized. Press 'a' to start accumulating, 'z' to stop accumulating, 'd' to save data. ")
    
    def LoadingYoloWorldModelWithClasses(self):
        

        # Initialize a YOLO-World model
        model = YOLO('yolov8s-world.pt')  # or select yolov8m/l-world.pt

        # Define custom classes
        model.set_classes(self.classes)

        # Save the model with the defined offline vocabulary
        model.save("custom_yolov8s.pt")

        # Load the model with the custom classes
        self.yolo_model = YOLO('custom_yolov8s.pt')

    def LoadingFastSamModel(self):
        # Create a FastSAM model
        self.fast_sam_model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt



    def image_callback(self, color_msg, depth_msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

            # Processing of synchronized images here
            image_data = ImageData(rgb_image=color_image, depth_image=depth_image, timestamp=rospy.Time.now().to_sec())
            self.images.append(image_data)
        except Exception as e:
            rospy.logerr(f"Error processing images: {e}")
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
        while not rospy.is_shutdown() and self.running:
            if self.images:  # Check if deque is not empty
                # Pop the oldest image data from the deque
                image_data = self.images.popleft()

                start_time = time.time()
                image_yolow, image_fast_sam = self.post_process_image(image_data.rgb_image)
                end_time = time.time()
                print(f"Processing time: {end_time - start_time:.2f} seconds")
                try:
                    ros_image_yolow = self.bridge.cv2_to_imgmsg(image_yolow, "bgr8")
                    ros_image_fast_sam = self.bridge.cv2_to_imgmsg(image_fast_sam, "bgr8")
                    self.processed_image_yolow_pub.publish(ros_image_yolow)
                    self.processed_image_fastsam_pub.publish(ros_image_fast_sam)
                except Exception as e:
                    rospy.logerr("Failed to convert or publish image: %s", e)

    def post_process_image(self, image):
        # Example image processing; for now, just return the same image
        # Add actual image processing logic here
        # Show results
        start_time = time.time()
        results=self.yolo_model.predict(image)
        image_yoloW = results[0].plot()

        rois=self.extract_rois( image, results[0].boxes)
        # extract all the mask from the results
        #masks = results[0].masks
        #for mask in masks:
        # Run inference on an image
        # TODO this passage can be optimized by resizing the image to all the same size perform the inference and scale the results back to the original size
        # optimization for later
        results_fs = []
        for roi in rois.images:
            results_fs.append(self.fast_sam_model.predict(roi, retina_masks=True))

        all_masks = []
        # extract all the mask from the results_fs and 
        for cur_result in results_fs:
            all_masks.append(cur_result[0].masks)
        
        end_time = time.time()
        print(f"Processing time inside post_process_image: {end_time - start_time:.2f} seconds")
        # extract mask from results_fs
        
        # Prepare a Prompt Process object
        #everything_results = self.fast_sam_model(image, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        #prompt_process = FastSAMPrompt(image, everything_results, device='gpu')
        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        #ann = prompt_process.box_prompt(bbox=mask)
                  
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
        for box in boxes.xyxy.cpu().numpy():  # Convert to numpy array if not already
            x1_cur, y1_cur, x2_cur, y2_cur = map(int, box)  # Ensure coordinates are integer values
            x1.append(x1_cur)
            y1.append(y1_cur)
            x2.append(x2_cur)
            y2.append(y2_cur)
            images.append(image[y1_cur:y2_cur, x1_cur:x2_cur])
        rois = Rois(images, x1, y1, x2, y2)
        return rois

    def run(self):
        self.processing_thread.start()
        #self.keyboard_thread.start()

        # Use rospy.spin() to keep your node alive and handle callbacks
        rospy.spin()
        #pygame.quit()
        #self.keyboard_thread.join()
        self.processing_thread.join()

    def cleanup(self):
        # Clean up and close the window when done
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        classes= ["monitor", "laptop", "keyboard", "mouse"]
        threedPan = Pan3D(classes)
        threedPan.run()
        threedPan.cleanup()
    except rospy.ROSInterruptException:
        rospy.loginfo("Image Processor node terminated.")