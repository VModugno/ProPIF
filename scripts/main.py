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
import torch
import time  # Import the time module
# local classes
import feat_manager as fm
import roi as roi




torch.set_grad_enabled(False)


# define the algorithm:

# 1. zero 3d points no images
# 2. get the image and the depth image
# 3. extract the mask for the images and the four bounding box points in 3d (each mask account for a different object in the scene)
# 4. extract point descriptors for each mask
# 4. using the depth map and the camera position and orientation i check if i have an object already instatiated in the proximity of the 3d world
# 5. in order to check if an object is already there look in my object kdtree (one situationt that could happen is that i have to BB that are one in another so i could have more that one object to test). (here i assume that the object is  a collection of 3d points and 2d features associated to them) 
# 5.1. if the object is there (we could use the object mask boundary to create a 3d bounding box and see if anything is there) i check if the 2d descriptor of the point i collected before match with the one in the mask. any new features point not mathching i add to the object structure with the label 
#      any descriptor matching i update the label count associated to each point. a good indication that two object are different is that points are in different class. if two object are in the same class 
#      they will belong to different mask which account for the instance segmentation
# 5.2. if the object is not there (we could use object boundary to create a 3d bounding box and see if anything is there) i create a new object and add the 3d points and the 2d features to the object structure  
# 6 i keep update the object and i keep track of the current object class by doing a majority voting on the label associated to each point in the object
# 7.1 from the label i can easily distiguish between stuff and object. is a label is an object i can epxect that at some i will fully cover the 3d point and ideally is hould walk al around it
# 7.2 if the label is stuff i could keep adding point to the object for a long time without fully covering it. In case of stuff i'm more interested in his 3d boundary rather rather than the point inside it 
#     (but we can look into that later and for now focus on objects and not use any stuff label)
# 8. once an object is fully covered i can get a couple of take to densify it and get a better 3d representation of it using the current mask and the depth map and a 3d bounding box to get a full coverage of it
# at the end of this i have the 3d representation of the object in my current scene with semantic information associated to it.
# moreover by exploiting the semantic meaning of the class i can easily understand if an object is a part of another object or if they are two different object (like flower > plant > tree)



@dataclass
class ImageData:
    rgb_image: np.ndarray
    depth_image: np.ndarray
    timestamp: float  # Assuming timestamp is required

#

class Pan3D:
    def __init__(self,classes,max_length=1000,video_input=False, video_path=None, start_minute=0):
        rospy.init_node('ThreeDPan', anonymous=True)
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
                rospy.logerr("Error opening video file.")
                return
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            rospy.loginfo(f"Video file opened: {self.video_path} ({self.video_length} frames, {self.video_fps} fps)")
            start_frame = start_minute * 60 * self.video_fps  # 60 seconds per minute
            # Set the current frame to the frame at start_minute
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # Initialize publisher
        self.processed_image_yolow_pub = rospy.Publisher("/processed_image_yolow", Image, queue_size=10)
        self.processed_image_fastsam_pub = rospy.Publisher("/processed_image_fastSAM", Image, queue_size=10)
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
        self.LoadingYoloWorldModelWithClasses()
        self.LoadingFastSamModel()

    
        # Initialize Pygame for keyboard handling
        #pygame.init()
        #pygame.display.set_mode((100, 100))

        #rospy.loginfo("Keyboard driver initialized. Press 'a' to start accumulating, 'z' to stop accumulating, 'd' to save data. ")
    
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

            # here i create the image from the video file 
            if self.video_input:
                # Read the next frame from the video file
                ret, frame = self.cap.read()
                if not ret:
                    rospy.loginfo("End of video file reached.")
                    break
                color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                depth_image = np.zeros_like(color_image)
                image_data = ImageData(rgb_image=color_image, depth_image=depth_image, timestamp=time.time())
                self.images.append(image_data)

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
        
        all_masks = []
        for roi_img in rois.images:
            cur_result = self.fast_sam_model.predict(roi_img, retina_masks=True)
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