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

@dataclass
class ImageData:
    rgb_image: np.ndarray
    depth_image: np.ndarray
    timestamp: float  # Assuming timestamp is required



class Pan3D:
    def __init__(self,classes,max_length=1000):
        rospy.init_node('ThreeDPan', anonymous=True)
        
        # index to check if the method is advancing
        self.index_debug=0

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
        self.model = None
        self.loadingYoloWorldModelWithClasses()

        # intialize the cv window:
        cv2.namedWindow("Processed Images", cv2.WINDOW_NORMAL)


        # Initialize Pygame for keyboard handling
        #pygame.init()
        #pygame.display.set_mode((100, 100))

        rospy.loginfo("Keyboard driver initialized. Press 'a' to start accumulating, 'z' to stop accumulating, 'd' to save data. ")
    
    def loadingYoloWorldModelWithClasses(self):
        

        # Initialize a YOLO-World model
        model = YOLO('yolov8s-world.pt')  # or select yolov8m/l-world.pt

        # Define custom classes
        model.set_classes(self.classes)

        # Save the model with the defined offline vocabulary
        model.save("custom_yolov8s.pt")

        # Load the model with the custom classes
        self.model = YOLO('custom_yolov8s.pt')


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
                processed_image = self.post_process_image(image_data.rgb_image)
                cv2.imshow("Processed Images", processed_image)
                # Wait for 1 ms and break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    def post_process_image(self, image):
        # Example image processing; for now, just return the same image
        # Add actual image processing logic here
        # Show results
        results=self.model.predict(image)
        image = results[0].plot()
        self.index_debug+=1
        print(self.index_debug)
        #print(image)  
        return image

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