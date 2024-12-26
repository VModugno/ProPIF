# Description: This script contains the class Rois that is used to handle regions of interest
#  (ROIs) in an image, including methods to map detections within the ROI back to the original
#  full image.
# every time we get a new image the ROIs class is fully recomputed and the masks associated 
# to each roi are stored in the class
import numpy as np

class Rois:
    """
    A class to handle regions of interest (ROIs) in an image, including methods to map detections within the ROI
    back to the original full image.

    Attributes:
        images list(numpy.ndarray): The ROI image extracted from the original image.
        x1 (int): The x-coordinate of the top-left corner of the ROI in the original image.
        y1 (int): The y-coordinate of the top-left corner of the ROI in the original image.
        x2 (int): The x-coordinate of the bottom-right corner of the ROI in the original image.
        y2 (int): The y-coordinate of the bottom-right corner of the ROI in the original image.
        cx (int): The x-coordinate of the center of the ROI in the original image.
        cy (int): The y-coordinate of the center of the ROI in the original image.
        masks list(): associated to each roi
        class list(): associated to each roi
    """
    
    def __init__(self, images:list, x1:list, y1:list, x2:list, y2:list, classes:list):
        """
        Initialize the Roi object with the all the ROI images and its coordinates in the original image.
        
        Args:
            image (numpy.ndarray): The ROI image.
            x1, y1 (int): Coordinates of the top-left corner of the ROI in the original image.
            x2, y2 (int): Coordinates of the bottom-right corner of the ROI in the original image.
        """
        self.images = images
        # here i want to store the masks associated to each roi 
        self.masks = [] # this are the mask from SAM, (roi_index, mask)
        self.features = [] # this are the features from lightglue
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cx = [(x1_i + x2_i) // 2 for x1_i, x2_i in zip(x1, x2)]
        self.cy = [(y1_i + y2_i) // 2 for y1_i, y2_i in zip(y1, y2)]
        self.classes = classes

    # in order to guarantee the correct mathching between rois and mask that should be called in order to associate the mask to the correct roi
    def add_mask(self, index, mask):
        """
        Add a mask to the list of masks associated with the ROI.
        
        Args:
            mask: The mask to add to the list.
        """
        self.masks.append((index, mask))

    def add_features(self, features):
        """
        Add a features to the list associated with the ROI.
        
        Args:
            features: The features from lightglue to add to the list.
        """
        self.features.append(features)


    def apply_roi_masks_to_original(self, image):
        """
        Apply ROI masks back to the original image at the specified bounding box locations.

        Args:
            image (numpy.ndarray): The original image.

        Returns:
            numpy.ndarray: The original image with the ROI masks applied.
        """
        h, w = image.shape[:2]

        full_mask = np.zeros((h, w), dtype=np.uint8)

        for index, mask_array in self.masks:
            print("Applying mask to ROI", index)
            x1_i, y1_i = self.x1[index], self.y1[index]
            x2_i, y2_i = self.x2[index], self.y2[index]
            roi_area = full_mask[y1_i:y2_i, x1_i:x2_i]
            full_mask[y1_i:y2_i, x1_i:x2_i] = np.logical_or(roi_area, mask_array).astype(np.uint8)
        masked_image = image.copy()
        masked_image[full_mask > 0] = (255, 0, 0)

        return masked_image

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