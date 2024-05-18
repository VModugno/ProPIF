#merge roi class with the mask object or at least add all the data in ROI to process the mask 
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
        masks list(): associated to each roi
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
        # here i want to store the masks associated to each roi 
        self.masks = []
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    # in order to guarantee the correct mathching between rois and mask that should be called in order to associate the mask to the correct roi
    def add_mask(self, mask):
        """
        Add a mask to the list of masks associated with the ROI.
        
        Args:
            mask: The mask to add to the list.
        """
        self.masks.append(mask)


    def apply_roi_masks_to_original(self,image):
        """
        Apply an ROI mask back to the original image at the specified bounding box location.

        Args:
            image (numpy.ndarray): The original image.
            roi_mask (numpy.ndarray): The mask obtained from the ROI.
            bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) from which the ROI was extracted.

        Returns:
            numpy.ndarray: The original image with the ROI mask applied.
        """
        h, w = image.shape[:2]  # Height and width of the original image

        # Create a full-size mask that matches the original image dimensions
        full_mask = np.zeros((h, w), dtype=np.uint8)

        # Ensure the ROI mask fits into the full mask at the specified coordinates
        for i in len(self.images):
            full_mask[self.y1[i]:self.y2[i], self.x1[i]:self.x2[i]] = self.masks[i]

        # Apply the mask to the original image
        # For visualization, you can color the mask region - here we simply highlight it
        masked_image = image.copy()
        masked_image[full_mask > 0] = (255, 0, 0)  # Example: paint the mask region blue

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