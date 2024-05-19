import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import numpy as np
import pymc as pm


# this class contains all the potential objects that has been extracted so far from the image
class PotentialObjects:
    def __init__(self, classes_number):
        # here i have all the list of the existing keypoints and descriptors
        self.classes_number = classes_number
        self.existing_keypoints = []
        self.existing_descriptors = []
        # object_id is the list of the object id (the keypoint that belongs to the same object has the same id)
        # keypoints that sit in the same roi the first time are assigned to the same new object id
        # once we are s
        self.object_id = []
        
        #self.alpha = np.ones(classes_number)
        #self.p_label = pm.Dirichlet('class_probs', a=np.ones(classes_number))  # Uniform prior over the simplex
        self.alpha = []
        self.p_label = [] 
       
    def add_new_object(self, keypoints, descriptors):
        # Store new keypoints and descriptors
        self.existing_keypoints.append(keypoints)
        self.existing_descriptors.append(descriptors)
        # assign a new object id to the new object
        self.object_id.append(len(self.object_id))
        self.alpha.append(np.ones(self.classes_number))
        self.p_label.append(pm.Dirichlet('class_probs', a=np.ones(self.classes_number)))  # Uniform prior over the simplex

    def update_model(self,one_hot_label, object_id):
        """
        Update the model with a new observation and sample from the posterior.
        
        Parameters:
        - one_hot_label: One-hot encoded label, e.g., [0, 1, 0]
        
        Returns:
        """
        # Update the prior parameters
        self.alpha[object_id,:] += one_hot_label  # Update alpha directly here

        # Reset model's parameter 'a' with updated alpha
        self.p_label[object_id]['class_probs'].distribution.a.set_value(self.alpha)
        
    def evaluate_model(self,object_id):
         # Sampling using NUTS via JAX with numpyro backend
        trace = pm.sampling_jax.sample_numpyro_nuts(model=self.p_label[object_id], draws=500, tune=200, target_accept=0.9, random_seed=42)
    
        return trace
       
# once the object has been fully identified its point are stored using this class which contains the 3d point cloud of the object and the object center
# the object center is the center of the object in the image and other information such as preshape position and oriention to interact with the object    
class threeDObject:
    # this class need to contains the 3d point cloud 
    pass

class FeatureManager:
    def __init__(self, device, classes_number):
        # Initialize the device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize the feature extractor and matcher
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        self.objects=PotentialObjects(classes_number)

    def process_new_image(self, image, rois):
        # Create an image where only the rois are visible
        rois_image = self.create_masked_image(image, rois)
        
        # Extract features from the masked image
        feats = self.extract_features(rois_image)
        
        if self.existing_descriptors:  # Only compare if there are existing features
            if not self.is_matching_existing_features(feats):
                self.store_new_features(feats)
                print("New features stored from ROIs.")
            else:
                print("Existing features from ROIs matched.")
        else:
            self.store_new_features(feats)  # Store features if no existing features are present
            print("Initial features from ROIs stored.")

    def create_masked_image(self, image, rois):
        # Start with a black image of the same size as the original
        masked_image = np.zeros_like(image)
        
        # Fill in the regions defined by ROIs from the Roi object
        for img, x1, y1, x2, y2 in zip(rois.images, rois.x1, rois.y1, rois.x2, rois.y2):
            masked_image[y1:y2, x1:x2] = img

        return masked_image

    def extract_features(self, image):
        # Prepare the image and extract features
        tensor_image = torch.tensor(image).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.
        return self.extractor.extract(tensor_image)

    def is_matching_existing_features(self, new_feats):
        # Prepare for matching
        new_kpts, new_desc = new_feats['keypoints'], new_feats['descriptors']

        for idx, existing_desc in enumerate(self.existing_descriptors):
            # Match against each existing set of descriptors
            matches = self.matcher({
                "image0": {'keypoints': new_kpts, 'descriptors': new_desc},
                "image1": {'keypoints': self.existing_keypoints[idx], 'descriptors': existing_desc}
            })
            matches = rbd(matches)  # Assume 'rbd' function to remove batch dimension exists

            if matches['matches'].size(0) > 0:
                return True  # Match found

        return False  # No matches found

    def store_new_features(self, feats):
        # Store new keypoints and descriptors
        self.existing_keypoints.append(feats['keypoints'])
        self.existing_descriptors.append(feats['descriptors'])

# Helper function to remove batch dimensions (assuming its implementation is provided)
def rbd(features):
    return {k: v.squeeze(0) for k, v in features.items()}