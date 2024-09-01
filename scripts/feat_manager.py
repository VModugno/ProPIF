import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import numpy as np
from twoDthreeDObjects import Potential2dObjectsManager, Potential2dObjects


class FeatureManager:
    def __init__(self, device, classes_number):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize the feature extractor and matcher
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
       

        self.objects2dMan=Potential2dObjectsManager()

    def process_new_image(self, image, rois, DEBUGWINDOWVIDEO):
        # Create an image where only the rois are visible
        if len(rois.images) == 0:
            print("No ROIs found.")
            
        else:
            # TODO rather than doing this we process the image one by one and we add the features to the each ROI
            #rois_image = self.create_masked_image(image, rois)
            # Extract features from the masked image
            #feats = self.extract_features(rois_image)
            for idx, roi_image in enumerate(rois.images):
                rois.add_features(self.extractor.extract(torch.tensor(roi_image).to(self.device)))

            if DEBUGWINDOWVIDEO:
                for idx, roi_image in enumerate(rois.images):
                    viz2d.plot_images([roi_image])
                    viz2d.plot_keypoints([rois.features[idx]['keypoints']], ps=10)
                    input("Press Enter to continue...")
            
            # here i check if there are existing descripotrs and keypoints
            if len(self.objects2dMan.get_potential_objects())>0:  # Only compare if there are existing features
                for idx, cur_potential_2dobject in enumerate(self.objects2dMan.get_potential_objects()):
                    # Check if the new features match any existing features
                    self.matching_existing_features(cur_potential_2dobject, rois)
                if len(rois.images)>0:
                    self.store_new_features(rois)
                    print("New features stored from ROIs.")
            else:
                self.store_new_features(rois)  # Store features if no existing features are present
                print("Initial features from ROIs stored.")

    def create_masked_image(self, image, rois):
        # Start with a black image of the same size as the original
        masked_image = np.zeros_like(image)
        
        # Fill in the regions defined by ROIs from the Roi object
        for img, x1, y1, x2, y2 in zip(rois.images, rois.x1, rois.y1, rois.x2, rois.y2):
            masked_image[y1:y2, x1:x2] = img

        return masked_image
    
    # TODO for now we proceed by extracting the features from each ROI one by one is easier to manage
    def extract_features(self, image):
        # Prepare the image and extract features
        tensor_image = torch.tensor(image).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.
        return self.extractor.extract(tensor_image)


    #TODO to redo this functions 
    #def is_matching_existing_features(self, new_feats):
        # Prepare for matching
    #    new_kpts, new_desc = new_feats['keypoints'], new_feats['descriptors']

    #    for idx, existing_desc in enumerate(self.existing_descriptors):
            # Match against each existing set of descriptors
    #        matches = self.matcher({
    #            "image0": {'keypoints': new_kpts, 'descriptors': new_desc},
    #            "image1": {'keypoints': self.existing_keypoints[idx], 'descriptors': existing_desc}
    #        })
    #        matches = rbd(matches)  # 'rbd' function to remove batch dimension exists

    #        if matches['matches'].size(0) > 0:
    #            return True  # Match found

    #    return False  # No matches found

    
    #TODO to redo this functions 
    #def store_new_features(self, feats):
        # Store new keypoints and descriptors
    #    self.existing_keypoints.append(feats['keypoints'])
    #    self.existing_descriptors.append(feats['descriptors'])

