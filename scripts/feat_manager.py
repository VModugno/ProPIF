import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d


class FeatureManager:
    def __init__(self, device):
        # Initialize the device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize the feature extractor and matcher
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        # Lists to store keypoints and descriptors for existing features
        self.existing_keypoints = []
        self.existing_descriptors = []

    def process_new_image(self, image):
        # Extract features from the new image
        feats = self.extract_features(image)
        if self.existing_descriptors:  # Only compare if there are existing features
            if not self.is_matching_existing_features(feats):
                self.store_new_features(feats)
                print("New features stored.")
            else:
                print("Existing features matched.")
        else:
            self.store_new_features(feats)  # Store features if no existing features are present
            print("Initial features stored.")

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