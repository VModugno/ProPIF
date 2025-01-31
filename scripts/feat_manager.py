import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
import numpy as np
import cv2
from twoDthreeDObjects import Potential2dObjectsManager, Potential2dObjects
from ultralytics import FastSAM
import time


class FeatureManager:
    def __init__(self, device, classes_number):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize the feature extractor and matcher
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        self.objects2dMan=Potential2dObjectsManager(classes_number)
        
        # Loading FastSAM model
        self.fast_sam_model = FastSAM('FastSAM-s.pt')
        self.fast_sam_model.model.to(self.device)

    def process_new_image(self, image, rois, classes, cam_loc_manager, DEBUGWINDOWVIDEO):
        # Create an image where only the rois are visible
        if len(rois.images) == 0:
            print("No ROIs found.")
        else:
            rois_image = self.create_masked_image(image, rois)
            rois_image_rgb = cv2.cvtColor(rois_image, cv2.COLOR_BGR2RGB)
            rois_image_with_keypoints = rois_image_rgb.copy()

            for idx, roi_image in enumerate(rois.images):
                features = self.extract_features(roi_image)
    
                # COnvert keypoints to global coordinates
                x1, y1 = rois.x1[idx], rois.y1[idx]
                keypoints_np = features['keypoints'][0].cpu().numpy()
                keypoints_np[:, 0] += x1
                keypoints_np[:, 1] += y1

                device = features['keypoints'][0].device
                features['keypoints'][0] = torch.from_numpy(keypoints_np).to(device)

                rois.add_features(features)

                keypoints_cv = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints_np]
                rois_image_with_keypoints = cv2.drawKeypoints(rois_image_with_keypoints, keypoints_cv, None, color=(0, 255, 0))

            cv2.imshow("rois_image_with_keypoints", rois_image_with_keypoints)
            cv2.waitKey(1)

            #! Debug frame by frame
            if DEBUGWINDOWVIDEO:
                input("Press Enter to continue...")
            
            # here i check if there are existing descripotrs and keypoints
            #! Note that is matching function will remove the roi from the list of rois
            if len(self.objects2dMan.get_potential_objects())>0:  # Only compare if there are existing features
                pop_idx_list = []
                for idx, cur_potential_2dobject in enumerate(self.objects2dMan.get_potential_objects()):
                    # Check if the new features match any existing features, and update the objects stored
                    pop_idx_list += self.matching_existing_features(cur_potential_2dobject, rois)
                #! Densify objects
                self.check_and_densify_objects(image, rois, classes, cam_loc_manager)
                # Remove the ROIs that have been matched
                pop_idx_list = list(set(pop_idx_list))
                self.pop_rois(rois, pop_idx_list)
                # Store new objects if there are any remaining ROIs
                if len(rois.images)>0:
                    self.store_new_2dobjects(rois)
                    print("New features stored from ROIs.")
            else:
                # Initiate new objects if there's no existing objects
                self.store_new_2dobjects(rois)  # Store features if no existing features are present
                print("Initial features from ROIs stored.")
    
    def check_and_densify_objects(self, image, rois, classes, cam_loc_manager):
        print(f'length of objects list: {len(self.objects2dMan.get_potential_objects())}')
        for obj in self.objects2dMan.get_potential_objects():
            if not obj.is_filtered:
                if obj.evaluate_model():
                    obj_idx = obj.get_idx()
                    full_mask = self.generate_combined_mask_for_object(rois, image, classes, obj_idx)
                    if full_mask is not None:
                        obj.filter_SAM(full_mask)
                        cv2.imwrite('.cache/query/query.png', image)
                        cam_loc = cam_loc_manager.get_cam_loc()
                        print(f'Camera location: {cam_loc.Rotation_matrix}, {cam_loc.Translation_vector}')
                        # TODO Convert keypoints to 3D points, Store 3D objects here!!!
                        #! plot filtered keypoints on the image, debug
                        keypoints_np = obj.existing_keypoints.cpu().numpy()
                        keypoints_cv = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints_np[0]]
                        image_with_keypoints = cv2.drawKeypoints(image, keypoints_cv, None, color=(0, 255, 0))
                        cv2.imshow("image_with_filtered_keypoints", image_with_keypoints)
                        cv2.waitKey(0)

    def generate_combined_mask_for_object(self, rois, image, classes, obj_idx):
        print(f'Generating mask for object {classes[int(rois.classes[obj_idx])]}')
        roi_center_x = rois.images[obj_idx].shape[1] // 2
        roi_center_y = rois.images[obj_idx].shape[0] // 2
        roi_center_point = [roi_center_x, roi_center_y]
        roi_img = rois.images[obj_idx]
        cur_results = self.fast_sam_model.predict(rois.images[obj_idx], retina_masks=True, conf=0.4, iou=0.5, points=[roi_center_point])
        
        combined_mask = np.zeros((roi_img.shape[0], roi_img.shape[1]), dtype=np.uint8)
        for cur_result in cur_results:
            if cur_result.masks is not None and len(cur_result.masks.data) > 0:
                for mask_tensor in cur_result.masks.data:
                    mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
                    combined_mask = np.logical_or(combined_mask, mask_array).astype(np.uint8)
        rois.add_mask(obj_idx, combined_mask)

        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        for index, mask_array in rois.masks:
            x1_i, y1_i = rois.x1[index], rois.y1[index]
            x2_i, y2_i = rois.x2[index], rois.y2[index]
            roi_area = full_mask[y1_i:y2_i, x1_i:x2_i]
            #! Debug
            # cv2.imshow('mask', mask_array*255)
            # cv2.waitKey(0)
            full_mask[y1_i:y2_i, x1_i:x2_i] = np.logical_or(roi_area, mask_array).astype(np.uint8)

        return full_mask

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

    
    def matching_existing_features(self, cur_potential_2dobject, rois):
        # Prepare for matching
        # create a list of index with the rois from the closest to the farthest to the current object
        
        distances = []
        pop_idx_list = []
        for idx, _ in enumerate(rois.images):
            distance = np.linalg.norm(np.array([cur_potential_2dobject.last_roi_center_position['x'], cur_potential_2dobject.last_roi_center_position['y']]) - np.array([rois.cx[idx], rois.cy[idx]]))
            distances.append((distance, idx))
        
        # Step 2: Sort pairs based on distance
        distances.sort(key=lambda x: x[0])
        
        # Step 3: Extract sorted indices
        sorted_indices = [idx for _, idx in distances]
        
        for idx in sorted_indices:
            # Extract features from the ROI
            cur_feats = rois.features[idx]
            classes = rois.classes[idx]
            
            #! Debug
            # print(f'existing keypoints: {cur_potential_2dobject.existing_keypoints.size()}')
            # print(f'existing descriptors: {cur_potential_2dobject.existing_descriptors.size()}')
            
            # Match against each existing set of descriptors
            matches = self.matcher({
                "image0": {'keypoints': cur_potential_2dobject.existing_keypoints, 'descriptors': cur_potential_2dobject.existing_descriptors},
                "image1": {'keypoints': cur_feats['keypoints'], 'descriptors': cur_feats['descriptors']}
            })
            matches = rbd(matches)
            # if i have at least 80% match i consider the object to be the same
            if matches['matches'].size(0) > 0 and matches['matches'].size(0)/cur_feats['keypoints'].size(0) > 0.8:
                # Update the object with the new features, only update if the object is not filled
                if not cur_potential_2dobject.is_filtered:
                    cur_potential_2dobject.update_model(classes, rois.features[idx]['keypoints'], rois.features[idx]['descriptors'], idx)
                    cur_potential_2dobject.last_roi_center_position_x = rois.cx[idx]
                    cur_potential_2dobject.last_roi_center_position_y = rois.cy[idx]
                #! remove the current roi from the list of rois
                pop_idx_list.append(idx)
                break
        return pop_idx_list
    
    def pop_rois(self, rois, idx_list):
        # Sort the index list in descending order to avoid shifting indices
        sorted_indices = sorted(idx_list, reverse=True)
        print(f'Popping ROIs: {sorted_indices}')
        for idx in sorted_indices:
            print(f'Removing ROI {idx} from list.')
            rois.images.pop(idx)
            rois.features.pop(idx)
            rois.cx.pop(idx)
            rois.cy.pop(idx)
            rois.classes.pop(idx)

    def store_new_2dobjects(self,rois):
        for idx, _ in enumerate(rois.images):
            print(f'Adding new object with class {rois.classes[idx]}')
            if rois.classes[idx] != 3.0:
                self.objects2dMan.add_potential_object(rois.features[idx]['keypoints'], rois.features[idx]['descriptors'], rois.classes[idx],rois.cx[idx],rois.cy[idx], idx)

