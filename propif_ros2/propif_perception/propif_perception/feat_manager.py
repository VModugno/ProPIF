import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
import numpy as np
import cv2
from propif_perception.twoDthreeDObjects import Potential2dObjectsManager, ThreeDObject
from ultralytics import FastSAM
from dataclasses import dataclass

@dataclass
class PlaneInfo:
    object_idx: int
    normal: np.ndarray  # shape (3,) float
    centroid: np.ndarray # shape (3,) float
    threed_object: object = None


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

    def process_new_image(self, image, depth_image, rois, classes, rotation_matrix, translation_vector, camera_intrinsics, DEBUGWINDOWVIDEO):
        plane_info_list = []

        if len(rois.images) == 0:
            print("No ROIs found.")
        else:
            # Extract features from ROIs
            for idx, roi_image in enumerate(rois.images):
                features = self.extract_features(roi_image)
        
                # Convert keypoints to global coordinates
                x1, y1 = rois.x1[idx], rois.y1[idx]
                keypoints_np = features['keypoints'][0].cpu().numpy()
                keypoints_np[:, 0] += x1
                keypoints_np[:, 1] += y1

                device = features['keypoints'][0].device
                features['keypoints'][0] = torch.from_numpy(keypoints_np).to(device)

                rois.add_features(features)

            cv2.waitKey(1)

            if DEBUGWINDOWVIDEO:
                input("Press Enter to continue...")
                
            # Process existing objects
            if len(self.objects2dMan.get_potential_objects()) > 0:  # Only compare if there are existing features
                pop_idx_list = []
                for idx, cur_potential_2dobject in enumerate(self.objects2dMan.get_potential_objects()):
                    # Check if the new features match any existing features, and update the objects stored
                    pop_idx_list += self.matching_existing_features(cur_potential_2dobject, rois)

                # Get 3D data with direct camera pose parameters
                plane_info_list = self.convert_to_3d(image, depth_image, rois, classes, 
                                                   rotation_matrix, translation_vector, camera_intrinsics)

                # Remove the ROIs that have been matched
                pop_idx_list = list(set(pop_idx_list))
                self.pop_rois(rois, pop_idx_list)

                # Store new objects if there are any remaining ROIs
                if len(rois.images) > 0:
                    self.store_new_2dobjects(rois)
            else:
                # Initiate new objects if there's no existing objects
                self.store_new_2dobjects(rois)  # Store features if no existing features are present
                print("Initial features from ROIs stored.")
        if plane_info_list:
            print(f'plane info list: {plane_info_list} !!!!!!!!!!!!!!!!!!!!!!')
        return plane_info_list

    
    def convert_to_3d(self, image, depth_image, rois, classes, rotation_matrix, translation_vector, camera_intrinsics):
        plane_info_list = []
        for obj in self.objects2dMan.get_potential_objects():
            if obj.is_filtered:
                continue
            if not obj.evaluate_model():
                continue
                
            obj_idx = obj.get_idx()
            matching_roi_idx = self.find_matching_roi(obj, rois)
            
            if matching_roi_idx is not None:
                print(f'Found matching ROI at index {matching_roi_idx}')
                full_mask = self.generate_combined_mask_for_object(rois, image, classes, matching_roi_idx)
                if full_mask is not None:
                    obj.filter_SAM(full_mask)
            else:
                print('No matching ROI found, using existing keypoints without filtering')
                full_mask = None
                
            keypoints_np = obj.existing_keypoints.cpu().numpy()  # shape: (1, N, 2)
            pixel_coords = keypoints_np[0]  # shape: (N, 2)
            
            # Debug visualization: draw keypoints on image copy
            debug_image = image.copy()
            for point in pixel_coords:
                x, y = int(point[0]), int(point[1])
                cv2.circle(debug_image, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow('Keypoints before 3D conversion', debug_image)
            cv2.waitKey(1)  # Show for at least 1ms
            
            three_d_object = ThreeDObject(depth_image, pixel_coords, 
                                        rotation_matrix, 
                                        translation_vector, 
                                        camera_intrinsics)
            normal, centroid = three_d_object.compute_main_plane()
            if normal is not None and centroid is not None:
                info = PlaneInfo(
                    object_idx=obj_idx,
                    normal=normal,
                    centroid=centroid,
                    threed_object=three_d_object
                )
                plane_info_list.append(info)
        
        return plane_info_list
    
    def find_matching_roi(self, obj, rois):
        if len(rois.images) == 0:
            return None
            
        distances = []
        for idx, _ in enumerate(rois.images):
            distance = np.linalg.norm(
                np.array([obj.last_roi_center_position['x'], obj.last_roi_center_position['y']]) - 
                np.array([rois.cx[idx], rois.cy[idx]])
            )
            distances.append((distance, idx))
        
        distances.sort(key=lambda x: x[0])
        
        if distances and distances[0][0] < 100:
            return distances[0][1]
        return None

    def generate_combined_mask_for_object(self, rois, image, classes, roi_idx):
        if roi_idx >= len(rois.images):
            print(f"Error: ROI index {roi_idx} out of range")
            return None
            
        print(f'Generating mask for object {classes[int(rois.classes[roi_idx])]}')
        roi_center_x = rois.images[roi_idx].shape[1] // 2
        roi_center_y = rois.images[roi_idx].shape[0] // 2
        roi_center_point = [roi_center_x, roi_center_y]
        roi_img = rois.images[roi_idx]
        cur_results = self.fast_sam_model.predict(rois.images[roi_idx], retina_masks=True, conf=0.1, iou=0.2, points=[roi_center_point])
        
        combined_mask = np.zeros((roi_img.shape[0], roi_img.shape[1]), dtype=np.uint8)
        for cur_result in cur_results:
            if cur_result.masks is not None and len(cur_result.masks.data) > 0:
                for mask_tensor in cur_result.masks.data:
                    mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
                    combined_mask = np.logical_or(combined_mask, mask_array).astype(np.uint8)
        rois.add_mask(roi_idx, combined_mask)

        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        for index, mask_array in rois.masks:
            x1_i, y1_i = rois.x1[index], rois.y1[index]
            x2_i, y2_i = rois.x2[index], rois.y2[index]
            roi_area = full_mask[y1_i:y2_i, x1_i:x2_i]
            #! Debug
            cv2.imshow('mask', mask_array*255)
            cv2.waitKey(1)
            full_mask[y1_i:y2_i, x1_i:x2_i] = np.logical_or(roi_area, mask_array).astype(np.uint8)

        return full_mask
    
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
        for idx in sorted_indices:
            rois.images.pop(idx)
            rois.features.pop(idx)
            rois.cx.pop(idx)
            rois.cy.pop(idx)
            rois.classes.pop(idx)

    def store_new_2dobjects(self,rois):
        for idx, _ in enumerate(rois.images):
            self.objects2dMan.add_potential_object(
                rois.features[idx]['keypoints'],
                rois.features[idx]['descriptors'],
                rois.classes[idx],
                rois.cx[idx],
                rois.cy[idx],
                len(self.objects2dMan.get_potential_objects())
            )
    def set_model_completed(self):
        self.objects2dMan.set_model_completed()