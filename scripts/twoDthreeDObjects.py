import pymc as pm
import numpy as np
import torch
import cv2, time

# potential object class that contains the keypoints and descriptors of the object
class Potential2dObject:
    # class_numbers is the total number of classes that im looking for in the current experiment
    def __init__(self, total_classes, keypoints=[], descriptors=[], init_class=0, init_roi_center_position_x=0, init_roi_center_position_y=0, idx=0, model_completed=False):   
        # here i have all the list of the existing keypoints and descriptors
        self.total_classes_number = total_classes
        self.existing_keypoints = keypoints
        self.existing_descriptors = descriptors
        #self.alpha = np.ones(classes_number)
        #self.p_label = pm.Dirichlet('class_probs', a=np.ones(classes_number))  # Uniform prior over the simplex
        self.alpha = np.ones(self.total_classes_number)
        class_hot_encoding = self.class_hot_encoding(int(init_class))
        self.alpha += class_hot_encoding
        self.p_label = pm.Dirichlet.dist(a=np.ones(self.total_classes_number), shape=self.total_classes_number)
        # in this attribute i want to store the last roi position in which i have seen the object
        self.last_roi_center_position = {
            'x': init_roi_center_position_x,
            'y': init_roi_center_position_y
        }
        self.is_filtered = False
        self.idx = idx
        self.model_completed = model_completed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def class_hot_encoding(self,cur_class_number):
        cur_class_hot_encoding = np.zeros(self.total_classes_number)
        cur_class_hot_encoding[cur_class_number] = 1
        return cur_class_hot_encoding
    
    def get_idx(self):
        return self.idx

    # update the model with new class information
    def update_model(self, new_class, keypoints, descriptors, idx):
        if self.is_filtered:
            print("Object has been filtered. Skipping update.")
            return
        class_hot_encoding = self.class_hot_encoding(int(new_class))
        self.alpha += class_hot_encoding
        self.p_label = pm.Dirichlet.dist(a=self.alpha, shape=self.total_classes_number)
        # Use new keypoints and descriptors
        self.existing_keypoints = keypoints
        # Use new descriptors
        self.existing_descriptors = descriptors
        self.idx = idx

    # this provide the probability of the to be in certain class
    def evaluate_model(self):
        alpha_sum = np.sum(self.alpha)
        expected_probs = self.alpha / alpha_sum
        max_prob = np.max(expected_probs)
        return max_prob > 0.9 and self.model_completed

    
    def filter_SAM(self, mask):
        if self.is_filtered:
            print("Object has already been filtered. Skipping.")
            return

        #! debug
        # cv2.imshow('mask', mask*255)
        # cv2.waitKey(1)
        # keypoints shape: (1, N, 2)
        keypoints_np = self.existing_keypoints.cpu().numpy()  # shape: (1, N, 2)
        keypoints_np = keypoints_np[0]  # shape: (N, 2)

        # descriptors shape: (1, N, D)
        descriptors_np = self.existing_descriptors.cpu().numpy()  # shape: (1, N, D)
        descriptors_np = descriptors_np[0]  # shape: (N, D)

        filtered_keypoints = []
        filtered_descriptors = []

        for kp, desc in zip(keypoints_np, descriptors_np):
            kp_x, kp_y = kp[0], kp[1]
            kp_x_int, kp_y_int = int(kp_x), int(kp_y)
            
            if 0 <= kp_y_int < mask.shape[0] and 0 <= kp_x_int < mask.shape[1]:
                if mask[kp_y_int, kp_x_int] > 0:
                    filtered_keypoints.append([kp_x, kp_y])
                    filtered_descriptors.append(desc)

        # Transfer filtered keypoints and descriptors back to the device
        if len(filtered_keypoints) > 0:
            filtered_keypoints_array = np.array(filtered_keypoints, dtype=np.float32)  # (M, 2)
            filtered_keypoints_tensor = torch.from_numpy(filtered_keypoints_array).unsqueeze(0)  # (1, M, 2)

            filtered_descriptors_array = np.array(filtered_descriptors, dtype=np.float32)  # (M, D)
            filtered_descriptors_tensor = torch.from_numpy(filtered_descriptors_array).unsqueeze(0)  # (1, M, D)

            filtered_keypoints_tensor = filtered_keypoints_tensor.to(self.device)
            filtered_descriptors_tensor = filtered_descriptors_tensor.to(self.device)

            self.existing_keypoints = filtered_keypoints_tensor
            self.existing_descriptors = filtered_descriptors_tensor
            self.is_filtered = True
            print("Object has been filtered using SAM mask. Keypoints updated.")
        else:
            print("No keypoints left after filtering!")


class Potential2dObjectsManager:
    def __init__(self, total_classes):
        self.total_classes_number = total_classes
        self.model_completed = False
        self.potential_objects = []
    
    def add_potential_object(self, keypoints, descriptors, init_class, init_roi_center_position_x, init_roi_center_position_y, idx):
        potential_object = Potential2dObject(
            self.total_classes_number,
            keypoints,
            descriptors,
            init_class,
            init_roi_center_position_x,
            init_roi_center_position_y,
            idx,
            self.model_completed
        )
        self.potential_objects.append(potential_object)
    
    def get_potential_object(self, object_id):
        return self.potential_objects[object_id]
    
    def get_potential_objects(self):
        return self.potential_objects

    def set_model_completed(self):
        self.model_completed = True
        for potential_object in self.potential_objects:
            potential_object.model_completed = True



# once the object has been fully identified its point are stored using this class which contains the 3d point cloud of the object and the object center
# the object center is the center of the object in the image and other information such as preshape position and oriention to interact with the object    
class threeDObject:
    # this class need to contains the 3d point cloud 
    pass