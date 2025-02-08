import pymc as pm
import numpy as np
import torch

# potential object class that contains the keypoints and descriptors of the object
class Potential2dObject:
    # class_numbers is the total number of classes that im looking for in the current experiment
    def __init__(self, total_classes, keypoints=[], descriptors=[], init_class=0, init_roi_center_position_x=0, init_roi_center_position_y=0, idx=0, model_completed=False):   
        # here i have all the list of the existing keypoints and descriptors
        self.total_classes_number = total_classes
        self.existing_keypoints = keypoints
        self.existing_descriptors = descriptors
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
import open3d as o3d

class ThreeDObject:
    def __init__(self, depth_image, pixel_coords, rotation_matrix, translation_vector, camera_intrinsics):
        self.point_cloud = o3d.geometry.PointCloud()

        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        
        points_camera = []
        for (u, v) in pixel_coords:
            u_int = int(round(u))
            v_int = int(round(v))
            if v_int < 0 or v_int >= depth_image.shape[0] or u_int < 0 or u_int >= depth_image.shape[1]:
                continue
            d = depth_image[v_int, u_int]
            if d <= 0:
                continue
            x = (u - cx) * d / fx
            y = (v - cy) * d / fy
            z = d
            points_camera.append([x, y, z])
        points_camera = np.array(points_camera)

        if points_camera.size == 0:
            print("No depth points found for the given pixel coordinates.")
        else:
            # Convert points to global coordinates
            global_points = (rotation_matrix @ points_camera.T).T + translation_vector.reshape(1, 3)
            self.point_cloud.points = o3d.utility.Vector3dVector(global_points)
            
            # # Remove statistical outliers
            cl, ind = self.point_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
            self.point_cloud = self.point_cloud.select_by_index(ind)

    def compute_main_plane(self, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        if np.asarray(self.point_cloud.points).shape[0] == 0:
            print("Error: The points cloud is empty, cannot calculate the main plane.")
            return None, []

        plane_model, inlier_indices = self.point_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        return plane_model, inlier_indices

    def visualize(self):
        o3d.visualization.draw_geometries([self.point_cloud])
