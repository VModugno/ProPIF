import pymc as pm
import numpy as np

# potential object class that contains the keypoints and descriptors of the object
class Potential2dObjects:
    # class_numbers is the total number of classes that im looking for in the current experiment
    def __init__(self, total_classes, keypoints=[], descriptors=[], init_class=0, init_roi_center_position=[]):   
        # here i have all the list of the existing keypoints and descriptors
        self.total_classes_number = total_classes
        self.existing_keypoints = keypoints
        self.existing_descriptors = descriptors
        #self.alpha = np.ones(classes_number)
        #self.p_label = pm.Dirichlet('class_probs', a=np.ones(classes_number))  # Uniform prior over the simplex
        self.alpha = np.ones(self.total_classes_number)
        self.p_label = pm.Dirichlet('class_probs', a=np.ones(self.total_classes_number)) # Uniform prior over the simplex
        # in this attribute i want to store the last roi position in which i have seen the object
        self.last_roi_center_position = init_roi_center_position
        # transform class in one hot encoding
        init_class_hot_encoding = self.class_hot_encoding(init_class)
        
        self.update_model(init_class_hot_encoding)
    
    def class_hot_encoding(self,cur_class_number):
        cur_class_hot_encoding = np.zeros(self.total_classes_number)
        cur_class_hot_encoding[cur_class_number] = 1
        return cur_class_hot_encoding

    def update_model(self,one_hot_label):
        """
        Update the model with a new observation and sample from the posterior.
        
        Parameters:
        - one_hot_label: One-hot encoded label, e.g., [0, 1, 0]
        
        Returns:
        """
        # Update the prior parameters
        self.alpha += one_hot_label  # Update alpha directly here

        # Reset model's parameter 'a' with updated alpha
        self.p_label['class_probs'].distribution.a.set_value(self.alpha)
    
    # this provide the probability of the to be in certain class
    def evaluate_model(self,object_id):
         # Sampling using NUTS via JAX with numpyro backend
        trace = pm.sampling_jax.sample_numpyro_nuts(model=self.p_label[object_id], draws=500, tune=200, target_accept=0.9, random_seed=42)
    
        return trace


class Potential2dObjectsManager:
    def __init__(self, total_classes):
        self.total_classes_number = total_classes
        self.potential_objects = []
    
    def add_potential_object(self, keypoints, descriptors, init_class, init_roi_center_position):
        self.potential_objects.append(Potential2dObjects(self.total_classes_number, keypoints, descriptors, init_class, init_roi_center_position))
    
    def get_potential_object(self, object_id):
        return self.potential_objects[object_id]
    
    def get_potential_objects(self):
        return self.potential_objects



# once the object has been fully identified its point are stored using this class which contains the 3d point cloud of the object and the object center
# the object center is the center of the object in the image and other information such as preshape position and oriention to interact with the object    
class threeDObject:
    # this class need to contains the 3d point cloud 
    pass