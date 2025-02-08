from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive

import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class CameraLoc:
    Rotation_matrix: np.ndarray
    Translation_vector: np.ndarray

class CameraLocManager:
    def __init__(self, height, width, focal_length, center_x, center_y, k):
        if not os.path.exists('.cache'):
            os.makedirs('.cache')
        if not os.path.exists('.cache/outputs'):
            os.makedirs('.cache/outputs')
        if not os.path.exists('.cache/mapping'):
            os.makedirs('.cache/mapping')
        if not os.path.exists('.cache/query'):
            os.makedirs('.cache/query')
        self.cache_folder = Path('.cache/')
        self.outputs_folder = Path('.cache/outputs/')
        self.query_path = 'query/query.png'
        self.sfm_pairs = self.outputs_folder / 'pairs-sfm.txt'
        self.loc_pairs = self.outputs_folder / 'pairs-loc.txt'
        self.sfm_dir = self.outputs_folder / 'sfm'
        self.features = self.outputs_folder / 'features.h5'
        self.matches = self.outputs_folder / 'matches.h5'
        
        self.feature_conf = extract_features.confs['disk']
        self.matcher_conf = match_features.confs['disk+lightglue']
        
        self.references = [str(p.relative_to(self.cache_folder)) for p in (self.cache_folder / 'mapping/').iterdir()]
        
        self.height = height
        self.width = width
        self.camera_intrinsics = [focal_length, center_x, center_y, k]
        
        self.model = None
    
    def reconstruction_3D(self):
        extract_features.main(self.feature_conf, self.cache_folder, image_list=self.references, feature_path=self.features)
        pairs_from_exhaustive.main(self.sfm_pairs, image_list=self.references)
        match_features.main(self.matcher_conf, self.sfm_pairs, features=self.features, matches=self.matches)
        self.model = reconstruction.main(self.sfm_dir, self.cache_folder, self.sfm_pairs, self.features, self.matches, image_list=self.references)
        
    def get_cam_loc(self):
        references_registered = [self.model.images[i].name for i in self.model.reg_image_ids()]
        extract_features.main(self.feature_conf, self.cache_folder, image_list=[self.query_path], feature_path=self.features, overwrite=True)
        pairs_from_exhaustive.main(self.loc_pairs, image_list=[self.query_path], ref_list=references_registered)
        match_features.main(self.matcher_conf, self.loc_pairs, features=self.features, matches=self.matches, overwrite=True)
        
        camera = pycolmap.Camera(
            model="SIMPLE_RADIAL",
            width=self.width,
            height=self.height,
            params=self.camera_intrinsics
        )
        
        ref_ids = [self.model.find_image_with_name(n).image_id for n in references_registered]
        conf = {
            'estimation': {'ransac': {'max_error': 12}},
            'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
        }
        localizer = QueryLocalizer(self.model, conf)
        ret, _ = pose_from_cluster(localizer, self.query_path, camera, ref_ids, self.features, self.matches)
        
        Rc = np.array(ret["cam_from_world"].rotation.matrix())
        tc = ret["cam_from_world"].translation
        return CameraLoc(Rotation_matrix=Rc, Translation_vector=tc)
                