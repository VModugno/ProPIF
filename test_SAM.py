import cv2
from ultralytics import FastSAM
import numpy as np

frame = cv2.imread('./images/image_2.png')
model = FastSAM('FastSAM-s.pt')
# tracker_results=model.track(frame, imgsz=1024, retina_masks=True, 
#                                    conf=0.7, iou=0.5, 
#                                    tracker="botsort.yaml", persist=True)

# text_prompt_results = model.predictor.prompt(tracker_results[0].cuda(), texts="flower")

# combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
# for mask_tensor in text_prompt_results[0].masks.data:
#     mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
#     combined_mask = np.logical_or(combined_mask, mask_array).astype(np.uint8)
#     cv2.imshow('mask', mask_array*255)
#     cv2.waitKey(0)
# cv2.imshow('mask', combined_mask*255)
print(f'frame dtype: {frame.dtype}')
print(f'frame shape: {frame.shape}')
point = [frame.shape[1]//2, frame.shape[0]//2]
cur_results = model.predict(frame, retina_masks=True, conf=0.5, iou=0.5, points=[point])
combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
for cur_result in cur_results:
    if cur_result.masks is not None and len(cur_result.masks.data) > 0:
        for mask_tensor in cur_result.masks.data:
            mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
            combined_mask = np.logical_or(combined_mask, mask_array).astype(np.uint8)
cv2.imshow('mask', combined_mask*255)
cv2.waitKey(0)