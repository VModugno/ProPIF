import os
from ultralytics import SAM
import matplotlib.pyplot as plt
import cv2
import numpy as np

def get_image_paths(base_folder):
    image_paths = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Base directory containing the images subfolders
base_dir = 'images'

# Paths to subfolders
kaggle_flowers_path = os.path.join(base_dir, 'kaggle_flowers')
oxford_flowers_path = os.path.join(base_dir, 'oxford_flowers')

# Get image paths from both main subfolders
kaggle_flower_images = get_image_paths(kaggle_flowers_path)
oxford_flower_images = get_image_paths(oxford_flowers_path)




# Load a model
model = SAM('sam_b.pt')

# Run inference
results= model(kaggle_flower_images[100])

image = cv2.imread(kaggle_flower_images[100])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = results[0].masks.data

# Convert CUDA tensor to CPU and then to a NumPy array
masks = masks.cpu().numpy()

# Create a copy of the image for overlay
masked_image = image.copy()

# Assuming masks are numpy arrays and overlaying each mask
for mask in masks:
    # Overlay mask; you can adjust the color and transparency as needed
    masked_image[mask == 1] = [255, 0, 0]  # Example: Red color for the mask



# Set up matplotlib figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display original image
ax1.imshow(image)
ax1.set_title('Original Image')
ax1.axis('off')  # Hide axes

# Display masked image
ax2.imshow(masked_image)
ax2.set_title('Image with Mask')
ax2.axis('off')  # Hide axes

plt.show()