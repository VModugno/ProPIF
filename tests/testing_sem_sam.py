from semantic_sam import prepare_image, plot_multi_results, build_semantic_sam, SemanticSAMPredictor
import os

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


original_image, input_image = prepare_image(image_pth=kaggle_flower_images[100])  # change the image path to your image
mask_generator = SemanticSAMPredictor(build_semantic_sam(model_type='<model_type>', ckpt='</your/ckpt/path>')) # model_type: 'L' / 'T', depends on your checkpint
iou_sort_masks, area_sort_masks = mask_generator.predict_masks(original_image, input_image, point='<your prompts>') # input point [[w, h]] relative location, i.e, [[0.5, 0.5]] is the center of the image
plot_multi_results(iou_sort_masks, area_sort_masks, original_image, save_path='/vis/')  # results and original images will be saved at save_path
