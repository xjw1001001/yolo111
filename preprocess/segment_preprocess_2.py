import cv2
import json
import numpy as np
import os
import shutil
from PIL import Image

# Define the size of the patches and sliding window
patch_size = 768
slide_size = 384
resized_patch_size = 512

# Load the annotations
with open('All_pictures/found_pics.json', 'r') as f:
    annotations = json.load(f)

patch_annotations = {}

# Loop over all images and their annotations
for annotation in annotations:
    # Load the image
    train_path = f"All_pictures/train/{annotation['filename']}"
    val_path = f"All_pictures/val/{annotation['filename']}"
    if os.path.exists(train_path):
        image = cv2.imread(train_path)
        flag = "train"
    else:
        image = cv2.imread(val_path)
        flag = "val"
    
    for i in range(0, image.shape[0], slide_size):
        for j in range(0, image.shape[1], slide_size):
            # Get the patch
            res_i = i
            res_j = j
            if i+patch_size <= image.shape[0] and j+patch_size <= image.shape[1]:
                patch = image[i:i+patch_size, j:j+patch_size]
            elif i+patch_size > image.shape[0] and j+patch_size <= image.shape[1]:
                patch = image[(image.shape[0]-patch_size):image.shape[0], j:j+patch_size]
                res_i = image.shape[0]-patch_size
            elif i+patch_size <= image.shape[0] and j+patch_size > image.shape[1]:
                patch = image[i:i+patch_size, (image.shape[1]-patch_size):image.shape[1]]
                res_j = image.shape[1]-patch_size
            elif i+patch_size > image.shape[0] and j+patch_size > image.shape[1]:
                patch = image[image.shape[0]-patch_size:image.shape[0], (image.shape[1]-patch_size):image.shape[1]]
                res_i = image.shape[0]-patch_size
                res_j = image.shape[1]-patch_size
            
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                print("pass")
                continue  # Skip patches that are not the correct size

            # Get the bounding boxes inside this patch
            patch_bboxes = []
            for bbox in annotation['annotations']:
                # Convert the bbox to the patch's coordinate system
                bbox_x = bbox['bbox'][0] - j
                bbox_y = bbox['bbox'][1] - i
                bbox_w = bbox['bbox'][2]
                bbox_h = bbox['bbox'][3]

                # Check if the midpoint of the bbox is inside the patch
                midpoint_x = bbox_x + (bbox_w / 2)
                midpoint_y = bbox_y + (bbox_h / 2)
                if 0 <= midpoint_x < patch_size and 0 <= midpoint_y < patch_size:
                    # Check if at least 3/5 of the bbox is inside the patch
                    if (bbox_w * bbox_h) * 0.6 <= min(patch_size - bbox_x, bbox_w) * min(patch_size - bbox_y, bbox_h):
                        patch_bboxes.append({'type': bbox['type'], 'bbox': [bbox_x, bbox_y, bbox_w, bbox_h]})

            # If there are no bboxes in this patch, skip it
            if not patch_bboxes:
                continue


            # Resize the patch
            patch = cv2.resize(patch, (resized_patch_size, resized_patch_size))

            # Save the patch
            patch_filename = f"{annotation['filename'].split('.')[0]}_{res_i}_{res_j}.png"
            folder = f"All_pictures/patches/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            cv2.imwrite(f"All_pictures/patches/{patch_filename}", patch)

            # Update the bboxes to the new patch size
            for bbox in patch_bboxes:
                bbox['bbox'] = [int(bbox['bbox'][0] * (resized_patch_size / patch_size)), 
                                int(bbox['bbox'][1] * (resized_patch_size / patch_size)), 
                                int(bbox['bbox'][2] * (resized_patch_size / patch_size)), 
                                int(bbox['bbox'][3] * (resized_patch_size / patch_size))]

            # Add the patch and its bboxes to the patch annotations
            patch_annotations[patch_filename] = {'filename': patch_filename, 'split': flag, 'annotations': patch_bboxes}

# Save the patch annotations
with open('All_pictures/patches/patch_annotations.json', 'w') as f:
    json.dump(patch_annotations, f)


# Load the annotations
with open('All_pictures/patches/patch_annotations.json', 'r') as f:
    patch_annotations = json.load(f)

# Loop over all patch images and their annotations
for file_name, image_annotations in patch_annotations.items():
    # Read the image
    patch = cv2.imread(f'All_pictures/patches/{file_name}')
    
    # Loop over all annotations for this patch
    for annotation in image_annotations['annotations']:
        # Get the bounding box coordinates
        bbox = annotation['bbox']
        
        # Draw the bounding box on the patch
        top_left = (bbox[0], bbox[1])
        bottom_right = (bbox[0]+bbox[2], bbox[1]+bbox[3])
        # Set color according to class
        class_name = annotation["type"]
        if class_name == "virus":
            color = (0, 255, 0)  # Green for virus
        elif class_name == "ice":
            color = (255, 0, 0)  # Red for ice
        elif class_name == "unsure":
            color = (0, 0, 255)  # Blue for unsure
        else:
            color = (21, 255, 255)  # Blue for other classes
        thickness = 2  # Thickness of 2 px
        cv2.rectangle(patch, top_left, bottom_right, color, thickness)

    # Save the patch with bounding boxes
    cv2.imwrite(f'All_pictures/patches_with_bbox/{file_name}', patch)