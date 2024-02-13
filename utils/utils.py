import os
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import glob
import shutil
import json

def xywh_to_xyxy(box):
    x_top_left, y_top_left, width, height = box
    x_bottom_right = x_top_left + width
    y_bottom_right = y_top_left + height

    return [x_top_left, y_top_left, x_bottom_right, y_bottom_right]


def format_boxesxyxy(boxes, resize = False):
    results = []
    for i in range(len(boxes.xyxy)):
        # convert from xyxy to xywh format
        x1, y1, x2, y2 = boxes.xyxy[i]

        # get the class and confidence score
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())

        # add the result to the list
        results.append({
            "class": cls,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "prob": conf,
        })

    return results

def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def draw_image(image, masks, boxes, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes),width=2)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

# Helper functions
import re

def is_close_to_edge(bbox, edge_criterion, picture_x, picture_y):
    x, y, w, h = bbox

    # Calculate the start and end coordinates for x and y
    x_start, x_end = x, x + w
    y_start, y_end = y, y + h

    # Check if the bounding box is close to the edge
    if (x_start < edge_criterion or x_end > picture_x - edge_criterion or
        y_start < edge_criterion or y_end > picture_y - edge_criterion):
        return True

    return False

def format_boxes(boxes, resize = False, x_start = 0, y_start = 0, edge_criterion = 75):
    results = []
    for i in range(len(boxes.xyxy)):
        # convert from xyxy to xywh format
        x1, y1, x2, y2 = boxes.xyxy[i]
        
        if resize == True:
            # Resize the coordinates from 512 to 768
            x1, x2 = [x * (768 / 512) for x in [x1, x2]]
            y1, y2 = [y * (768 / 512) for y in [y1, y2]]

            # Add the start position
            x1 += x_start
            x2 += x_start
            y1 += y_start
            y2 += y_start
        
        width = x2 - x1
        height = y2 - y1
        
        # get the class and confidence score
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())

        
        # add the result to the list
        results.append({
            "class": cls,
            "bbox": [int(x1), int(y1), int(width), int(height)],
            "prob": conf,
            "edge": is_close_to_edge(bbox = [int(x1), int(y1), int(width), int(height)],
                                     edge_criterion = edge_criterion, 
                                     picture_x = 4096, picture_y = 4096)
        })

    return results


def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Compute the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1+w1, x2+w2)
    y_bottom = min(y1+h1, y2+h2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou

def compute_overlap_percentage(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Compute the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1+w1, x2+w2)
    y_bottom = min(y1+h1, y2+h2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    smaller_box_area = min(bbox1_area, bbox2_area)
    overlap_percentage = intersection_area / smaller_box_area

    return overlap_percentage


def remove_overlapping_boxes(data, iou_threshold=0.3, overlap_small_threshold = 0.24):
    boxes = data["bbox"]
    boxes.sort(key=lambda x: x['bbox'][2]*x['bbox'][3])  # Sort boxes by area (smaller first)
    
    # Compare each box with all other boxes and remove if it overlaps too much
    i = 0
    while i < len(boxes):
        box1 = boxes[i]
        
        j = i + 1
        while j < len(boxes):
            box2 = boxes[j]
            
            if box1['class'] != box2['class'] and compute_overlap_percentage(box1['bbox'], box2['bbox']) > 0.66:
                # If box1 is smaller than box2, remove box1
                if box1['bbox'][2]*box1['bbox'][3] < box2['bbox'][2]*box2['bbox'][3]:
                    boxes.pop(i)
                    break  # Exit the inner loop and recheck box at index i
                else:  # If box2 is smaller or equal, remove box2
                    boxes.pop(j)
            # Check if the boxes have the same class and overlap more than the threshold
            if box1['class'] == box2['class'] and compute_iou(box1['bbox'], box2['bbox']) > iou_threshold:
                # If box1 is smaller than box2, remove box1
                if box1['bbox'][2]*box1['bbox'][3] < box2['bbox'][2]*box2['bbox'][3]:
                    boxes.pop(i)
                    break  # Exit the inner loop and recheck box at index i
                else:  # If box2 is smaller or equal, remove box2
                    boxes.pop(j)
            elif box1['class'] == box2['class'] and compute_overlap_percentage(box1['bbox'], box2['bbox']) > overlap_small_threshold:
                # If box1 is smaller than box2, remove box1
                if box1['bbox'][2]*box1['bbox'][3] < box2['bbox'][2]*box2['bbox'][3]:
                    boxes.pop(i)
                    break  # Exit the inner loop and recheck box at index i
                else:  # If box2 is smaller or equal, remove box2
                    boxes.pop(j)
            
            else:
                j += 1
        
        else:  # Increment i only if the inner loop wasn't exited by 'break'
            i += 1
                
    return data

def predict_sam(sam, image_pil, boxes, device):
    image_array = np.asarray(image_pil)
    sam.set_image(image_array)
    transformed_boxes = sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    return masks.cpu()


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def draw_image(image, masks, boxes, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes),width=2)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

def draw_bbox(origin_image_path, patch_folder_path, data):
    # Load the annotations
    img = cv2.imread(origin_image_path)
    # Make sure the image was successfully loaded
    if img is None:
        print(f'Failed to load image: {origin_image_path}')
    annotations = []
    # Loop over all regions (bounding boxes)
    for region in data[origin_image_path]:
        shape_attr = region["bbox"]
        region_class = region["class"]
        region_prob = region["prob"]
        # Get the bounding box coordinates and scale up by a factor of 10
        x = shape_attr[0]
        y = shape_attr[1]
        width = shape_attr[2]
        height = shape_attr[3]

        # Append this annotation to the list
        annotations.append({
            "type": region_class,
            "bbox": [x, y, width, height],
            "prob": region_prob
        })
        
        # Set color according to class
        class_name = region_class
        if class_name == 0:
            color = (0, 255, 0)  # Green for virus
        elif class_name == 1:
            color = (255, 0, 0)  # Red for ice
        elif class_name == 2:
            color = (0, 0, 255)  # Blue for unsure
        else:
            color = (21, 255, 255)  # Blue for other classes

        # Draw the bounding box on the image
        top_left = (int(x), int(y))
        bottom_right = (int(x + width), int(y + height))
        thickness = 2  # Thickness of 2 px
        cv2.rectangle(img, top_left, bottom_right, color, thickness)


    plt.imshow(img)
    plt.show()
    
def draw_mask_annotations(img, annotation, segment = True):
    # define color for each class (red, green, blue)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] #
    bbox_dict = annotation["bbox"]
    mask_dict = annotation["segment"]

    # Iterate over each class
    for class_type in [0,1,2]:
        # Iterate over each annotation of the class
        if len(bbox_dict[class_type]) > 0:
            for bbox in bbox_dict[class_type]:
                # Convert bbox from xywh to xyxy format
                bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]      
                # Draw the bounding box
                img = cv2.rectangle(img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), colors[class_type], 2)

            if segment == True:
                mask_img = np.zeros_like(img)
                mask_img[mask_dict[class_type] == True] = colors[class_type]
                # Create alpha mask
                alpha_mask = np.zeros_like(img, dtype=np.uint8)
                alpha_mask[mask_dict[class_type] == True] = 255
                # Alpha composite mask_img and img
                foreground = cv2.bitwise_and(mask_img, alpha_mask)
                background = cv2.bitwise_and(img, cv2.bitwise_not(alpha_mask))
                img = cv2.add(foreground, background)

    return img

def sort_regions(regions):
    return sorted(regions, key=lambda r: ((r['bbox'][1] + r['bbox'][3]) // 2, (r['bbox'][0] + r['bbox'][2]) // 2))