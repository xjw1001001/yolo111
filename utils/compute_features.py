import numpy as np
import pandas as pd
from skimage.measure import regionprops

def is_close_to_edge(bbox, edge_threshold=1, img_dim=4096):
    """Check if the bounding box is close to the edge of the image."""
    y_top, x_left, y_bottom, x_right = bbox

    if (x_left < edge_threshold or x_right > img_dim - edge_threshold or
        y_top < edge_threshold or y_bottom > img_dim - edge_threshold):
        return True

    return False

def compute_features(annotations, edge_threshold = 1):
    # List to hold the results
    features = []
    
    for annotation in annotations:
        path = annotation['path']
        name = annotation['name']
        masks = annotation['result']['segment']
        
        for instance in masks:
            mask = instance['mask']
            cls = instance['class']

            # Get region properties
            props = regionprops(mask.astype(int))
            
            if not props:
                # If no regions are found, continue to next mask
                continue

            prop = props[0]  # Assuming a single region per mask

            # Compute properties
            mask_total_pixels = np.sum(mask)
            eccentricity = prop.eccentricity
            major_axis_length = prop.major_axis_length
            minor_axis_length = prop.minor_axis_length
            perimeter = prop.perimeter
            aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0
            bbox = prop.bbox  # This is (minr, minc, maxr, maxc)

            edge_detection = is_close_to_edge(bbox, edge_threshold = edge_threshold,
                                              img_dim=4096)

            # Compile the feature values for this mask
            mask_features = {
                'path': path,
                'name': name,
                'mask_total_pixels': mask_total_pixels,
                'eccentricity': eccentricity,
                'major_axis_length': major_axis_length,
                'minor_axis_length': minor_axis_length,
                'perimeter': perimeter,
                'aspect_ratio': aspect_ratio,
                'y_top': bbox[0],
                'x_left': bbox[1],
                'y_bottom': bbox[2],
                'x_right': bbox[3],
                'class': cls,
                'close_to_edge': edge_detection
            }

            features.append(mask_features)

    return pd.DataFrame(features)
