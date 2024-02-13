import cv2
import os
import time
from ultralytics import YOLO
from .utils import format_boxes, remove_overlapping_boxes
import matplotlib.pyplot as plt

# Patch and bbox from a picture
def patch_bbox_from_picture(picture_path = '/home/xionjing/br-bards-digitalpathology/CyroEM_segmentation/Inference_pipeline/T39 Rd2 FAP Arm C_15-9-8-2_73000x.png',
                            model_path = "/home/xionjing/CyroEM_segment/best.pt",
                            edge_criterion = 75):
    """
    Patch the picture by 768*768 with sliding window 384, and then resize into 512.
    
    input: 
        picture_path: input cryoEM picture path
        model_path: trained YoloV8 model path: 'best.pt'
        edge_criterion: how close from edge will be determined as "edge"

    Output:
        results_dict{ "name", "path",
            "bbox": [
                {"class":0, # 0 for virus, 1 for ice, 2 for unsure
                "bbox":[1,2,3,4], (xywh)
                "prob":0.8,
                "edge": bool},
                {"class":1, 
                "bbox":[11,12,13,14],
                "prob":0.8,
                "edge": bool},    
        ]}
    """
    
    # Define the size of the patches and sliding window
    patch_size = 768
    slide_size = patch_size // 2
    resized_patch_size = 512

    name = os.path.basename(picture_path)
    print(f"File: {name}")
    # ---------------------------------------
    image = cv2.imread(picture_path)
    
    # Store the bboxes
    results_dict = {"name": name,
                    "path": picture_path,
                    "bbox": []} 
    model = YOLO(model_path)
    start = time.time()
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

            # Resize the patch
            patch = cv2.resize(patch, (resized_patch_size, resized_patch_size))

            y_start = res_i
            x_start = res_j
            ## Obtain bounding boxes
            # Run inference
            result = model(patch, verbose=False)
            # Process the result
            boxes = result[0].boxes
            # Iterate over each detected bounding box
            bbox = format_boxes(boxes, resize = True, x_start = x_start, y_start = y_start,
                                edge_criterion = edge_criterion)

            # Add results
            if len(results_dict["bbox"]) == 0:
                results_dict["bbox"] = bbox
            else:
                results_dict["bbox"] = results_dict["bbox"] + bbox
            
    # Post process the bounding boxes
    results_dict = remove_overlapping_boxes(results_dict, iou_threshold=0.3, overlap_small_threshold = 0.35)        
    print("Bounding boxes:" + str(len(results_dict["bbox"])))
    print("YoloV8 Inference time:" + str(round(time.time() - start, 2)) + " seconds.")
    return(results_dict)


def plot_bboxes(results_dict, saving_folder = None, plot = True):
    """
    Plot the bounding boxes prediction
    
    input: 
        results_dict: from function patch_bbox_from_picture
        saving_folder: image saving folder.

    Output: Draw a picture.
    """
    img = cv2.imread(results_dict["path"])
    annotations = []
    # Loop over all regions (bounding boxes)
    for region in results_dict["bbox"]:
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

        # Draw the bounding box on the image
        top_left = (int(x), int(y))
        bottom_right = (int(x + width), int(y + height))
        thickness = 2  # Thickness of 2 px
        cv2.rectangle(img, top_left, bottom_right, color, thickness)
    # Save the image with bounding boxes
    if saving_folder:
        os.makedirs(saving_folder, exist_ok=True)
        cv2.imwrite(saving_folder + "/" +results_dict["name"] +"bbox.jpg", img)
        print("Bounding box prediction image generated: " + saving_folder + "/" +results_dict["name"] +"_bbox.jpg")
    if plot == True:
        plt.imshow(img)
        plt.show()