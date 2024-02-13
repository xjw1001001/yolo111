from .utils import sort_regions, xywh_to_xyxy, predict_sam
import torch
import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

def Load_SAM(ckpt = "/home/xionjing/segment-anything/model_checkpoint/sam_vit_h_4b8939.pth"):
    """
    load the SAM model from ckpt
    """
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(device)
    sam_pred = SamPredictor(sam)
    print("SAM Model load time:" + str(round((start1:=time.time()) - start, 2)) + " seconds.")
    
    return sam_pred

def SAM_infer(results_dict,
              sam_pred = None,
              ckpt = "/home/xionjing/segment-anything/model_checkpoint/sam_vit_h_4b8939.pth"
              ):
    """
    Obtain the segmentation prediction by SAM.
    
    input: 
        results_dict: YoloV8infer result
        sam_pred: SAM model loaded from Load_SAM
        ckpt: check point of SAM model if not loaded

    Output:
        annotation{ "path", "name",
            "result": 
             {"bbox":
                    {0: [],
                    1: [], 
                    2: []} ,
                    "segment": [{
                            "mask": resized_mask,
                            "class": class_id
                        }]}
            }
    """
    start1 = time.time()
    if sam_pred:
        pass
    else:
        Load_SAM(ckpt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    origin_image_path = results_dict["path"]
    img = Image.open(origin_image_path).convert('RGB')
    annotation = {}
    print(len(results_dict["bbox"]))
    # Create list of regions sorted by their center
    regions = sort_regions(results_dict["bbox"])
    annotation["path"] = origin_image_path
    annotation["name"] = results_dict["name"]
    annotation["result"] = {"bbox":
                    {0: [],
                    1: [], 
                    2: []} ,
                    "segment": []}

    # Loop over all regions (bounding boxes) from top-left to bottom-right
    while len(regions) > 0:
        region = regions.pop(0)
        shape_attr = region["bbox"]
        region_class = region["class"]
        
        # Calculate the center of the box
        x_center = shape_attr[0] + shape_attr[2] // 2
        y_center = shape_attr[1] + shape_attr[3] // 2

        # Segment a 1024x1024 image that includes the bounding box
        xmin_segment = max(0, min(x_center - 1024 // 2, 4096 - 1024))
        ymin_segment = max(0, min(y_center - 1024 // 2, 4096 - 1024))
        xmax_segment = xmin_segment + 1024
        ymax_segment = ymin_segment + 1024
        image_1024 = img.crop((xmin_segment,ymin_segment, xmax_segment,  ymax_segment))
        # Other regions in the same patch
        same_patch_regions = []
        
        # Loop over remaining regions to find those in the same patch
        for other_region in regions:
            # Check if the center of other_region is within the same 1024x1024 patch as region
            # If yes, add it to same_patch_regions and remove it from regions
            if other_region['bbox'][0] >= xmin_segment + 64 and other_region['bbox'][1] >= ymin_segment + 64 and \
            other_region['bbox'][0] + other_region['bbox'][2] <= xmax_segment - 64 and \
            other_region['bbox'][1] + other_region['bbox'][3] <= ymax_segment - 64:
                same_patch_regions.append(other_region)
                regions.remove(other_region)
        
        
        # gather the bboxes for each class in that patch
        class_shapes = {0: [], 1: [], 2: []}
        for r in same_patch_regions:
            class_shapes[r["class"]].append(r["bbox"])
        class_shapes[region["class"]].append(region["bbox"])
        # Now you have a dictionary where each class maps to a list of shape_attr values

        # Dictionary to store masks for each class
        class_masks = []
        
        # Process shape_attr for each class
        for class_id, shapes in class_shapes.items():
            # List to store boxes in 1024 size
            boxes_1024 = []
            
            for shape_attr in shapes:
                # Bounding box in xyxy format: [xmin, ymin, xmax, ymax]
                box = xywh_to_xyxy(shape_attr)
                # Convert the original box to the coordinate system of the 1024x1024 image
                # enlarge the box by margin pixels of each endpoint
                ## TODO: change another margin if needed
                margin = 30
                box_1024 = [box[0] - xmin_segment - margin, box[1] - ymin_segment - margin,
                            box[2] - xmin_segment + margin, box[3] - ymin_segment + margin]
                boxes_1024.append(box_1024)
            
            # Need at least have 1 bbox
            if len(boxes_1024)>=1:
                # Reshape boxes and send to the SAM model
                boxes_1024 = torch.tensor(boxes_1024).to(device)
                mask_1024 = predict_sam(sam=sam_pred, 
                    image_pil=image_1024, boxes=boxes_1024, device=device).transpose(2,3)#.flip(-1)
                # Now convert the 1024x1024 mask back to the original 4096x4096 size
                for i in range(mask_1024.shape[0]):
                    mask_4096_after = torch.zeros((4096, 4096)).cpu().numpy()
                    mask_4096_after[ymin_segment:ymax_segment, xmin_segment:xmax_segment] = mask_1024[i,0,:,:].transpose(0,1).cpu().numpy()
                    resized_mask = cv2.resize(mask_4096_after.astype(float), (512, 512), interpolation=cv2.INTER_AREA)
                    resized_mask = (resized_mask > 0.5) != 0
                    class_masks.append(
                        {
                            "mask": resized_mask,
                            "class": class_id
                        }
                        ) 
        
        # Append this annotation to the list
        for class_id, shapes in class_shapes.items():
            annotation["result"]["bbox"][class_id] += shapes
        
        annotation["result"]["segment"] += class_masks
        
    print("SAM Inference time:" + str(round(time.time()-start1,2)) + " seconds.")
    return annotation
    
    
def draw_mask_annotations_512(annotation, segment = True, num_colors = 30,
                              saving_folder = None, plot = True):
    """
    Plot the mask prediction
    
    input: 
        annotation: from function SAM_infer
        saving_folder: image saving folder.
        segment = True: include the segmentation

    Output: Draw a picture.
    """
    img = cv2.imread(annotation["path"])
    # define color for each class (green, blue, red)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)] # colors for bounding box
    bbox_dict = annotation["result"]["bbox"]
    mask_list = annotation["result"]["segment"]
    # Iterate over each class
    for class_type in [0,1,2]:
        # Iterate over each annotation of the class
        if len(bbox_dict[class_type]) > 0:
            for bbox in bbox_dict[class_type]:
                # Convert bbox from xywh to xyxy format
                bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]      
                # Draw the bounding box
                img = cv2.rectangle(img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), colors[class_type], 5)

    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    if segment == True:
        numbers = len(mask_list)
        colors1 = plt.cm.hsv(np.linspace(0, 1, num_colors))
        colors_255 = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors1]
        colors1 = plt.cm.twilight_shifted(np.linspace(0, 1, num_colors))
        colors_255 += ([(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors1])
        for index, mask_dict in enumerate(mask_list):
            mask_img = np.zeros_like(img)
            mask_img[mask_dict['mask'] == True] = colors_255[index*7 % (num_colors*2)]
            # Create alpha mask
            alpha_mask = np.zeros_like(img, dtype=np.uint8)
            alpha_mask[mask_dict['mask'] == True] = 255
            # Alpha composite mask_img and img
            foreground = cv2.bitwise_and(mask_img, alpha_mask)
            background = cv2.bitwise_and(img, cv2.bitwise_not(alpha_mask))
            img = cv2.add(foreground, background)
            
    # Save the image with segments
    if saving_folder:
        os.makedirs(saving_folder, exist_ok=True)
        cv2.imwrite(saving_folder + "/" +annotation["name"] +"bbox.jpg", img)
        print("Segment mask prediction image generated: " + saving_folder + "/" +annotation["name"] +"_segment.jpg")
    if plot == True:
        plt.imshow(img)
        plt.show()