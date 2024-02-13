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
import utils.utils
import json
from glob import glob
from PIL import Image
import shutil
import time

folder_path = '/home/xionjing/CyroEM_segment/All pictures/'
# it contains sub folders: T35_Rd2_FAP_Arm_2,...
desired_path = '/home/xionjing/CyroEM_segment/Mask_working/'

# Define the size of the patches and sliding window
patch_size = 768
slide_size = 384
resized_patch_size = 512


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
import utils.utils
import json
# This is the path for sam_vit_b.pth.
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = "/home/xionjing/segment-anything/model_checkpoint/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(device)
sam_pred = SamPredictor(sam)
# This is my trained YoloV8 model best.pt
model_path = "/home/xionjing/CyroEM_segment/best.pt"
model = YOLO(model_path)
# The path to the patched picture for segmentation
one_patch_path = "/home/xionjing/CyroEM_segment/Cocodataset/train/images/T39_Rd2_FAP_Arm_A_16-5-3-5_73000x_0_3072.png"

# Bbox results:
bbox_result_path = "/home/xionjing/CyroEM_segment/Mask_working/bbox_result.json"

if not os.path.exists(bbox_result_path):
    patch_list = utils.utils.make_patches(folder_path, desired_path, slide_size, patch_size, resized_patch_size)
    # [(ori_picture_path, patch_folder)]
    # YOU MAY WANT TO REPLACE THIS
    model_path = "/home/xionjing/CyroEM_segment/best.pt"
    if not os.path.exists(bbox_result_path):
        import os 
        from ultralytics import YOLO
        import cv2
        import time
        #load the model, You Only Look Once V8
        model = YOLO(model_path)
        start = time.time()
        results_dict = {}
        for ori_picture_path, patch_folder_path in patch_list:

            # Get all the images
            image_paths = glob(os.path.join(patch_folder_path, "*.png"))
            for image_path in image_paths:
                # Extract the y and x start positions from the filename
                filename = os.path.basename(image_path)
                name, y_start, x_start = utils.utils.parse_filename(filename)

                # Load image
                img = Image.open(image_path)

                # Run inference
                result = model(img)

                # Process the result
                boxes = result[0].boxes

                # Iterate over each detected bounding box
                bbox = utils.utils.format_boxes(boxes, resize = True, x_start = x_start, y_start = y_start)

            # Add to results dictionary
            if ori_picture_path not in results_dict.keys():
                results_dict[ori_picture_path] = bbox
            else:
                results_dict[ori_picture_path] = results_dict[ori_picture_path] + bbox

    
    results = {"image_and_path": patch_list, "bbox": results_dict, "time_usage": time.time() - start}
    with open(bbox_result_path, "w") as f:
        json.dump(results, f)  


# Approximate total 5 minutes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bbox_json = "/home/xionjing/CyroEM_segment/Mask_working/bbox_result.json"
bbox_json_processed = "/home/xionjing/CyroEM_segment/Mask_working/bbox_result_processed.json"

if not os.path.exists(bbox_json_processed):
    with open(bbox_json, "r") as f:
        data = json.load(f)
        
    results_dict = utils.utils.remove_overlapping_boxes(data["bbox"], iou_threshold=0.3, overlap_small_threshold = 0.35)
    with open(bbox_json_processed, "w") as f:
        json.dump(results_dict, f)  

with open(bbox_json, "r") as f:
        data = json.load(f)
with open(bbox_json_processed, "r") as f:
    results_dict = json.load(f)
results = {"image_and_path": data["image_and_path"], "bbox": results_dict}


import time
import matplotlib.pyplot as plt
segment_method = "multi_box_new"
annotation = {}
start = time.time()

def sort_regions(regions):
    return sorted(regions, key=lambda r: ((r['bbox'][1] + r['bbox'][3] // 2), (r['bbox'][0] + r['bbox'][2] // 2)))

if segment_method == "multi_box_new":
    for origin_image_path, patch_folder_path in results["image_and_path"]:
        start1 = time.time()
        image_name = origin_image_path
        img = Image.open(origin_image_path).convert('RGB')
        # Make sure the image was successfully loaded
        if img is None:
            print(f'Failed to load image: {image_name}')
            continue

        annotations = []
        print(len(results["bbox"][image_name]))
        
        # Create list of regions sorted by their center
        regions = sort_regions(results["bbox"][image_name])

        annotation[origin_image_path] = {"bbox":
                        {0: [],
                        1: [], 
                        2: []} ,
                                        "segment": 
                        {0: torch.zeros((4096, 4096)).cpu().numpy(),
                        1: torch.zeros((4096, 4096)).cpu().numpy(), 
                        2: torch.zeros((4096, 4096)).cpu().numpy()}}
        
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
            class_masks = {0: torch.zeros((4096, 4096)).cpu().numpy(),
                        1: torch.zeros((4096, 4096)).cpu().numpy(), 
                        2: torch.zeros((4096, 4096)).cpu().numpy()}
            
            # Process shape_attr for each class
            for class_id, shapes in class_shapes.items():
                # List to store boxes in 1024 size
                boxes_1024 = []
                
                for shape_attr in shapes:
                    # Bounding box in xyxy format: [xmin, ymin, xmax, ymax]
                    box = utils.utils.xywh_to_xyxy(shape_attr)
                    # Convert the original box to the coordinate system of the 1024x1024 image
                    box_1024 = [box[0] - xmin_segment, box[1] - ymin_segment, box[2] - xmin_segment, box[3] - ymin_segment]
                    boxes_1024.append(box_1024)
                
                # Need at least have 1 bbox
                if len(boxes_1024)>=1:
                    # Reshape boxes and send to the SAM model
                    boxes_1024 = torch.tensor(boxes_1024).to(device)
                    mask_1024 = utils.utils.predict_sam(sam=sam_pred, 
                        image_pil=image_1024, boxes=boxes_1024, device=device).transpose(2,3)#.flip(-1)
                    # Now convert the 1024x1024 mask back to the original 4096x4096 size
                    combined_mask = (mask_1024.sum(dim=0)> 0)
                    class_masks[class_id][ymin_segment:ymax_segment, 
                    xmin_segment:xmax_segment] = combined_mask[0,:,:].transpose(0,1).cpu().numpy()
            
            
            # Append this annotation to the list
            for class_id, shapes in class_shapes.items():
                annotation[origin_image_path]["bbox"][class_id] += shapes
            for class_id, mask in class_masks.items():
                res = annotation[origin_image_path]["segment"][class_id] + mask
                annotation[origin_image_path]["segment"][class_id] = (res>0)
                if class_id == 0:
                    pass
                    #plt.imshow(mask, cmap='gray')
                    #plt.show()
                    #plt.imshow(annotation[origin_image_path]["segment"][class_id], cmap='gray')
                    #plt.show()
                
        print(time.time()-start1)
 
    import pickle
    segment_res = "/home/xionjing/CyroEM_segment/Mask_working/segment_result_with_strat.pkl"
    with open(segment_res, 'wb') as file:
        pickle.dump(annotation, file)

print("Used time:" )
print(time.time()-start)


import matplotlib.pyplot as plt

for i in range(len(results["image_and_path"])):
    segment_res = "/home/xionjing/CyroEM_segment/Mask_working/segment_result_with_strat.pkl"
    
    if os.path.exists(segment_res):
        with open(segment_res, 'rb') as file:
            annotation = pickle.load(file)
    origin_image_path = results["image_and_path"][i][0]
    # Read the image
    img = cv2.imread(origin_image_path)
    img_annotated_with = utils.utils.draw_mask_annotations(img, annotation[origin_image_path], segment = True)
   
    img_bbox = utils.utils.draw_mask_annotations(img, annotation[origin_image_path], segment = False)
    
    #plt.imshow(img_bbox)
    #plt.show()
    
    # Get directory and filename
    dir_name = os.path.basename(os.path.dirname(origin_image_path))
    file_name = os.path.basename(origin_image_path)
    # Combine and replace spaces with underscores
    output_folder = "/home/xionjing/CyroEM_segment/Mask_working/Segment_result/" + (dir_name + "/" + file_name).replace(' ', '_').replace(".png","")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(output_folder + "/" + "Segment.png", img_annotated_with)
    cv2.imwrite(output_folder + "/" + "bbox.png", img_bbox)