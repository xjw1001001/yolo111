import cv2
import json
import os
import re


# Load the annotations
with open('C:/Users/xionjing/OneDrive - Merck Sharp & Dohme LLC/Desktop/CyroEM_segment/results_train.json', 'r') as f:
    data = json.load(f)

# Loop over all images
# T35_Rd2_FAP_Arm_2_T35FAP~1
#T39 Rd2 FAP Arm B_10-7-7-3_73000x
# T39 Rd2 FAP Arm B_10-7-3-2_73000x
# T39_Rd2_FAP_Arm_A_16-5-2-4_73000x

for image_name1 in data.keys():
    image_name = image_name1+".png"
    path = "C:/Users/xionjing/OneDrive - Merck Sharp & Dohme LLC/Desktop/CyroEM_segment/All_pictures/train"
    img = cv2.imread(path + "/" + image_name)

    # Make sure the image was successfully loaded
    if img is None:
        print(f'Failed to load image: {image_name}')
        continue

    annotations = []
    # Loop over all regions (bounding boxes)
    for region in data[image_name1]:
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


    # Save the image with bounding boxes
    output_filename = f'bbox_{image_name}'
    path = "C:/Users/xionjing/OneDrive - Merck Sharp & Dohme LLC/Desktop/CyroEM_segment/"
    folder = path +  "bbox_prediction/" 
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(folder + "/" +output_filename, img)