import cv2
import json
import os
import re
import shutil

# Load the annotations
with open('C:/Users/xionjing/OneDrive - Merck Sharp & Dohme LLC/Desktop/CyroEM_segment/via_project_1Mar2023_full.json', 'r') as f:
    data = json.load(f)

# dict_keys(['_via_settings', '_via_img_metadata', 
# '_via_attributes', '_via_data_format_version', 
# '_via_image_id_list'])


def split_filename(filename):
    # filename = "HPV_T35_Rd2_FAP_NIS_T35_Rd2_FAP_Arm_4_T3E06C~1_downsized.png"
    # filename = "HPV_T39_Rd2_FAP_NIS_T39_Rd2_FAP_Arm_A_T39_Rd2_FAP_Arm_A_10-2-2-2_73000x_downsized.png"
    # Use a regular expression to split the filename
    match = re.match(r'HPV_(.*?)_FAP_NIS_(T\d+_Rd2_FAP_Arm_[\dA-Z])_(.*?)_downsized.png', filename)

    if match:
        folder_name = match.group(2)
        file_name = match.group(3) + ".png"

        return folder_name, file_name

    else:
        print("Filename format not recognized.")
        return None, None


# Loop over all images
res = []
found_pics = []
unfound_pics = []
found_pics_dict = []
for img_metadata in data["_via_img_metadata"].values():
    res.append(img_metadata["filename"])
    folder_name, file_name = split_filename(img_metadata["filename"])
    print(folder_name)
    print(file_name)
    path = "C:/Users/xionjing/OneDrive - Merck Sharp & Dohme LLC/Desktop/CyroEM_segment/"
    img = cv2.imread(path + folder_name + "/" + file_name)

    # Make sure the image was successfully loaded
    if img is None:
        print(f'Failed to load image: {file_name}')
        unfound_pics.append(file_name)
        continue
    else:
        found_pics.append(file_name)

    
    # If file_name's length is <= 16, prepend folder_name
    if len(file_name) <= 16:
        new_file_name = f'{folder_name}_{file_name}'
    else:
        new_file_name = file_name
    
    # Copy the image to the All_Pictures folder
    destination_folder = "C:/Users/xionjing/OneDrive - Merck Sharp & Dohme LLC/Desktop/CyroEM_segment/All_label_Pictures"
    #shutil.copy(path + folder_name + "/" + file_name, destination_folder + "/" + new_file_name)
    
    annotations = []
    # Loop over all regions (bounding boxes)
    for region in img_metadata["regions"]:
        shape_attr = region["shape_attributes"]
        region_attributes = region["region_attributes"]
        # Get the bounding box coordinates and scale up by a factor of 10
        x = shape_attr["x"] * 10
        y = shape_attr["y"] * 10
        width = shape_attr["width"] * 10
        height = shape_attr["height"] * 10

        # Append this annotation to the list
        annotations.append({
            "type": region_attributes["class"],
            "bbox": [x, y, width, height]
        })
        
        # Set color according to class
        class_name = region_attributes["class"]
        if class_name == "virus":
            color = (0, 255, 0)  # Green for virus
        elif class_name == "ice":
            color = (255, 0, 0)  # Red for ice
        elif class_name == "unsure":
            color = (0, 0, 255)  # Blue for unsure
        else:
            color = (21, 255, 255)  # Blue for other classes



        # Draw the bounding box on the image
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        thickness = 2  # Thickness of 2 px
        cv2.rectangle(img, top_left, bottom_right, color, thickness)

    # Add the image and its annotations to the dictionary
    found_pics_dict.append({
        "filename": new_file_name,
        "annotations": annotations
    })

    # Save the image with bounding boxes
    output_filename = f'bbox_{file_name}'
    folder = path +  "bbox/" + folder_name 
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(folder + "/" +output_filename, img)

print(res)
print(f'Found pictures: {found_pics}')
print(f'Unfound pictures: {unfound_pics}')
with open('found_pics.json', 'w') as f:
    json.dump(found_pics_dict, f)