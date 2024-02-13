from .YoloV8infer import patch_bbox_from_picture, plot_bboxes
from .SAM_infer import SAM_infer, draw_mask_annotations_512, Load_SAM
from .compute_features import compute_features
import os
import pickle
import pandas as pd
import time

def analysis_one_picture(picture_path, Yolo_path, SAM_path, plot_folder = None, plot_result = True):
    """
    Main pipeline, return one annotation dict
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
        
    Sample usage:

    picture_path = '/home/xionjing/br-bards-digitalpathology/CyroEM_segmentation/Inference_pipeline/T39 Rd2 FAP Arm C_15-9-8-2_73000x.png'
    Yolo_path = "/home/xionjing/CyroEM_segment/best.pt"
    SAM_path = "/home/xionjing/segment-anything/model_checkpoint/sam_vit_h_4b8939.pth"
    plot_folder = "/home/xionjing/cuda"

    annotation = analysis_one_picture(picture_path, Yolo_path, SAM_path, plot_folder = plot_folder, plot_result = True)
    """
    results_dict = patch_bbox_from_picture(picture_path = picture_path,
                            model_path = Yolo_path)
    sam_pred = Load_SAM(ckpt = SAM_path)
    annotation = SAM_infer(results_dict = results_dict, sam_pred = sam_pred)
    if plot_result == True or plot_folder != None:
        plot_bboxes(results_dict, plot_folder)
        draw_mask_annotations_512(annotation, segment = True, num_colors = 30, saving_folder = plot_folder)
        
    return annotation

def analysis_pictures_from_folder(picture_folder_path, Yolo_path, SAM_path, 
                                  result_save_path = None, 
                                  plot_folder = None, 
                                  feature_table_path = None):
    """
    Processes all pictures in the specified folder and returns a list of annotation dicts.

    Args:
    - picture_folder_path: Path to the folder containing the pictures to process.
    - Yolo_path: Path to the YOLO model.
    - SAM_path: Path to the SAM model.
    - result_save_path: Optional. The annotation result save path for the json file.
    - plot_folder: Optional. Folder to save the plots.
    - feature_table_path: Optional. Feature from annotation result save path (csv)

    Returns:
    - List of annotation dicts.
    """
    start = time.time()
    annotations = []
    # Loop over all the files in the folder
    sam_pred = Load_SAM(ckpt = SAM_path)
    for filename in os.listdir(picture_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            picture_path = os.path.join(picture_folder_path, filename)
            
            # analysis_one_picture
            results_dict = patch_bbox_from_picture(picture_path = picture_path,
                                    model_path = Yolo_path)
            annotation = SAM_infer(results_dict = results_dict, sam_pred = sam_pred)
            if plot_folder != None:
                plot_bboxes(results_dict, plot_folder,plot = False)
                draw_mask_annotations_512(annotation, segment = True, num_colors = 30, saving_folder = plot_folder, plot = False)
                    
            annotations.append(annotation)
    
    if result_save_path:
        with open(result_save_path, 'wb') as f:
            pickle.dump(annotations, f)
        print("Annotation result saved at: " + result_save_path)

    if feature_table_path:
        df = compute_features(annotations)
        df.to_csv(feature_table_path, encoding='utf-8')
    
    print("Total used time:" + str(round(time.time()-start,2)) + " seconds.")
    return annotations

