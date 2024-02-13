from itertools import chain
import json
import os
import shutil
from tqdm import tqdm
from colorama import Fore
import yaml
import numpy as np
from typing import List, Dict

class COCODataset:
    def __init__(self, images_dirpath: str, annotations_filepath: str, dataset_dirpath: str, length: int = 1746):
        self.train_size = None
        self.val_size = None
        self.length = length
        self.labels_counter = None
        self.normalize = True
        
        self.images_dirpath = images_dirpath
        self.annotations_filepath = annotations_filepath
        self.dataset_dirpath = dataset_dirpath
        self.train_dirpath =  os.path.join(self.dataset_dirpath, "train")
        self.val_dirpath =  os.path.join(self.dataset_dirpath, "val")
        self.config_path = os.path.join(self.dataset_dirpath, "coco.yaml")

        self.samples = self.parse_jsonl(annotations_filepath)
        self.classes_dict = {
            "virus": 0,
            "ice": 1,
            "unsure": 2
        }
        self.classes = [0,1,2]

    def __prepare_dirs(self) -> None:
        if not os.path.exists(self.dataset_dirpath):
            os.makedirs(os.path.join(self.train_dirpath, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.train_dirpath, "labels"), exist_ok=True)
            os.makedirs(os.path.join(self.val_dirpath, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.val_dirpath, "labels"), exist_ok=True)
        else:
            raise RuntimeError("Dataset already exists!")

    def __define_splitratio(self) -> None:
        self.train_size = round(self.length * self.train_size)
        self.val_size = self.length - self.train_size
        assert self.train_size + self.val_size == self.length

    def parse_jsonl(self, path: str) -> List[Dict]:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            samples = [{'id': key, **value} for key, value in data.items()]
        return samples

    def __define_paths(self, split: str) -> dict:
        data_path = self.val_dirpath if split == 'val' else self.train_dirpath
        return {
            "images": os.path.join(data_path, "images"),
            "labels": os.path.join(data_path, "labels")
        }

    @staticmethod
    def __get_label_path(paths_dict: dict, identifier: str) -> str:
        return os.path.join(
            paths_dict["labels"],
            f"{identifier.split('.')[0]}.txt"
        )

    @staticmethod
    def __get_image_path(paths_dict: dict, identifier: str) -> str:
        return os.path.join(
            paths_dict["images"],
            f"{identifier}"
        )

    def __copy_image(self, dst_path: str, identifier: str) -> str:
        shutil.copyfile(
            os.path.join(self.images_dirpath, f"{identifier}"),
            dst_path
        )

    def __copy_label(self, annotations: list, dst_path: str) -> None:
        with open(dst_path, "w") as file:
            for annotation in annotations:
                bbox = annotation["bbox"]
                label = self.classes_dict[annotation["type"]]
                if label in self.classes:
                    if bbox:
                        if self.normalize:
                            # Normalizing bbox coordinates
                            bbox = [coord / 512.0 for coord in bbox]
                            bbox = [max(0, min(x, 1)) for x in bbox]
                            x, y, width, height = bbox
                            x_center = x + width / 2
                            y_center = y + height / 2
                            bbox = [x_center, y_center, width, height]
                            
                        coordinates = " ".join(map(str, bbox))
                        file.write(f"{label} {coordinates}\n")
                        self.labels_counter += 1

    def __splitfolders(self):
        for line in tqdm(
                self.samples,
                desc="Dataset creation"
        ):
            self.labels_counter = 0
            identifier = line["id"]
            annotations = line["annotations"]
            paths_dict = self.__define_paths(line["split"])

            dst_image_path = self.__get_image_path(paths_dict, identifier)
            dst_label_path = self.__get_label_path(paths_dict, identifier)

            self.__copy_image(dst_image_path, identifier)
            self.__copy_label(annotations, dst_label_path)

            if self.labels_counter == 0:
                os.remove(dst_image_path)
                os.remove(dst_label_path)

    def __count_dataset(self) -> dict:
        train_images = len(os.listdir(os.path.join(self.train_dirpath, "images")))
        train_labels = len(os.listdir(os.path.join(self.train_dirpath, "labels")))
        val_images = len(os.listdir(os.path.join(self.val_dirpath, "images")))
        val_labels = len(os.listdir(os.path.join(self.val_dirpath, "labels")))
        return {
            "train_images": train_images,
            "train_labels": train_labels,
            "val_images": val_images,
            "val_labels": val_labels
        }

    @staticmethod
    def __check_sanity(d: dict) -> None:
        assert d["train_images"] == d["train_labels"]
        assert d["val_images"] == d["val_labels"]

    @staticmethod
    def __finalizing(d: dict) -> None:
        print(f"\n\n{Fore.GREEN}Dataset creation finalized with success. Summary:")
        print(yaml.dump(d, sort_keys=False))
        print(Fore.RESET)

    def __call__(self, *args, **kwargs):
        self.__prepare_dirs()
        self.__splitfolders()
        dataset_summary = self.__count_dataset()
        self.__check_sanity(dataset_summary)
        self.__finalizing(dataset_summary)

    def get_config(self) -> None:
        with open(self.config_path, "w") as file:
            file.write("train: /app/dataset/train\n")
            file.write("val: /app/dataset/val\n\n")
            for class_name, idx in self.classes_dict.items():
                file.write(f"{class_name}: {idx}\n")

    def display_config(self) -> None:
        with open(self.config_path, "r") as file:
            print("\n\nDataset config:\n")
            print(file.read())

    def write_config(self) -> None:
        self.get_config()
        self.display_config()


# Create an instance of the COCODataset class
coco = COCODataset(
    images_dirpath="All_pictures/patches",
    annotations_filepath="All_pictures/patch_annotations.json",
    dataset_dirpath = "All_pictures/Cocodataset"
)

# Call the instance to start the conversion process
coco()