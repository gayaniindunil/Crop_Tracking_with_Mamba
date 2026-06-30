from sympy import im

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image

import matplotlib.patches as patches
import matplotlib.pyplot as plt

# functions
# 1.gt_item // return imag, target
# 2.get all bounding box for a given image id
# 3. get all image ids and bounding boxes for a given track id 



class GrowthStrawberryDataset(Dataset):
    def __init__(self, json_path, images_dir, image_size=(512, 512)):
        super().__init__()
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.image_size = image_size

        with self.json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.images = sorted(data.get("images", []), key=lambda item: int(item["id"]))
        self.annotations_by_image = {}
        for annotation in data.get("annotations", []):
            image_id = int(annotation["image_id"])
            self.annotations_by_image.setdefault(image_id, []).append(annotation)

    def __len__(self):
        return len(self.images)
    
    def _load_image(self, file_name):
        image_path = self.images_dir / file_name
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image_array).permute(2, 0, 1) #HWC to CHW

    def __getitem__(self, idx):
        # 1. Fetch metadata
        image_meta = self.images[idx]
        image_id = int(image_meta["id"])
        
        # 2. Process image
        image = self._load_image(image_meta["file_name"])
        annotations = self.annotations_by_image.get(image_id, [])
        
        original_w = float(image_meta["width"])
        original_h = float(image_meta["height"])
        annotations = self.annotations_by_image.get(image_id, [])

        print(f"Processing image ID {image_id} with {len(annotations)} annotations.")
         
        # for annotation in annotations:
        #     print(f"Annotation: {annotation}")

        boxes = []
        category_ids = []
        track_ids = []
        
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            x = x * (self.image_size[0] / image.shape[2])
            y = y * (self.image_size[1] / image.shape[1])
            w = w * (self.image_size[0] / image.shape[2])
            h = h * (self.image_size[1] / image.shape[1])

            boxes.append([x, y, w, h])
            category_ids.append(annotation["category_id"])
            track_id = annotation.get("track_id", "?")
            track_ids.append(track_id)

        target = {
            "image_id": torch.tensor(image_id),
            "class_target": torch.tensor(category_ids, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "track_ids": torch.tensor(track_ids, dtype=torch.long),
        }

        return image, target 


    def resize_targets(self,annotations, original_w, original_h):
        resized_annotations = []
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            x = x * (self.image_size[0] / original_w)
            y = y * (self.image_size[1] / original_h)
            w = w * (self.image_size[0] / original_w)
            h = h * (self.image_size[1] / original_h)
            resized_annotations.append({
                "bbox": [x, y, w, h],
                "category_id": annotation["category_id"],
                "track_id": annotation.get("track_id", "?")
            })
        return resized_annotations
 
    # # 3. get all image ids and bounding boxes for a given track id 
    def get_images_and_boxes_for_track_id(self, track_id):
        images_and_boxes = []
        for image_meta in self.images:
            image_id = int(image_meta["id"])
            annotations = self.annotations_by_image.get(image_id, [])
            for annotation in annotations:
                if annotation.get("track_id") == track_id:
                    images_and_boxes.append((image_meta, annotation))
        return images_and_boxes


def visualize_actual_grid_targets(img_id, save_path: Path | None = None):
    # 1. Fetch raw data
    image,target = dataset.__getitem__(img_id)
    image_meta = dataset.images[img_id]
    image_path = dataset.images_dir / image_meta["file_name"]

    # 2. Resize image to exactly what the model expects
    # (Assuming self.image_size is an integer like 448 or 512)
    img_size = dataset.image_size
    image = Image.open(image_path).convert("RGB")#.resize(img_size)
    # img_w, img_h = dataset.image_size 

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f'First training image: {image_meta["file_name"]}')
    ax.axis("off")

    for i in range(len(target["boxes"])):
        x, y, w, h = target["boxes"][i]
        category_id = target["class_target"][i].item()
        track_id = target["track_ids"][i].item()

        rect = patches.Rectangle((float(x), float(y)), float(w), float(h), fill=False, edgecolor="lime", linewidth=2)
        ax.add_patch(rect)
        ax.text(
            float(x),
            max(0.0, float(y) - 6.0),
            f"class:{category_id} track:{track_id}",
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
        )

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved first training image visualization to {save_path}")
    else:
        plt.show()

    return fig, ax   

if __name__ == "__main__":
    GSD_JSON_PATH = Path("D:/Work_Space/D/1_AUT_MPhil_research/Experiments/crop_tracker/src/crop_tracker/Dataset/GSD-Annotations/RGB-1-2021.json")
    GSD_IMAGES_DIR = Path( "D:/Work_Space/D/1_AUT_MPhil_research/Experiments/crop_tracker/src/crop_tracker/Dataset/GSD-Images/GSD-Images-2021/RGB-1-2021/img/") 
    INPUT_IMAGE_SIZE = (512, 512)
    FEATURE_MAP_SIZE = (16, 16)


    dataset = GrowthStrawberryDataset(
        json_path=GSD_JSON_PATH,
        images_dir=GSD_IMAGES_DIR,
        image_size=INPUT_IMAGE_SIZE,
    )

    print(dataset.__getitem__(0))  # Fetch the first item to trigger debug prints

    visualize_actual_grid_targets(
        2,
        save_path=Path.cwd() / "visualize_image_2_targets.png",
    )