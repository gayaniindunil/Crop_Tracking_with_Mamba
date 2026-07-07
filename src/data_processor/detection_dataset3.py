from sympy import im
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from pathlib import Path

# Add the project 'src' directory path explicitly to runtime path lookups
src_path = Path(__file__).resolve().parents[1]  # Points to .../Crop_Tracking_with_Mamba/src
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from visualizer.visualizationUtils import VisualizationUtils

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
        self.feature_map_size = (image_size[0] // 32,image_size[1] // 32)

        with self.json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.images = sorted(data.get("images", []), key=lambda item: int(item["id"]))
        self.image_meta_by_id = {int(img["id"]): img for img in self.images}


        self.annotations_by_image = {}
        self.annotations_by_track = {}

        for annotation in data.get("annotations", []):
            image_id = int(annotation["image_id"])
            track_id = annotation.get("track_id")

            self.annotations_by_image.setdefault(image_id, []).append(annotation)

            if track_id is not None and track_id != "?":
                self.annotations_by_track.setdefault(int(track_id), []).append(annotation)

    def __len__(self):
        return len(self.images)
    
    def _load_image(self, file_name):
        image_path = self.images_dir / file_name
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image_array).permute(2, 0, 1) #HWC to CHW

    def resize_targets(self, annotations, original_w, original_h):
        """Converts raw coco bounding boxes to scaled [x, y, w, h] targets."""
        resized_annotations = []
        img_w, img_h = self.image_size
        
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            
            # Math Correction: Scale using the un-resized spatial dimensions
            scaled_x = x * (img_w / original_w)
            scaled_y = y * (img_h / original_h)
            scaled_w = w * (img_w / original_w)
            scaled_h = h * (img_h / original_h)
            
            # Fallback handling if a track_id value is structural string placeholder "?"
            t_id = annotation.get("track_id", -1)
            t_id = -1 if t_id == "?" else int(t_id)

            resized_annotations.append({
                "bbox": [scaled_x, scaled_y, scaled_w, scaled_h],
                "category_id": int(annotation["category_id"]),
                "track_id": t_id
            })
        return resized_annotations
    
    def __getitem__(self, idx):
        image_meta = self.images[idx]
        image_id = int(image_meta["id"])
        
        image_tensor = self._load_image(image_meta["file_name"])
        raw_annotations = self.annotations_by_image.get(image_id, [])

        original_w=image_meta["width"]
        original_h=image_meta["height"]
        feature_h, feature_w = self.feature_map_size

        class_grid = torch.zeros((feature_h, feature_w), dtype=torch.long)
        box_grid = torch.zeros((4, feature_h, feature_w), dtype=torch.float32)
        object_mask = torch.zeros((feature_h, feature_w), dtype=torch.float32)
        track_grid = torch.zeros((feature_h, feature_w), dtype=torch.long)

        for annotation in raw_annotations:
            x, y, w, h = annotation["bbox"]
            category_id = annotation["category_id"]
            track_id  = annotation["track_id"]

            x = x * (self.image_size[0] / original_w)
            y = y * (self.image_size[1] / original_h)
            w = w * (self.image_size[0] / original_w)
            h = h * (self.image_size[1] / original_h)
            
            center_x = x + (w / 2.0)
            center_y = y + (h / 2.0)

            grid_x = min(feature_w - 1, max(0, int(center_x / self.image_size[0] * feature_w)))
            grid_y = min(feature_h - 1, max(0, int(center_y / self.image_size[1] * feature_h)))

            # Overlap handling: If a cell is already occupied, keep the larger object
            if object_mask[grid_y, grid_x] > 0:
                prev_area = box_grid[2, grid_y, grid_x] * box_grid[3, grid_y, grid_x]
                new_area = (w / self.image_size[0]) * (h / self.image_size[1])
                if new_area <= prev_area:
                    continue

            class_grid[grid_y, grid_x] = int(category_id)
            box_grid[:, grid_y, grid_x] = torch.tensor([
                x / self.image_size[0],
                y / self.image_size[1],
                w / self.image_size[0],
                h / self.image_size[1],
            ], dtype=torch.float32)
            
            # Flip the object mask to 1.0 for this cell to flag an active object
            object_mask[grid_y, grid_x] = 1.0
            track_grid[grid_y, grid_x] = int(track_id)

        target = {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "class_target": class_grid,       # Shape: [16, 16]
            "box_target": box_grid,           # Shape: [4, 16, 16]
            "object_mask": object_mask,       # Shape: [16, 16]
            "track_ids": track_grid           # Shape: [16, 16] (Spatially mapped track IDs)
        }

        return image_tensor, target
        
    def __get_raw_item__(self, idx):
        image_meta = self.images[idx]
        image_id = int(image_meta["id"])
        
        image_tensor = self._load_image(image_meta["file_name"])
        raw_annotations = self.annotations_by_image.get(image_id, [])
        
        # Process bounding boxes through clean unified helper function
        processed_targets = self.resize_targets(
            raw_annotations, 
            original_w=float(image_meta["width"]), 
            original_h=float(image_meta["height"])
        )

        boxes = [t["bbox"] for t in processed_targets]
        category_ids = [t["category_id"] for t in processed_targets]
        track_ids = [t["track_id"] for t in processed_targets]

        target = {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "class_target": torch.tensor(category_ids, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "track_ids": torch.tensor(track_ids, dtype=torch.long),
        }

        return image_tensor,target

 
    def get_images_and_boxes_for_track_id(self, track_id):
        """
        Gathers tracking sequences for evaluation.
        
        Returns:
            list: List of dicts containing image paths, image IDs, and scaled target bounding boxes.
        """
        track_annotations = self.annotations_by_track.get(int(track_id), [])
        results = []
        
        for ann in track_annotations:
            image_id = int(ann["image_id"])
            img_meta = self.image_meta_by_id.get(image_id)
            if not img_meta:
                continue
                
            # Process strictly the target tracking box for this image frame
            scaled_ann = self.resize_targets(
                [ann], 
                original_w=float(img_meta["width"]), 
                original_h=float(img_meta["height"])
            )[0]
            
            results.append({
                "track_id": scaled_ann["track_id"],
                "image_id": image_id,
                "file_name": img_meta["file_name"],
                "image_path": self.images_dir / img_meta["file_name"],
                "bbox": scaled_ann["bbox"],  # Scaled [x, y, w, h] ready for visual layers
                "category_id": scaled_ann["category_id"]
            })
            
        # Ensure tracking returns in correct temporal frame order sequence
        return sorted(results, key=lambda x: x["image_id"])

    def visualize_actual_grid_targets(self, img_idx, save_path: Path | None = None):
        """Visualizes processed dataset targets directly on top of the loaded tensor image."""
        image_tensor, target = self.__get_raw_item__(img_idx)
        image_meta = self.images[img_idx]

        # Convert channels-first tensor back to standard display format (HWC)
        image_np = image_tensor.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_np)
        ax.set_title(f"Dataset Output Frame Index: {img_idx} | File: {image_meta['file_name']}")
        ax.axis("off")

        for i in range(len(target["boxes"])):
            x, y, w, h = target["boxes"][i].tolist()
            category_id = target["class_target"][i].item()
            track_id = target["track_ids"][i].item()

            rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2)
            ax.add_patch(rect)
            ax.text(
                x,
                max(0.0, y - 6.0),
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
            print(f"Saved targets visualization output to {save_path}")
            plt.close(fig)
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
    
    # 1. Verification structural check
    img, tgt = dataset[0]
    print(f"Dataset Loaded Successfully! Image shape: {img.shape}, Target keys: {list(tgt.keys())}")

    # 2. Render targets with self-scoping functional method signature
    dataset.visualize_actual_grid_targets(
        img_idx=2,
        save_path=Path.cwd() / "visualize_image_2_targets.png",
    )

    # 3. Requesting sequence map tracking information
    track_sequence = dataset.get_images_and_boxes_for_track_id(track_id=2)
    print(f"\nTrack 2 sequence found across {len(track_sequence)} total frames.")
    if track_sequence:
        print(f"First frame appearance metadata: {track_sequence[0]}")

    
    VisualizationUtils.generate_track_video(
        track_sequence=track_sequence,
        output_video_path=Path.cwd() / "track_2_video.mp4",
        fps=5,  # Control video playback speed here
        display_size=INPUT_IMAGE_SIZE
    )