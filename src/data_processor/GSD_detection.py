
from sympy import im

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image

import matplotlib.patches as patches
import matplotlib.pyplot as plt

class GrowthStrawberryDataset(Dataset):
    def __init__(self, json_path, images_dir, image_size=(512, 512), feature_map_size=(16, 16)):
        super().__init__()
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.feature_map_size = feature_map_size

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

    def _build_targets(self, image_meta):
        feature_h, feature_w = self.feature_map_size
        class_target = torch.zeros((feature_h, feature_w), dtype=torch.long)
        box_target = torch.zeros((4, feature_h, feature_w), dtype=torch.float32)
        object_mask = torch.zeros((feature_h, feature_w), dtype=torch.float32)

        image_id = int(image_meta["id"])
        original_w = float(image_meta["width"])
        original_h = float(image_meta["height"])
        annotations = self.annotations_by_image.get(image_id, [])

        for annotation in annotations:
            x, y, w, h = annotation["bbox"]

            x = x * (self.image_size[0] / original_w)

             
            y = y * (self.image_size[1] / original_h)
            w = w * (self.image_size[0] / original_w)
            h = h * (self.image_size[1] / original_h)

            center_x = x + (w / 2.0)
            center_y = y + (h / 2.0)

            grid_x = min(feature_w - 1, max(0, int(center_x / self.image_size[0] * feature_w)))
            grid_y = min(feature_h - 1, max(0, int(center_y / self.image_size[1] * feature_h)))

            # Express the center coordinate relative to the grid cell it belongs to (0.0 to 1.0 offset)
            offset_x = (center_x / self.image_size[0] * feature_w) - grid_x
            offset_y = (center_y / self.image_size[1] * feature_h) - grid_y

            # Normalize width and height relative to the entire image size
            norm_w = w / self.image_size[0]
            norm_h = h / self.image_size[1]

            

            if object_mask[grid_y, grid_x] > 0:
                prev_area = box_target[2, grid_y, grid_x] * box_target[3, grid_y, grid_x]
                new_area = (w / self.image_size[0]) * (h / self.image_size[1])
                if new_area <= prev_area:
                    continue

            class_target[grid_y, grid_x] = 1  # Kept as 1 per your original logic
            box_target[:, grid_y, grid_x] = torch.tensor([
                offset_x,
                offset_y,
                norm_w,
                norm_h
            ], dtype=torch.float32)
            object_mask[grid_y, grid_x] = 1.0

        return class_target, box_target, object_mask

    def __getitem__(self, index):
        image_meta = self.images[index]
        image_tensor = self._load_image(image_meta["file_name"])
        class_target, box_target, object_mask = self._build_targets(image_meta)
        return image_tensor, class_target, box_target, object_mask
    


def visualize_first_training_image(dataset: GrowthStrawberryDataset, save_path: Path | None = None):
    if len(dataset) == 0:
        raise ValueError("Cannot visualize an empty dataset.")

    image_meta = dataset.images[0]
    image_path = dataset.images_dir / image_meta["file_name"]
    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f'First training image: {image_meta["file_name"]}')
    ax.axis("off")

    annotations = dataset.annotations_by_image.get(int(image_meta["id"]), [])
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        category_id = annotation.get("category_id", "?")
        track_id = annotation.get("track_id", "?")

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


def visualize_actual_grid_targets(dataset, save_path: Path | None = None):
    if len(dataset) == 0:
        raise ValueError("Cannot visualize an empty dataset.")

    # 1. Fetch raw data
    image_meta = dataset.images[0]
    image_path = dataset.images_dir / image_meta["file_name"]
    
    # 2. Resize image to exactly what the model expects
    # (Assuming self.image_size is an integer like 448 or 512)
    img_size = dataset.image_size
    image = Image.open(image_path).convert("RGB").resize(img_size)
    img_w, img_h = dataset.image_size 

    # 3. Call your internal target builder to get the processed grids
    class_target, box_target, object_mask = dataset._build_targets(image_meta)
    feature_h, feature_w = dataset.feature_map_size

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f"Processed Grid Targets for: {image_meta['file_name']}")
    
    # 4. Optional: Draw the underlying target grid lines for structural clarity
    for i in range(feature_w + 1):
        x_line = i * (img_w / feature_w)
        ax.axvline(x_line, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    for j in range(feature_h + 1):
        y_line = j * (img_h / feature_h)
        ax.axhline(y_line, color='white', linestyle='--', alpha=0.3, linewidth=0.5)



    # 5. Decode the grid targets back into pixel spaces
    for grid_y in range(feature_h):
        for grid_x in range(feature_w):
            # Only draw if a target exists in this cell
            if object_mask[grid_y, grid_x] == 1.0:
                # Extract your cell-relative targets
                offset_x, offset_y, norm_w, norm_h = box_target[:, grid_y, grid_x]

                # DECODE MATH: Reverse your cell-relative assignment formulas
                center_x = ((offset_x + grid_x) / feature_w) * img_w
                center_y = ((offset_y + grid_y) / feature_h) * img_h
                w = norm_w * img_w
                h = norm_h * img_h

                # Convert decoded center back to top-left for matplotlib
                x = center_x - (w / 2.0)
                y = center_y - (h / 2.0)

                # Draw the cell center point to see the grid anchor assignment
                ax.plot(center_x, center_y, 'ro', markersize=5) 

                # Draw the decoded target box
                rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2)
                ax.add_patch(rect)
                
                # Highlight the grid cell that owns this strawberry
                cell_left = grid_x * (img_w  / feature_w)
                cell_top = grid_y * (img_h / feature_h)
                cell_w = img_w  / feature_w
                cell_h = img_h / feature_h
                cell_rect = patches.Rectangle((cell_left, cell_top), cell_w, cell_h, fill=True, color="red", alpha=0.2)
                ax.add_patch(cell_rect)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=160, bbox_inches="tight")
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
        feature_map_size=FEATURE_MAP_SIZE,
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        images, class_targets, box_targets, object_masks = batch
        print("Images shape:", images.shape)
        print("Class targets shape:", class_targets.shape)
        print("Box targets shape:", box_targets.shape)
        print("Object masks shape:", object_masks.shape)
        break

    visualize_first_training_image(
        dataset,
        save_path=Path.cwd() / "first_training_image_visualization.png",
    )


    visualize_actual_grid_targets(
        dataset,
        save_path=Path.cwd() / "visualize_actual_grid_targets.png",
    )

