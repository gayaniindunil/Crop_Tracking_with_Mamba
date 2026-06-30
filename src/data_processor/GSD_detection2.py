
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

    def __getitem__(self, idx):
        # 1. Fetch metadata
        image_meta = self.images[idx]
        image_id = int(image_meta["id"])
        
        # 2. Process image
        image_path = self.images_dir / image_meta["file_name"]
        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size
        
        # Resize to fixed target dimensions
        img_w, img_h = self.image_size
        image = image.resize((img_w, img_h))
        
        # Convert image to standard PyTorch shape: (Channels, Height, Width)
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        # 3. Process ALL annotation data cleanly
        annotations = self.annotations_by_image.get(image_id, [])
        
        boxes = []
        labels = []
        
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            
            # Rescale raw box coordinates to match the new image dimensions
            x1 = x * (img_w / original_w)
            y1 = y * (img_h / original_h)
            x2 = x1 + (w * (img_w / original_w))
            y2 = y1 + (h * (img_h / original_h))
            
            boxes.append([x1, y1, x2, y2])
            labels.append(int(annotation.get("category_id", 1)))

        # 4. Enforce standard PyTorch Tensor Formats
        if len(boxes) == 0:
            # Handle empty image fallback cleanly
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
                "image_id": torch.tensor([image_id], dtype=torch.long)
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "image_id": torch.tensor([image_id], dtype=torch.long)
            }

        return image_tensor, target


def visualize_dataset_output(dataset: GrowthStrawberryDataset, index: int = 0, save_path: Path | None = None):
    # Fetch exactly what PyTorch training loops receive
    image_tensor, target = dataset[index]
    
    # Convert image tensor back to NumPy format for matplotlib: (Height, Width, Channels)
    image_np = image_tensor.permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    ax.axis("off")
    
    boxes = target["boxes"].numpy()
    labels = target["labels"].numpy()
    
    # Iterate and draw every single labeled strawberry box
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        rect = patches.Rectangle((x1, y1), w, h, fill=False, edgecolor="lime", linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, max(0.0, y1 - 5), f"Class: {label}", color="white", bbox={"facecolor": "black", "alpha": 0.5})
        
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

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # for batch in dataloader:
    #     images, class_targets, box_targets, object_masks = batch
    #     print("Images shape:", images.shape)
    #     print("Class targets shape:", class_targets.shape)
    #     print("Box targets shape:", box_targets.shape)
    #     print("Object masks shape:", object_masks.shape)
    #     break


    visualize_dataset_output(
        dataset,
        save_path=Path.cwd() / "visualize_output2.png",
    )