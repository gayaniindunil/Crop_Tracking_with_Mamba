# this is to visualize the crop detection and tracking results
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class CropDetectionVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset

    def visualize_actual_grid_targets(self, img_id, save_path: Path | None = None):
        # 1. Fetch raw data
        image, target = self.dataset.__getitem__(img_id)
        image_meta = self.dataset.images[img_id]
        image_path = self.dataset.images_dir / image_meta["file_name"]

        # 2. Resize image to exactly what the model expects
        img_size = self.dataset.image_size
        image = Image.open(image_path).convert("RGB")#.resize(img_size)

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

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(Path(save_path), dpi=160, bbox_inches="tight")
        else:
            plt.show()

    def visualize_crop_life_cycle(self, dataset, track_id, save_path: Path | None = None):
        # This function can be implemented to visualize the crop's life cycle
        # show a all the images of the same crop with same track ID across different time points,
        #  with bounding boxes and track IDs
        annotations = []
        for img_id, image_meta in enumerate(dataset.images):


        