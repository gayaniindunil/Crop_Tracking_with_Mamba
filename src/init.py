from pathlib import Path
import torch


print("Crop Tracking with Mamba - Initializing...")

from torch.utils.data import DataLoader

def detection_collate_fn(batch):
    """
    Combines individual images into a single batch tensor, 
    but keeps targets as a raw list of dictionaries.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Stacks images cleanly into shape: (Batch_Size, Channels, Height, Width)
    images = torch.stack(images, dim=0) 
    return images, targets



if __name__ == "__main__":
    from src.data_processor.GSD_detection import GrowthStrawberryDataset, visualize_first_training_image

    dataset = GrowthStrawberryDataset(
        images_dir=Path("data/images"),
        annotations_file=Path("data/annotations.json"),
        grid_size=(16, 16),
        num_classes=3,
    )

    print(f"Dataset initialized with {len(dataset)} samples.")
    visualize_first_training_image(dataset, save_path=Path("output/first_training_image.png"))


    # Instantiate your data pipeline like this:
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=detection_collate_fn  # Handles the variable label count
    )
