from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split

# Add the project 'src' directory path explicitly to runtime path lookups
src_path = Path(__file__).resolve().parents[1]  # Points to .../Crop_Tracking_with_Mamba/src
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from visualizer.visualizationUtils import VisualizationUtils
from data_processor.detection_dataset3 import GrowthStrawberryDataset
from model.mamba_detector import MambaCropDetector
from model.mockModel import MockModel
from engine.trainer import DetectionTrainer, train_one_epoch, evaluate_one_epoch



print("Crop Tracking with Mamba - Initializing...")


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
    test_env = True  # Set to False when running in production
    GSD_JSON_PATH = Path("D:/Work_Space/D/1_AUT_MPhil_research/Experiments/crop_tracker/src/crop_tracker/Dataset/GSD-Annotations/RGB-1-2021.json")
    GSD_IMAGES_DIR = Path( "D:/Work_Space/D/1_AUT_MPhil_research/Experiments/crop_tracker/src/crop_tracker/Dataset/GSD-Images/GSD-Images-2021/RGB-1-2021/img/") 
    INPUT_IMAGE_SIZE = (512, 512)
    backbone_model_name = "nvidia/MambaVision-S-1K"  # Example backbone model name
    val_split = 0.2  # 20% for validation
    num_epochs = 1  # For testing, keep it low. Increase for actual training.

    dataset = GrowthStrawberryDataset(
        json_path=GSD_JSON_PATH,
        images_dir=GSD_IMAGES_DIR,
        image_size=INPUT_IMAGE_SIZE,
    )

    print(f"Dataset initialized with {len(dataset)} samples.")
    dataset.visualize_actual_grid_targets(img_idx=100, save_path=Path("outputs/visualize_100.png"))

    track_sequence = dataset.get_images_and_boxes_for_track_id(track_id=77)
    print(f"\nTrack 77 sequence found across {len(track_sequence)} total frames.")
    if track_sequence:
        print(f"First frame appearance metadata: {track_sequence[0]}")

    
    VisualizationUtils.generate_track_video(
        track_sequence=track_sequence,
        output_video_path=Path.cwd() / "outputs/track_77_video.mp4",
        fps=5,  # Control video playback speed here
        display_size=INPUT_IMAGE_SIZE
    )

    # Split dataset into train (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Instantiate your data pipeline like this:
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn  # Handles the variable label count
        
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0,
        collate_fn=detection_collate_fn  # Handles the variable label count
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if test_env:
        # Quick sanity check on the first batch
        for images, targets in train_loader:
            print(f"Batch Images shape: {images.shape}")
            print(f"Batch Targets length: {len(targets)}")
            break  # Only check the first batch for testing
            model = MockModel(dim=512)
    else:
        model = MambaCropDetector(backbone=backbone_model_name, num_classes=2)

    model = model.to(device)

    print(f"Starting training on {len(train_dataset)} samples (train) and {len(val_dataset)} samples (val).")
    print(f"Training for {num_epochs} epochs...")
    print(f"Device: {device}")

    trainer = DetectionTrainer(
        model=model,
        train_dataset=dataset,
        batch_size=4,
        lr=1e-4,
        device=device
    )
    trainer.fit(max_epochs=num_epochs, save_dir=Path.cwd() / "checkpoints")



