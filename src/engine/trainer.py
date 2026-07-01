import torch


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_value for loss_value in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_value for loss_value in loss_dict.values())
        total_loss += float(loss.item())

    return total_loss / max(1, len(dataloader))




import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Absolute imports from your module structure
from src.data_processor.detection_dataset3 import GrowthStrawberryDataset

def detection_collate_fn(batch):
    """
    Groups varying target box sizes into a list of dictionaries, 
    while stacking matching input images into a standard tensor grid.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Pack images into shape: (Batch_Size, Channels, Height, Width)
    images = torch.stack(images, dim=0) 
    return images, targets


class DetectionTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: GrowthStrawberryDataset,
        val_dataset: GrowthStrawberryDataset | None = None,
        batch_size: int = 4,
        lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        A model-agnostic Object Detection Trainer framework.
        
        Args:
            model (nn.Module): The model architecture (e.g., Mamba Detector).
            train_dataset (GrowthStrawberryDataset): Initialized PyTorch training dataset.
            val_dataset (GrowthStrawberryDataset, optional): Initialized PyTorch validation dataset.
            batch_size (int): Data batch partition constraints.
            lr (float): Base optimizer learning rate.
            device (str): Operational execution device targeting hardware ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        # Datasets & Custom Collated Dataloaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=detection_collate_fn,
            drop_last=True
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=detection_collate_fn
            )
            
        # Basic training ecosystem modules
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
    def dummy_loss_function(self, model_outputs, targets):
        """
        Placeholder loss function. Replace this with your specific loss criteria
        (e.g., GIoU loss, Cross-Entropy class loss, or tracking association loss).
        """
        # Example of unpacking variable labels safely inside an execution graph loop:
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for i, target in enumerate(targets):
            boxes = target["boxes"].to(self.device)         # Variable shape: (N, 4)
            labels = target["class_target"].to(self.device)  # Variable shape: (N,)
            
            if boxes.shape[0] == 0:
                continue # Skip background images with zero strawberries safely
                
            # Compute structural distance differentials against predictions here...
            # total_loss = total_loss + some_loss_calculation
            
        # Returning structural toy tracking loss scalar variable for framework evaluation
        return total_loss + torch.mean(model_outputs) * 0.0

    def train_epoch(self, epoch_idx: int) -> float:
        """Runs a single pass through the complete training data loader loop."""
        self.model.train()
        running_loss = 0.0
        
        # Wrapped loading bars using tqdm
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}")
        
        for images, targets in progress_bar:
            images = images.to(self.device)
            
            # Step 1: Zero parameter gradient matrices
            self.optimizer.zero_grad()
            
            # Step 2: Forward mathematical matrix pass through the architecture
            outputs = self.model(images)
            
            # Step 3: Compute localized detection penalties
            loss = self.dummy_loss_function(outputs, targets)
            
            # Step 4: Backward pass backpropagation
            loss.backward()
            
            # Step 5: Optimizer step execution parameter optimization adjustment
            self.optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        return running_loss / len(self.train_loader)

    def fit(self, max_epochs: int, save_dir: Path | str):
        """Executes full multi-epoch training pipeline execution steps."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training on device: {self.device}")
        
        for epoch in range(1, max_epochs + 1):
            avg_train_loss = self.train_epoch(epoch)
            print(f"Summary -> Epoch [{epoch}/{max_epochs}] | Average Loss: {avg_train_loss:.5f}")
            
            # Save periodic verification weights checkpoints
            if epoch % 5 == 0 or epoch == max_epochs:
                checkpoint_path = save_dir / f"detector_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_train_loss,
                }, checkpoint_path)
                print(f"Saved training framework structural weights snapshot: {checkpoint_path}")



