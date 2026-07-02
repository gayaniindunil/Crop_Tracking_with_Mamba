import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

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
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        batch_size: int = 4,
        lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        A model-agnostic Object Detection Trainer framework.
        
        Args:
            model (nn.Module): The model architecture (e.g., Mamba Detector).
            train_dataloader (DataLoader): Initialized PyTorch training dataloader.
            val_dataloader (DataLoader, optional): Initialized PyTorch validation dataloader.
            batch_size (int): Data batch partition constraints.
            lr (float): Base optimizer learning rate.
            device (str): Operational execution device targeting hardware ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size
        self.lr = lr
            
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
        """Runs one detection training epoch and returns the average loss."""
        self.model.train()
        running_loss = 0.0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx}")

        for images, targets in progress_bar:
            images = [image.to(self.device) for image in images]
            targets = [
                {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in target.items()}
                for target in targets
            ]

            loss_dict = self.model(images, targets)
            if not isinstance(loss_dict, dict):
                raise TypeError(
                    "Detection models must return a dictionary of losses during training."
                )

            loss = sum(loss_value for loss_value in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return running_loss / max(1, len(self.train_dataloader))
            
    
    @torch.no_grad()
    def evaluate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0

        for images, targets in self.val_dataloader:
            images = [image.to(self.device) for image in images]
            targets = [
                {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in target.items()}
                for target in targets
            ]

            loss_dict = self.model(images, targets)
            if not isinstance(loss_dict, dict):
                raise TypeError(
                    "Detection models must return a dictionary of losses during validation."
                )

            loss = sum(loss_value for loss_value in loss_dict.values())
            running_loss += float(loss.item())

        return running_loss / max(1, len(self.val_dataloader))

    def fit(self, max_epochs: int, save_dir: Path | str):
        """Executes full multi-epoch training pipeline execution steps."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training on device: {self.device}")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, max_epochs + 1):
            avg_train_loss = self.train_epoch(epoch)
            train_losses.append(avg_train_loss)
            print(f"Summary -> Epoch [{epoch}/{max_epochs}] | Average Loss: {avg_train_loss:.5f}")


            val_loss = self.evaluate_one_epoch()
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.5f}")

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

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o", markersize=3, linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", marker="s", markersize=3, linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss Curves - Growth Strawberry GSD Dataset", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plot_path = Path.cwd() / "training_loss_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nLoss curves saved to {plot_path}")
        plt.show()

        print(f"Training complete.")
        if train_losses and val_losses:
            print(f"Final Train Loss: {train_losses[-1]:.4f}, Final Val Loss: {val_losses[-1]:.4f}")

        final_model_path = Path.cwd() / "mamba_detector_final.pth"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        

