import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

# def train_one_epoch(model, dataloader, optimizer, device):
#     model.train()
#     total_loss = 0.0
#     for images, targets in dataloader:
#         images = [image.to(device) for image in images]
#         targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

#         loss_dict = model(images, targets)
#         loss = sum(loss_value for loss_value in loss_dict.values())

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += float(loss.item())

#     return total_loss / max(1, len(dataloader))


# @torch.no_grad()
# def evaluate_one_epoch(model, dataloader, device):
#     model.eval()
#     total_loss = 0.0
#     for images, targets in dataloader:
#         images = [image.to(device) for image in images]
#         targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

#         loss_dict = model(images, targets)
#         loss = sum(loss_value for loss_value in loss_dict.values())
#         total_loss += float(loss.item())

#     return total_loss / max(1, len(dataloader))




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
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_box = nn.SmoothL1Loss(reduction="none")
        # criterion_box = torch.nn.SmoothL1Loss(reduction='none')


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
    
    def compute_grid_accuracy(self, pred_classes, class_target, object_mask):
        '''
        Args:
            pred_classes: Model logits [Batch, NumClasses, 16, 16]
            class_target: Ground truth indices [Batch, 16, 16]
            object_mask: Binary mask [Batch, 16, 16] (1.0 for object, 0.0 for background)
        '''
        # Get the predicted class indices by taking the argmax along the channel dimension
        preds = torch.argmax(pred_classes, dim=1) # Shape: [Batch, 16, 16]
        
        # Create a boolean map of correct predictions
        correct = (preds == class_target).float()
        
        # 1. Calculate accuracy on cells containing actual objects (Foreground)
        fg_total = object_mask.sum().item()
        fg_correct = (correct * object_mask).sum().item()
        fg_acc = (fg_correct / fg_total) if fg_total > 0 else 0.0
        
        # 2. Calculate accuracy on background cells
        bg_mask = 1.0 - object_mask
        bg_total = bg_mask.sum().item()
        bg_correct = (correct * bg_mask).sum().item()
        bg_acc = (bg_correct / bg_total) if bg_total > 0 else 0.0
        
        return fg_acc, bg_acc

    def train_epoch(self, epoch_idx: int) -> float:
        """Runs one detection training epoch and returns the average loss."""
        self.model.train()
        running_loss = 0.0
        running_fg_acc = 0.0
        running_bg_acc = 0.0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx}")

        for images, targets in progress_bar:

            if isinstance(images, list):
                images = torch.stack(images).to(self.device)
            else:
                images = images.to(self.device)

            if isinstance(targets, list):
                class_target = torch.stack([t["class_target"] for t in targets]).to(self.device)
                box_target = torch.stack([t["box_target"] for t in targets]).to(self.device)
                object_mask = torch.stack([t["object_mask"] for t in targets]).to(self.device)
            else:
                class_target = targets["class_target"].to(self.device)
                box_target = targets["box_target"].to(self.device)
                object_mask = targets["object_mask"].to(self.device)

            pred_classes, pred_boxes = self.model(images)
            loss_cls = self.criterion_cls(pred_classes, class_target)


            box_loss_map = self.criterion_box(pred_boxes, box_target).mean(dim=1)
            loss_box = (box_loss_map * object_mask).sum() / object_mask.sum().clamp(min=1.0)
            # if not isinstance(loss_cls, dict):
            #     raise TypeError(
            #         "Detection models must return a dictionary of losses during training."
            #     )            

            total_loss = loss_cls + loss_box
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            fg_acc, bg_acc = self.compute_grid_accuracy(pred_classes, class_target, object_mask)
            running_fg_acc += fg_acc
            running_bg_acc += bg_acc
            running_loss += float(total_loss.item())
            
            # Show immediate batch accuracy in progress bar (multiplied by 100 for percentage)
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}", 
                "ObjAcc": f"{fg_acc * 100:.1f}%",
                "BgAcc": f"{bg_acc * 100:.1f}%"
            })
            
        num_batches = max(1, len(self.train_dataloader))
        epoch_loss = running_loss / num_batches
        epoch_fg_acc = running_fg_acc / num_batches
        epoch_bg_acc = running_bg_acc / num_batches
        
        print(f"\n>> [TRAIN END] Avg Loss: {epoch_loss:.4f} | Avg Crop Accuracy: {epoch_fg_acc*100:.2f}% | Avg Bg Accuracy: {epoch_bg_acc*100:.2f}%")
        return epoch_loss
            
    
    @torch.no_grad()
    def evaluate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        running_fg_acc = 0.0
        running_bg_acc = 0.0

        progress_bar = tqdm(self.val_dataloader, desc="Validating")
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images = images.to(self.device)

                class_target = targets["class_target"].to(self.device) # Shape: [Batch, 16, 16]
                box_target = targets["box_target"].to(self.device)     # Shape: [Batch, 4, 16, 16]
                object_mask = targets["object_mask"].to(self.device)

                pred_classes, pred_boxes = self.model(images)

                loss_cls = self.criterion_cls(pred_classes, class_target)

                raw_box_loss = self.criterion_box(pred_boxes, box_target).mean(dim=1)
                loss_box = (raw_box_loss * object_mask).sum() / object_mask.sum().clamp(min=1.0)
                total_loss = loss_cls + loss_box

                # --- CALCULATE ACCURACY ---
                fg_acc, bg_acc = self.compute_grid_accuracy(pred_classes, class_target, object_mask)
                running_fg_acc += fg_acc
                running_bg_acc += bg_acc
                running_loss += float(total_loss.item())
                
                progress_bar.set_postfix({
                    "Val_Loss": f"{total_loss.item():.4f}", 
                    "Val_ObjAcc": f"{fg_acc * 100:.1f}%"
                })
                            
                num_batches = max(1, len(self.val_dataloader))
                val_loss = running_loss / num_batches
                val_fg_acc = running_fg_acc / num_batches
                val_bg_acc = running_bg_acc / num_batches
                
            print(f">> [VAL END] Avg Loss: {val_loss:.4f} | Avg Crop Accuracy: {val_fg_acc*100:.2f}% | Avg Bg Accuracy: {val_bg_acc*100:.2f}%")
        return val_loss


    def fit(self, max_epochs: int, save_dir: Path | str, patience: int = 5):
        """Executes full multi-epoch training pipeline execution steps with early stopping and auto-resume."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training on device: {self.device}")
        
        # State tracking trackers
        train_losses = []
        val_losses = []
        start_epoch = 1
        self.best_val_loss = float('inf')
        patience_counter = 0

        # Search for any pre-saved epoch snapshot weights
        checkpoints = list(save_dir.glob("detector_epoch_*.pt"))
        if checkpoints:
            epochs_found = [int(p.stem.split('_')[-1]) for p in checkpoints]
            latest_epoch = max(epochs_found)
            latest_checkpoint = save_dir / f"detector_epoch_{latest_epoch}.pt"
            
            print(f"\n[RESUME DETECTED] Restoring engine weights from previous training stage -> Epoch {latest_epoch}")
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            
            # Load active states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Reconstruct training historical arrays
            train_losses = checkpoint.get('train_losses', [])[:latest_epoch]
            val_losses = checkpoint.get('val_losses', [])[:latest_epoch]
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            
            # Fast-forward your iterator slot index
            start_epoch = latest_epoch + 1
            print(f"Resuming training execution sequence safely from Epoch {start_epoch}\n")

        for epoch in range(start_epoch, max_epochs + 1):
            avg_train_loss = self.train_epoch(epoch)
            train_losses.append(avg_train_loss)
            print(f"Summary -> Epoch [{epoch}/{max_epochs}] | Average Loss: {avg_train_loss:.5f}")

            val_loss = self.evaluate_one_epoch()
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.5f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0 # Reset countdown timer flag
                
                # Save best structural checkpoints
                best_model_path = save_dir / "best_mamba_crop_detector.pth"
                torch.save(self.model.state_dict(), best_model_path)
                print(f"--> Saved a new best model checkpoint configuration with Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"--> No validation improvement. Early stopping patience step: {patience_counter}/{patience}")

            # Save periodic verification weights checkpoints (storing tracking arrays for resume)
            if epoch % 5 == 0 or epoch == max_epochs or patience_counter >= patience:
                checkpoint_path = save_dir / f"detector_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': self.best_val_loss,
                    'patience_counter': patience_counter,
                }, checkpoint_path)
                print(f"Saved training framework structural weights snapshot: {checkpoint_path}")

            # Trigger immediate loop termination if patience ran completely out
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation loss stalled for {patience} straight epochs. Halting pipeline.")
                break


        if train_losses and val_losses:
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
            plt.close() # Safely clear matplot workspace figures context

        print(f"Training complete.")
        if train_losses and val_losses:
            print(f"Final Train Loss: {train_losses[-1]:.4f}, Final Val Loss: {val_losses[-1]:.4f}")

        final_model_path = Path.cwd() / "mamba_detector_final.pth"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
