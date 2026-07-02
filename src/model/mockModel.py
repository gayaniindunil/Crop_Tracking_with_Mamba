from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MockDetectionModel(nn.Module):
    """A tiny CPU-friendly detection stub for exercising the training loop.

    The model accepts a batch of images and target dictionaries, produces a
    small set of detection-style predictions, and returns a loss dictionary that
    matches the trainer's expectations.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Linear(hidden_dim, 4)

    def _stack_images(self, images: torch.Tensor | Iterable[torch.Tensor]) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            return images
        return torch.stack(list(images), dim=0)

    def forward(self, images, targets=None):
        batch = self._stack_images(images)
        features = self.backbone(batch).flatten(1)

        class_logits = self.class_head(features)
        box_predictions = self.box_head(features)

        if targets is None:
            return {
                "pred_logits": class_logits,
                "pred_boxes": box_predictions,
            }

        device = batch.device
        class_targets = []
        box_targets = []

        for target in targets:
            boxes = target.get("boxes")
            labels = target.get("class_target")

            if boxes is None or boxes.numel() == 0:
                box_targets.append(torch.zeros(4, device=device, dtype=box_predictions.dtype))
                class_targets.append(torch.zeros((), device=device, dtype=torch.long))
                continue

            box_targets.append(boxes[0].to(device=device, dtype=box_predictions.dtype))
            if labels is None or labels.numel() == 0:
                class_targets.append(torch.zeros((), device=device, dtype=torch.long))
            else:
                class_targets.append(labels[0].to(device=device, dtype=torch.long).clamp(0, self.num_classes - 1))

        target_boxes = torch.stack(box_targets, dim=0)
        target_classes = torch.stack(class_targets, dim=0)

        loss_classifier = F.cross_entropy(class_logits, target_classes)
        loss_box_reg = F.smooth_l1_loss(box_predictions, target_boxes)
        loss_total = loss_classifier + loss_box_reg

        return {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
            "loss_total": loss_total,
        }


# Backwards-compatible name used by the current init script.
MockModel = MockDetectionModel
