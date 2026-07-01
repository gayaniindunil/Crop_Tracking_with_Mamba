import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests


# load the pre train model
#to do that yu need to know what is the libraray of mamaba a

# in huggingface the transformers library enable to call the huggingface models through AutoModel()

# then you need to know how the basic pytorch model implementation works, this is so fucking simple 

def get_backbone(model_name):
    print(f"Loading {model_name} Backbone...")
    backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    backbone.train() # only get the feature extractor part of the model, not the classifier head
    return backbone

class MambaCropDetector(nn.Module):
    def __init__(self, backbone, num_classes=20):
        super(MambaCropDetector, self).__init__()
        self.backbone = get_backbone(backbone) # this is for mamba backbine
        self.feature_channels = 512
        self.num_classes = num_classes
        self.channel_proj = torch.nn.Conv2d(768, 512, kernel_size=1)

        self.class_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

        self.box_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 4, kernel_size=1),
        )

    def forward(self, x):
        outputs = self.backbone(x)
        # select final stage features
        stage_features = outputs[1][-1]
        stage_features = self.channel_proj(stage_features)

        class_logits = self.class_head(stage_features)
        bbox_preds = self.box_head(stage_features)
        return class_logits, bbox_preds

# to use the model you need to load the backbone and then create an instance of the MambaCropDetector with the backbone
# then you can use the model to make predictions on your data   


        

