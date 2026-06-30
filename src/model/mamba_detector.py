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
    backbone = AutoModel.from_pretrained(model_name) # only get the feature extractor part of the model, not the classifier head
    return backbone

class MambaCropDetector(nn.Module):
    def __init__(self, backbone, num_classes = 2):
        super(MambaCropDetector, self).__init__()
        self.backbone = backbone 
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)
    def forward(self, x):
        features = self.backbone(x)[0][:, 0, :]
        logits = self.classifier(features)
        return logits

    def backward(self, loss):
        loss.backward()


# to use the model you need to load the backbone and then create an instance of the MambaCropDetector with the backbone
# then you can use the model to make predictions on your data   


        

