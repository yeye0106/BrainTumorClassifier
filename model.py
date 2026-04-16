import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES


def get_model():
    # 彻底回归 ResNet34
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    return model