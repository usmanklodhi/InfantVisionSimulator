import torch.nn as nn
from torchvision import models

def create_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(model_name, num_classes):
    if model_name == "resnet18":
        return create_resnet18(num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
