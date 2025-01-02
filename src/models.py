import torch
import torch.nn as nn
from torchvision import models
# from setting import DEVICE

def create_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# def create_vgg16(num_classes):
#     model = models.vgg16(weights=None)
#
#     # Override the forward method to handle adaptive pooling on CPU
#     class VGG16Custom(nn.Module):
#         def __init__(self, original_model):
#             super().__init__()
#             self.features = original_model.features
#             self.avgpool = original_model.avgpool
#             self.classifier = original_model.classifier
#             self.num_classes = num_classes
#             self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)
#
#         def forward(self, x):
#             x = self.features(x)
#             if x.device.type == DEVICE:  # If using MPS, move tensor to CPU for pooling
#                 x = x.cpu()
#                 x = self.avgpool(x)  # Perform pooling on CPU
#                 x = x.to(DEVICE)  # Move tensor back to MPS
#             else:
#                 x = self.avgpool(x)
#             x = x.view(x.size(0), -1)
#             x = self.classifier(x)
#             return x
#
#     return VGG16Custom(model)
#
# def create_alexnet(num_classes):
#     model = models.alexnet(weights=None)
#     model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
#     return model

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        return create_resnet18(num_classes)
    # elif model_name == "vgg16":
    #     return create_vgg16(num_classes)
    # elif model_name == "alexnet":
    #     return create_alexnet(num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
