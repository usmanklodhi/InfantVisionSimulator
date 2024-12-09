import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from models import resnet18
from src import dataloader as dl, preprocessed_dataset as ppd
from my_datasets import tiny_imagenet as ti
import os
from training import utils as ut
from training import train as tr
from torchvision.models import resnet18

# Curriculum learning and parameters
stages = ['young', 'mid', 'old']
# dataset_path = './tiny-imagenet-200'
batch_size = 128
num_epochs = 30
lr = 0.01

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_resnet18_LOCAL(num_classes=200):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    os.makedirs('outputs/models', exist_ok=True)

    data = datasets.load_dataset("zh-plus/tiny-imagenet")
    train_data, val_data = (data['train'], data['valid'])

    val_dataset = ppd.PreprocessedDataset(val_data, transform=ti.old_transform)
    val_dataloader = dl.create_dataloader_v3(val_dataset, batch_size=batch_size)

    model = get_resnet18_LOCAL().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for stage in stages:
        print(f"Starting training for stage: {stage}")
        dataloader = ti.get_data_loader(stage, train_data)
        train_losses, val_losses = tr.train_model(
            model, dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage
        )
        ut.plot_learning_curves(train_losses, val_losses, stage)

        torch.save(model.state_dict(), f'outputs/models/resnet18_{stage}.pth')


if __name__ == "__main__":
    main()