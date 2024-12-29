import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.dataloader import create_image_progression_dataloader
from training.train import train_model
from training.utils import plot_learning_curves
from setting import AGES, EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, DEVICE


# 1. Load Tiny ImageNet Data
def load_tiny_imagenet_data(split="train"):
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

# 2. Define the ResNet18 Model
def create_resnet18(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 3. Train and Save Model for Image-wise Curriculum
def train_and_save_model_imagewise_curriculum(model, dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, model_output_dir, loss_output_dir):
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(loss_output_dir, exist_ok=True)

    # Train the model
    stage_name = "Image-wise Curriculum"
    train_losses, val_losses = train_model(model, dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage_name)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(model_output_dir, "resnet18_imagewise_curriculum_final.pth"))

    # Save losses
    loss_file = "imagewise_curriculum_losses.json"
    with open(os.path.join(loss_output_dir, loss_file), "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, stage_name)

    return train_losses, val_losses

# 4. Main Training Loop for Image-wise Curriculum
def train_imagewise_curriculum(batch_size, num_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir, ages=AGES):
    # 4.1 Load dataset
    train_data = load_tiny_imagenet_data(split="train")
    val_data = load_tiny_imagenet_data(split="valid")

    # 4.2 Create image-wise curriculum DataLoader
    train_dataloader = create_image_progression_dataloader(train_data, ages, batch_size)
    val_dataloader = create_image_progression_dataloader(val_data, ages, batch_size)

    # 4.3 Create ResNet18 model
    model = create_resnet18(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
    model = model.to(device)

    # 4.4 Define criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * num_epochs)

    # 4.5 Train and save the model
    train_and_save_model_imagewise_curriculum(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, model_output_dir, loss_output_dir)

# 5. Example Entry Point
def main():
    # Hyperparameters
    batch_size = BATCH_SIZE
    num_epochs = EPOCHS
    learning_rate = LEARNING_RATE
    num_classes = NUM_CLASSES  # Tiny ImageNet has 200 classes
    ages = AGES                # e.g., [6, 9, 12]

    # Output folders
    model_output_dir = "outputs/models/imagewise_curriculum/"
    loss_output_dir = "outputs/loss_logs/imagewise_curriculum/"

    # Train with image-wise curriculum
    train_imagewise_curriculum(batch_size, num_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir, ages=ages)

if __name__ == "__main__":
    main()
