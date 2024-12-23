import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from datasets import load_dataset
from src.dataloader import create_no_curriculum_dataloader
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


# 3. No Curriculum Training Function
def train_and_save_model_no_curriculum(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, model_output_dir, loss_output_dir):
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(loss_output_dir, exist_ok=True)  

    # Train the model
    stage_name = "No Curriculum"
    train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage_name)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(model_output_dir, "resnet18_no_curriculum_final.pth"))

    # Save losses
    loss_file = "no_curriculum_losses.json"
    with open(os.path.join(loss_output_dir, loss_file), "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, stage_name)

    return train_losses, val_losses

# 4. Main Training Loop (Curriculum)
def train_no_curriculum(batch_size, num_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir):
    # 4.1 Load dataset
    train_data = load_tiny_imagenet_data(split="train")
    val_data = load_tiny_imagenet_data(split="valid")

    # 4.2 Create No curriculum DataLoaders
    train_dataloader = create_no_curriculum_dataloader(train_data, batch_size)
    val_dataloader = create_no_curriculum_dataloader(val_data, batch_size)

    # 4.3 Create ResNet18 model
    # Initialize model, criterion, optimizer, and scheduler
    model = create_resnet18(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
    #device = torch.device("mps")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * num_epochs)

    # Train and save model
    train_and_save_model_no_curriculum(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, model_output_dir, loss_output_dir)


# 5. Run Script
def main():
    # Hyperparameters
    batch_size = BATCH_SIZE
    num_epochs = EPOCHS
    learning_rate = LEARNING_RATE
    num_classes = NUM_CLASSES  # Tiny ImageNet has 200 classes
    
    model_output_dir = "outputs/models/no_curriculum/"
    loss_output_dir = "outputs/loss_logs/no_curriculum"

    train_no_curriculum(batch_size, num_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir)

if __name__ == "__main__":
    main()
