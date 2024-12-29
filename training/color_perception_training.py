import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from src.preprocessed_dataset import PreprocessedDataset
from src.dataloader import create_color_dataloader, create_no_transform
from training.train import train_model
from training.utils import plot_learning_curves
from setting import AGES, EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, DEVICE


# Define the ResNet18 model
def create_resnet18(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Helper function to create stage-to-epoch mapping
def create_stage_epoch_mapping(ages, total_epochs):
    """
    Split the total epochs across ages.
    """
    n_stages = len(ages)
    base_epochs = total_epochs // n_stages
    remainder = total_epochs % n_stages

    stage_epoch_map = {}
    for idx, age in enumerate(ages):
        extra = 1 if idx < remainder else 0
        stage_epoch_map[age] = base_epochs + extra

    return stage_epoch_map

def train_and_save_color_perception_model(model, dataloaders, val_dataloader, criterion, optimizer, scheduler, stage_epochs, model_output_dir, loss_output_dir):
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(loss_output_dir, exist_ok=True)

    all_train_losses = []
    all_val_losses = []

    for age in stage_epochs.keys():
        epochs_for_this_stage = stage_epochs[age]
        print(f"Training Color Perception for age: {age} months for {epochs_for_this_stage} epoch(s)")
        dataloader = dataloaders[age]
        stage_name = f"Color_Perception - Age {age} months"

        train_losses, val_losses = train_model(model, dataloader, val_dataloader, criterion, optimizer, scheduler, epochs_for_this_stage, stage_name)

        # Save the model weights for the current stage
        torch.save(model.state_dict(), os.path.join(model_output_dir, f"color_perception_model_age_{age}_months.pth"))

        # Append losses for plotting
        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)

        # Save losses
        loss_file = f"color_perception_{age}_losses.json"
        with open(os.path.join(loss_output_dir, loss_file), "w") as f:
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Save the final model after all stages
    torch.save(model.state_dict(), os.path.join(model_output_dir, "color_perception_final.pth"))

    # Plot aggregated learning curves
    final_stage_name = "Color Perception Learning Curves"
    plot_learning_curves(all_train_losses, all_val_losses, final_stage_name)

    return all_train_losses, all_val_losses

def main():
    # Hyperparameters
    batch_size = BATCH_SIZE
    total_epochs = EPOCHS  # Aligning with curriculum and no_curriculum
    learning_rate = LEARNING_RATE
    num_classes = NUM_CLASSES  # Tiny ImageNet has 200 classes
    model_output_dir = "outputs/models/color_perception/"
    loss_output_dir = "outputs/loss_logs/color_perception/"

    # Load dataset
    train_data = load_dataset("zh-plus/tiny-imagenet")['train']
    val_data = load_dataset("zh-plus/tiny-imagenet")['valid']

    # Create DataLoaders for Color Perception Transform
    color_perception_dataloaders = create_color_dataloader(train_data, AGES, batch_size)

    # Validation DataLoader
    val_transform = create_no_transform()
    val_dataset = PreprocessedDataset(val_data, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create stage-to-epoch mapping
    stage_epochs = create_stage_epoch_mapping(AGES, total_epochs)

    # Initialize model, criterion, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
    model = create_resnet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_and_save_color_perception_model(model, color_perception_dataloaders, val_dataloader, criterion, optimizer, scheduler, stage_epochs, model_output_dir, loss_output_dir)

if __name__ == "__main__":
    main()
