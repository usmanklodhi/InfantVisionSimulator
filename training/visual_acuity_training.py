import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from src.preprocessed_dataset import PreprocessedDataset
from src.dataloader import create_acuity_dataloader, create_no_transform
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

def train_and_save_visual_acuity_model(model, dataloaders, val_dataloader, criterion, optimizer, scheduler, stage_epochs, model_output_dir, loss_output_dir):
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(loss_output_dir, exist_ok=True)  # Directory for saving loss logs

    all_train_losses = []
    all_val_losses = []

    for age in stage_epochs.keys():
        epochs_for_this_stage = stage_epochs[age]
        print(f"Training Visual Acuity for age: {age} months for {epochs_for_this_stage} epoch(s)")
        dataloader = dataloaders[age]
        stage_name = f"Visual_Acuity - Age {age} months"

        train_losses, val_losses = train_model(
            model, dataloader, val_dataloader, criterion, optimizer, scheduler, epochs_for_this_stage, stage_name
        )

        # Save the model weights for the current stage
        torch.save(model.state_dict(), os.path.join(model_output_dir, f"visual_acuity_model_age_{age}_months.pth"))

        # Append losses for plotting
        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)

        # Save losses
        loss_file = f"visual_acuity_{age}_losses.json"
        with open(os.path.join(loss_output_dir, loss_file), "w") as f:
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Save the final model after all stages
    torch.save(model.state_dict(), os.path.join(model_output_dir, "visual_acuity_final.pth"))

    # Plot aggregated learning curves
    final_stage_name = "Visual Acuity Learning Curves"
    plot_learning_curves(all_train_losses, all_val_losses, final_stage_name)

    return all_train_losses, all_val_losses

def save_final_acuity_loss(ages, log_dir):
    """Aggregate all age-specific final_acuity_losses loss logs and save the final loss log."""
    final_losses = {"train_losses": [], "val_losses": []}

    for age in ages:
        loss_file = os.path.join(log_dir, f"visual_acuity_{age}_losses.json")
        if os.path.exists(loss_file):
            with open(loss_file, "r") as f:
                age_losses = json.load(f)
                final_losses["train_losses"].extend(age_losses["train_losses"])
                final_losses["val_losses"].extend(age_losses["val_losses"])

    # Save final aggregated losses
    final_loss_file = os.path.join(log_dir, "final_acuity_losses.json")
    with open(final_loss_file, "w") as f:
        json.dump(final_losses, f)
    print(f"Final Visual Acuity losses saved to {final_loss_file}")
    
    plot_learning_curves(final_losses["train_losses"],
                         final_losses["val_losses"],
                         "Curriculum_learning_curves"
                         )

def main():
    # Hyperparameters
    batch_size = BATCH_SIZE
    total_epochs = EPOCHS  # Aligning with curriculum and no_curriculum
    learning_rate = LEARNING_RATE
    num_classes = NUM_CLASSES  # Tiny ImageNet has 200 classes
    model_output_dir = "outputs/models/visual_acuity/"
    loss_output_dir = "outputs/loss_logs/visual_acuity/"

    # Load dataset
    train_data = load_dataset("zh-plus/tiny-imagenet")['train']
    val_data = load_dataset("zh-plus/tiny-imagenet")['valid']

    # Create DataLoaders for Visual Acuity Transform
    visual_acuity_dataloaders = create_acuity_dataloader(train_data, AGES, batch_size)

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

    train_and_save_visual_acuity_model(
        model, visual_acuity_dataloaders, val_dataloader, criterion, optimizer, scheduler, stage_epochs, model_output_dir, loss_output_dir
    )
    
    save_final_acuity_loss(AGES, loss_output_dir)

if __name__ == "__main__":
    main()
