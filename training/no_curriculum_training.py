import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_no_curriculum_dataloader
from training.train import train_model
from training.utils import plot_learning_curves
from models.resnet18 import get_model
from configuration.setting import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES
from training import utils as ut

def train_no_curriculum(batch_size, num_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir, model_names):
    # 1. Load dataset
    train_data = ut.load_data(split="train")
    val_data = ut.load_data(split="valid")

    # 2. Create No Curriculum DataLoaders
    train_dataloader = create_no_curriculum_dataloader(train_data, batch_size)
    val_dataloader = create_no_curriculum_dataloader(val_data, batch_size)

    # 3. Train each model
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(loss_output_dir, exist_ok=True)

    for model_name in model_names:
        # Initialize model, criterion, optimizer, and scheduler
        model = get_model(model_name, num_classes=num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * num_epochs)

        # Train the model
        stage_name = f"No Curriculum - {model_name}"
        train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage_name)

        # Save the final model
        torch.save(model.state_dict(), os.path.join(model_output_dir, f"{model_name}_no_curriculum_final.pth"))

        # Save losses
        loss_file = f"{model_name}_no_curriculum_losses.json"
        with open(os.path.join(loss_output_dir, loss_file), "w") as f:
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

        # Plot learning curves
        plot_learning_curves(train_losses, val_losses, stage_name)

# Run Script
def main():
    batch_size = BATCH_SIZE
    num_epochs = EPOCHS
    learning_rate = LEARNING_RATE
    num_classes = NUM_CLASSES
    model_output_dir = "outputs/models/no_curriculum/"
    loss_output_dir = "outputs/loss_logs/no_curriculum/"
    model_names = ["resnet18"]

    train_no_curriculum(batch_size, num_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir, model_names)

if __name__ == "__main__":
    main()
