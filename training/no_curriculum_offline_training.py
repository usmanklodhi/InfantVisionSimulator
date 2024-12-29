import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.tinyimagenet_dataloader import TrainTinyImageNetDataset, TestTinyImageNetDataset
from training.train import train_model
from training.utils import plot_learning_curves
from src.models import get_model
from torchvision.transforms import Normalize
from setting import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, DEVICE


def train_and_save_model_no_curriculum(model, trainset, testset, criterion, optimizer, scheduler, num_epochs, model_output_dir, loss_output_dir, model_name):
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(loss_output_dir, exist_ok=True)

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Train the model
    stage_name = f"No Curriculum - {model_name}"
    train_losses, val_losses = train_model(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs, stage_name)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(model_output_dir, f"{model_name}_no_curriculum_final.pth"))

    # Save losses
    loss_file = f"{model_name}_no_curriculum_losses.json"
    with open(os.path.join(loss_output_dir, loss_file), "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, stage_name)

    return train_losses, val_losses


def main():
    # Paths
    model_output_dir = "outputs/models/no_curriculum/"
    loss_output_dir = "outputs/loss_logs/no_curriculum/"

    # Load datasets
    from src.tinyimagenet_dataloader import id_dict
    transform = Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
    trainset = TrainTinyImageNetDataset(id_dict=id_dict, transform=transform)
    testset = TestTinyImageNetDataset(id_dict=id_dict, transform=transform)

    # Model names
    # model_names = ["resnet18", "alexnet", "vgg16"]
    model_names = ["resnet18"]

    # Train each model
    for model_name in model_names:
        print(f"Starting training for model: {model_name}")

        # Initialize model
        model = get_model(model_name, NUM_CLASSES).to(DEVICE)

        # Loss function, optimizer, and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Train and save
        train_and_save_model_no_curriculum(
            model=model,
            trainset=trainset,
            testset=testset,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=EPOCHS,
            model_output_dir=model_output_dir,
            loss_output_dir=loss_output_dir,
            model_name=model_name,
        )


if __name__ == "__main__":
    main()
