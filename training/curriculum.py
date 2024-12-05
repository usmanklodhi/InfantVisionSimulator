import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models import resnet18
from matplotlib import pyplot as plt
from scripts import plot_transformed_images as pt
from src import dataloader as dl, preprocessed_dataset as ppd
from datasets import load_dataset
from my_datasets import tiny_imagenet as ti

# Curriculum learning and parameters
stages = ['young', 'mid', 'old']
dataset_path = './tiny-imagenet-200'
batch_size = 128
num_epochs = 10
lr = 0.01

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training function
def train_model(model, dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage_name):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\n[{stage_name}] Starting Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        print(f"[{stage_name}] Epoch {epoch + 1}: Starting training phase")
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"[{stage_name}] Epoch {epoch + 1}: Processing batch {batch_idx + 1}/{len(dataloader)}")
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            print(f"[{stage_name}] Epoch {epoch + 1}: Batch {batch_idx + 1}: Forward pass completed")

            # Compute loss
            loss = criterion(outputs, targets)
            print(f"[{stage_name}] Epoch {epoch + 1}: Batch {batch_idx + 1}: Loss computed: {loss.item()}")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            print(f"[{stage_name}] Epoch {epoch + 1}: Batch {batch_idx + 1}: Backward pass and optimization completed")

            running_loss += loss.item()

        avg_train_loss = running_loss / len(dataloader)
        train_losses.append(avg_train_loss)
        print(f"[{stage_name}] Epoch {epoch + 1}: Training phase completed. Avg Train Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        print(f"[{stage_name}] Epoch {epoch + 1}: Starting validation phase")
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_dataloader):
                print(f"[{stage_name}] Epoch {epoch + 1}: Validation batch {batch_idx + 1}/{len(val_dataloader)}")
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                print(f"[{stage_name}] Epoch {epoch + 1}: Validation batch {batch_idx + 1}: Forward pass completed")

                # Compute loss
                loss = criterion(outputs, targets)
                print(
                    f"[{stage_name}] Epoch {epoch + 1}: Validation batch {batch_idx + 1}: Loss computed: {loss.item()}")

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(f"[{stage_name}] Epoch {epoch + 1}: Validation phase completed. Avg Val Loss: {avg_val_loss}")

        # Step the scheduler
        scheduler.step()
        print(f"[{stage_name}] Epoch {epoch + 1}: Scheduler step completed")

        # Epoch summary
        print(f"[{stage_name}] Epoch {epoch + 1} Summary: Train Loss = {avg_train_loss}, Val Loss = {avg_val_loss}")

    print(f"\n[{stage_name}] Training completed for all {num_epochs} epochs.")
    return train_losses, val_losses


# Plot learning curves
def plot_learning_curves(train_losses, val_losses, stage_name):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Learning Curves ({stage_name})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/figures/{stage_name}_learning_curves.png')
    plt.show()

def main():
    # Load the Hugging Face dataset
    data = datasets.load_dataset("zh-plus/tiny-imagenet")
    train_data, val_data = (data['train'], data['valid'])

    # Prepare validation dataloader
    val_dataset = ppd.PreprocessedDataset(val_data, transform=ti.old_transform)
    val_dataloader = dl.create_dataloader_v3(val_dataset, batch_size=batch_size)

    # Initialize the model
    model = resnet18.get_resnet18().to(device)

    # Define optimizer, criterion, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Train across curriculum stages
    for stage in stages:
        print(f"Starting training for stage: {stage}")
        dataloader = ti.get_data_loader(stage, train_data)
        train_losses, val_losses = train_model(
            model, dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage
        )
        plot_learning_curves(train_losses, val_losses, stage)

        # Save model after each stage
        torch.save(model.state_dict(), f'outputs/models/resnet18_{stage}.pth')


if __name__ == "__main__":
    main()