import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import datetime
from src import dataloader as dl, preprocessed_dataset as ppd
from my_datasets import tiny_imagenet as ti
import os
from training import utils as ut
from training import train as tr
from models import resnet18 as rn
from datasets import load_from_disk


# Curriculum learning and parameters
stages = ['young', 'mid', 'old']
batch_size = 128
num_epochs = 30
lr = 0.01

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    os.makedirs('outputs/models', exist_ok=True)
    log_dir = os.path.join("runs", f"curriculum_learning_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)

    data = load_from_disk("./tiny-imagenet")
    train_data, val_data = (data['train'], data['valid'])

    val_dataset = ppd.PreprocessedDataset(val_data, transform=ti.old_transform)
    val_dataloader = dl.create_dataloader_v3(val_dataset, batch_size=batch_size)

    model = rn.get_resnet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    all_train_losses = []
    all_val_losses = []

    for stage in stages:
        print(f"Starting training for stage: {stage}")
        dataloader = ti.get_data_loader(stage, train_data)
        train_losses, val_losses = tr.train_model(
            model, dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage
        )

        # Log losses for each epoch to TensorBoard
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            writer.add_scalar(f"{stage}/Train Loss", train_loss, epoch)
            writer.add_scalar(f"{stage}/Validation Loss", val_loss, epoch)

        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)

    # Save final model
    torch.save(model.state_dict(), 'outputs/models/resnet18_final_curriculum.pth')
    writer.add_text("Model", "Final model saved: resnet18_final_curriculum.pth")

    # Log combined losses
    ut.plot_combined_losses(all_train_losses, all_val_losses)

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
