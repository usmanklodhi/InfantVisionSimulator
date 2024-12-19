# Train the model without curriculum learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import datetime
from src import dataloader as dl, preprocessed_dataset as ppd
import os
from training import utils as ut
from training import train as tr
from models import resnet18 as rn
from datasets import load_from_disk

batch_size = 128
num_epochs = 30
lr = 0.01

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    os.makedirs('outputs/models', exist_ok=True)
    log_dir = os.path.join("runs", f"non_curriculum_learning_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)

    data = load_from_disk("./tiny-imagenet")
    train_data, val_data = (data['train'], data['valid'])

    train_dataset = ppd.PreprocessedDataset(train_data)
    train_dataloader = dl.create_dataloader_v3(train_dataset, batch_size=batch_size)

    val_dataset = ppd.PreprocessedDataset(val_data)
    val_dataloader = dl.create_dataloader_v3(val_dataset, batch_size=batch_size)

    model = rn.get_resnet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, val_losses = [], []

    train_losses, val_losses = tr.train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        stage_name="non_curriculum",
    )

    for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
        writer.add_scalar("loss/train", t_loss, epoch)
        writer.add_scalar("loss/valid", v_loss, epoch)


    # Save final model
    torch.save(model.state_dict(), 'outputs/models/resnet18_final_non_curriculum.pth')
    writer.add_text("Model", "Final model saved: resnet18_final_non_curriculum.pth")

    # Log combined losses
    ut.plot_combined_losses(train_losses, val_losses, "non_curriculum")

    # Close TensorBoard writer
    writer.close()
