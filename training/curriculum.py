import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_no_curriculum_dataloader
from training.train import train_model
from training.utils import plot_learning_curves
from models.resnet18 import get_model
from configuration.setting import AGES
from training import utils as ut

def train_curriculum_model(
    batch_size,
    total_epochs,
    learning_rate,
    num_classes,
    model_output_dir,
    loss_output_dir,
    model_names,
    ages=AGES,
    user_epoch_map=None,
    curriculum_type="default",
    dataset_loader=ut.load_data,
    dataloader_creator=None
):

    # 1. Load datasets
    train_data = dataset_loader(split="train")
    val_data = dataset_loader(split="valid")

    # 2. Create curriculum-based DataLoaders
    curriculum_dataloaders = dataloader_creator(
        dataset=train_data,
        ages=ages,
        batch_size=batch_size
    )
    val_dataloader = create_no_curriculum_dataloader(val_data, batch_size)

    # 3. Train each model
    for model_name in model_names:
        print(f"Starting training for model: {model_name} with {curriculum_type} curriculum.")

        # Initialize the model
        model = get_model(model_name, num_classes=num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Define loss function, optimizer, and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        # Define stage-epoch mapping
        stage_epochs = ut.create_stage_epoch_mapping(
            ages=ages,
            total_epochs=total_epochs,
            user_mapping=user_epoch_map
        )

        # Track overall losses
        overall_train_losses = []
        overall_val_losses = []

        # Train in stages
        for age in ages:
            stage_name = f"{curriculum_type}_Age_{age}mo_{model_name}"
            dataloader = curriculum_dataloaders[age]
            epochs_for_this_stage = stage_epochs[age]

            print(f"[{stage_name}] -> Training for {epochs_for_this_stage} epoch(s).")

            train_losses, val_losses = train_model(
                model=model,
                dataloader=dataloader,
                val_dataloader=val_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=epochs_for_this_stage,
                stage_name=stage_name
            )

            overall_train_losses.extend(train_losses)
            overall_val_losses.extend(val_losses)

        # Save model and loss logs
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(loss_output_dir, exist_ok=True)

        model_save_path = os.path.join(model_output_dir, f"{model_name}_{curriculum_type}_final.pth")
        torch.save(model.state_dict(), model_save_path)

        loss_log_path = os.path.join(loss_output_dir, f"{model_name}_{curriculum_type}_losses.json")
        with open(loss_log_path, "w") as f:
            json.dump({"train_losses": overall_train_losses, "val_losses": overall_val_losses}, f)

        # Plot learning curves
        plot_learning_curves(overall_train_losses, overall_val_losses, f"{curriculum_type}_{model_name}")

        print(f"Training completed for model: {model_name}. Model saved to {model_save_path}.")

