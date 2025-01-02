from matplotlib import pyplot as plt
from datasets import load_dataset, load_from_disk
import os


def plot_learning_curves(train_losses, val_losses, stage_name):
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title(f"Learning Curves ({stage_name})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Across All Stages')
    plt.legend()
    plt.savefig(f'{output_dir}/{stage_name}_learning_curves.png')
    plt.close()

def load_data(split="train"):
    local_path = "./tiny-imagenet" # Change this to the path where the dataset is stored
    data = load_from_disk(local_path)
    return data[split]

def download_data(split="train"):
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

def create_stage_epoch_mapping(ages, total_epochs, user_mapping=None):
    """
    Returns a dictionary that maps each age to a specific number of epochs.
    If user_mapping is provided, use that directly (and check correctness).
    Otherwise, split total_epochs equally across ages.
    """
    if user_mapping is not None:
        # Validate sum
        if sum(user_mapping.values()) != total_epochs:
            raise ValueError(
                "Sum of user-provided stage epochs != total_epochs. "
                f"Sum = {sum(user_mapping.values())}, total_epochs = {total_epochs}"
            )
        # Validate that all ages in 'ages' are in user_mapping
        for age in ages:
            if age not in user_mapping:
                raise ValueError(f"Missing stage epoch allocation for age {age}")
        return user_mapping
    else:
        # Default: evenly split total_epochs among the ages
        n_stages = len(ages)
        base_epochs = total_epochs // n_stages
        remainder = total_epochs % n_stages

        stage_epoch_map = {}
        for idx, age in enumerate(ages):
            # Distribute any remainder (1 extra epoch) among the first 'remainder' stages
            extra = 1 if idx < remainder else 0
            stage_epoch_map[age] = base_epochs + extra

        return stage_epoch_map