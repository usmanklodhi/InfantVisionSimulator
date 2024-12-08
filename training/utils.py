from matplotlib import pyplot as plt
import os


def plot_learning_curves(train_losses, val_losses, stage_name):
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Learning Curves ({stage_name})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_dir}/{stage_name}_learning_curves.png')
    plt.show()