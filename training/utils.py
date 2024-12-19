from matplotlib import pyplot as plt
import os


def plot_combined_losses(train_losses, val_losses):
    """Plot combined training and validation losses."""
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Across All Stages')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/loss_plot.png')
    plt.show()