import os
import json
import torch
import numpy as np
from matplotlib import pyplot as plt
from src.models import get_model
from setting import DEVICE, NUM_CLASSES, MODELS

def load_model(model_name, model_path):
    """
    Load a trained model from the saved model file.
    """
    model = get_model(model_name, num_classes=NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader):
    """
    Evaluate the model accuracy on the given DataLoader.
    """
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    return accuracy

def plot_accuracy_histogram(accuracies, output_path):
    """
    Plot a histogram of model accuracies.
    """
    plt.figure()
    plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_learning_curves(loss_logs, output_path):
    """
    Plot training and validation loss curves for all models.
    """
    plt.figure(figsize=(12, 8))

    for model_name, losses in loss_logs.items():
        train_losses = losses['train_losses']
        val_losses = losses['val_losses']
        plt.plot(train_losses, label=f'{model_name} - Train Loss')
        plt.plot(val_losses, label=f'{model_name} - Val Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Directories
    model_dirs = {
        "no_curriculum": "outputs/models/no_curriculum/",
        #"curriculum": "outputs/models/curriculum/",
        "layerwise": "outputs/models/layerwise",
        "layerwise_color": "outputs/models/layerwise_color",
        "layerwise_acuity": "outputs/models/layerwise_acuity",
        #"color_perception": "outputs/models/color_perception/",
        #"visual_acuity": "outputs/models/visual_acuity/"
    }

    loss_dirs = {
        "no_curriculum": "outputs/loss_logs/no_curriculum/",
        #"curriculum": "outputs/loss_logs/curriculum/",
        "layerwise": "outputs/loss_logs/layerwise",
        "layerwise_color": "outputs/loss_logs/layerwise_color",
        "layerwise_acuity": "outputs/loss_logs/layerwise_acuity",
        #"color_perception": "outputs/loss_logs/color_perception/",
        #"visual_acuity": "outputs/loss_logs/visual_acuity/"
    }

    output_dir = "outputs/comparisons/"
    os.makedirs(output_dir, exist_ok=True)

    # Load validation dataset
    from datasets import load_dataset
    from src.dataloader import create_no_curriculum_dataloader
    val_data = load_dataset("zh-plus/tiny-imagenet", split="valid")
    val_dataloader = create_no_curriculum_dataloader(val_data, batch_size=128)

    # Evaluate models
    accuracies = {}
    loss_logs = {}

    for category, model_dir in model_dirs.items():
        for model_name in MODELS:
            # Load model
            model_path = os.path.join(model_dir, f"{model_name}_{category}_final.pth")
            model = load_model(model_name, model_path)

            # Evaluate accuracy
            accuracy = evaluate_model(model, val_dataloader)
            accuracies[f"{model_name} ({category})"] = accuracy

            # Load loss logs
            loss_file = os.path.join(loss_dirs[category], f"{model_name}_{category}_losses.json")
            with open(loss_file, "r") as f:
                loss_logs[f"{model_name} ({category})"] = json.load(f)

    # Plot accuracy histogram
    accuracy_plot_path = os.path.join(output_dir, "model_accuracies_histogram.png")
    plot_accuracy_histogram(accuracies, accuracy_plot_path)

    # Plot learning curves
    learning_curves_plot_path = os.path.join(output_dir, "learning_curves_comparison.png")
    plot_learning_curves(loss_logs, learning_curves_plot_path)

    print("Comparison plots saved in", output_dir)

if __name__ == "__main__":
    main()
