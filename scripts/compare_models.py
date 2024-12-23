import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
from src.preprocessed_dataset import PreprocessedDataset
from datasets import load_dataset
from setting import DEVICE

def validate_model(model, dataloader, criterion, device):
    """Validate the model and return accuracy and average loss."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)

    accuracy = correct / total
    avg_loss = running_loss / total
    return accuracy, avg_loss

def compute_aulc(losses):
    """Compute Area Under Learning Curve (AULC)."""
    normalized_epochs = np.linspace(0, 1, len(losses))  # Normalize epochs
    return np.trapz(losses, x=normalized_epochs)  # Compute integral

def compare_models_performance(models_info, dataloader, criterion, device):
    """Compare the accuracy, final loss, and AULC of different models."""
    results = {}

    for model_name, model_path in models_info.items():
        # Load model
        model = models.resnet18(num_classes=200)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        # Validate model
        accuracy, avg_loss = validate_model(model, dataloader, criterion, device)
        results[model_name] = {
            "accuracy": accuracy,
            "final_loss": avg_loss
        }

        print(f"{model_name}: Accuracy = {accuracy:.2%}, Final Loss = {avg_loss:.4f}")

    return results

def plot_learning_curves(log_paths, output_path):
    """Plot learning curves for all models."""
    plt.figure(figsize=(12, 8))

    for model_name, log_path in log_paths.items():
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                data = json.load(f)
                train_losses = data["train_losses"]
                val_losses = data["val_losses"]
                aulc = compute_aulc(val_losses)
                plt.plot(train_losses, label=f"{model_name} - Train Loss")
                plt.plot(val_losses, linestyle="--", label=f"{model_name} - Val Loss (AULC={aulc:.4f})")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curves Comparison")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def save_accuracy_vs_loss(results, output_path):
    """Plot accuracy vs. final loss for all models."""
    plt.figure(figsize=(10, 6))

    for model_name, metrics in results.items():
        plt.scatter(metrics["final_loss"], metrics["accuracy"], label=model_name, s=100)

    plt.xlabel("Final Validation Loss")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy vs. Loss Comparison")
    plt.legend()
    plt.grid()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def save_accuracy_histogram(results, output_path):
    """Save a histogram of model accuracies."""
    plt.figure(figsize=(10, 6))
    accuracies = {model: metrics["accuracy"] for model, metrics in results.items()}
    plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.grid(axis='y')
    plt.savefig(output_path)
    plt.close()

def main():
    # Define the directories where loss logs and models are stored
    models_info = {
        "Curriculum Learning": "outputs/models/resnet18_final_curriculum.pth",
        "No Curriculum": "outputs/models/resnet18_final_non_curriculum.pth",
        #"Visual Acuity Transform": "outputs/models/visual_acuity/visual_acuity_final.pth",
        #"Color Perception Transform": "outputs/models/color_perception/color_perception_final.pth"
    }

    loss_logs = {
        "Curriculum Learning": "outputs/loss_logs/curriculum_flexible/curriculum_losses.json",
        "No Curriculum": "outputs/loss_logs/no_curriculum/no_curriculum_losses.json",
        #"Visual Acuity Transform": "outputs/loss_logs/visual_acuity/final_acuity_losses.json",
        #"Color Perception Transform": "outputs/loss_logs/color_perception/final_color_losses.json"
    }

    # Load validation dataset
    val_data = load_dataset("zh-plus/tiny-imagenet")["valid"]
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    val_dataset = PreprocessedDataset(val_data, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=250, shuffle=False, num_workers=4, pin_memory=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)

    # Criterion for loss calculation
    criterion = torch.nn.CrossEntropyLoss()

    # Compare models
    results = compare_models_performance(models_info, val_dataloader, criterion, device)

    # Plot learning curves
    #learning_curve_output = "outputs/plots/learning_curves_comparison.png"
    #plot_learning_curves(loss_logs, learning_curve_output)

    # Save accuracy vs. loss scatter plot
    scatter_output = "outputs/plots/accuracy_vs_loss.png"
    save_accuracy_vs_loss(results, scatter_output)

    # Save accuracy histogram
    accuracy_output = "outputs/plots/model_accuracy_histogram.png"
    save_accuracy_histogram(results, accuracy_output)

    # Print results
    print("\nModel Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy = {metrics['accuracy']:.2%}, Final Loss = {metrics['final_loss']:.4f}")
    #print(f"Learning curves saved to {learning_curve_output}")
    print(f"Accuracy vs. Loss plot saved to {scatter_output}")
    print(f"Accuracy histogram saved to {accuracy_output}")

if __name__ == "__main__":
    main()
