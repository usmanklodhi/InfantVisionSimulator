import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from src.preprocessed_dataset import PreprocessedDataset
from src.models import get_model
from datasets import load_dataset
from setting import DEVICE, NUM_CLASSES


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

    for model_name, model_paths in models_info.items():
        results[model_name] = {}
        for task_name, model_path in model_paths.items():
            # Load model
            model = get_model(model_name, NUM_CLASSES)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

            # Validate model
            accuracy, avg_loss = validate_model(model, dataloader, criterion, device)
            results[model_name][task_name] = {
                "accuracy": accuracy,
                "final_loss": avg_loss
            }

            print(f"{model_name} - {task_name}: Accuracy = {accuracy:.2%}, Final Loss = {avg_loss:.4f}")

    return results


def plot_learning_curves(model_name, log_paths, output_path):
    """Plot learning curves for a specific model across all tasks."""
    plt.figure(figsize=(12, 8))

    for task_name, log_path in log_paths.items():
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                data = json.load(f)
                train_losses = data["train_losses"]
                val_losses = data["val_losses"]
                aulc = compute_aulc(val_losses)
                plt.plot(train_losses, label=f"{task_name} - Train Loss")
                plt.plot(val_losses, linestyle="--", label=f"{task_name} - Val Loss (AULC={aulc:.4f})")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Learning Curves Comparison - {model_name}")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_accuracy_vs_loss(results, model_name, output_path):
    """Plot accuracy vs. final loss for a specific model across all tasks."""
    plt.figure(figsize=(10, 6))

    for task_name, metrics in results[model_name].items():
        plt.scatter(metrics["final_loss"], metrics["accuracy"], label=task_name, s=100)

    plt.xlabel("Final Validation Loss")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Accuracy vs. Loss Comparison - {model_name}")
    plt.legend()
    plt.grid()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_accuracy_histogram(results, model_name, output_path):
    """Save a histogram of task accuracies for a specific model."""
    plt.figure(figsize=(10, 6))
    accuracies = {task: metrics["accuracy"] for task, metrics in results[model_name].items()}
    plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
    plt.xlabel("Tasks")
    plt.ylabel("Accuracy")
    plt.title(f"Task Accuracy Comparison - {model_name}")
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.grid(axis='y')
    plt.savefig(output_path)
    plt.close()


def main():
    # Define the directories where loss logs and models are stored
    models_info = {
        "resnet18": {
            "Curriculum Learning": "outputs/models/curriculum/resnet18_curriculum_final.pth",
            #"Reverse Curriculum Learning": "outputs/models/reverse_curriculum/resnet18_reverse_curriculum_final.pth",
            "No Curriculum": "outputs/models/no_curriculum/resnet18_no_curriculum_final.pth",
            "Visual Acuity Transform": "outputs/models/visual_acuity/resnet18_visual_acuity_final.pth",
            "Color Perception Transform": "outputs/models/color_perception/resnet18_color_perception_final.pth"
        },
        "alexnet": {
            "Curriculum Learning": "outputs/models/curriculum/alexnet_curriculum_final.pth",
            #"Reverse Curriculum Learning": "outputs/models/reverse_curriculum/alexnet_reverse_curriculum_final.pth",
            "No Curriculum": "outputs/models/no_curriculum/alexnet_no_curriculum_final.pth",
            "Visual Acuity Transform": "outputs/models/visual_acuity/alexnet_visual_acuity_final.pth",
            "Color Perception Transform": "outputs/models/color_perception/alexnet_color_perception_final.pth"
        },
        "vgg16": {
            "Curriculum Learning": "outputs/models/curriculum/vgg16_curriculum_final.pth",
            #"Reverse Curriculum Learning": "outputs/models/reverse_curriculum/vgg16_reverse_curriculum_final.pth",
            "No Curriculum": "outputs/models/no_curriculum/vgg16_no_curriculum_final.pth",
            "Visual Acuity Transform": "outputs/models/visual_acuity/vgg16_visual_acuity_final.pth",
            "Color Perception Transform": "outputs/models/color_perception/vgg16_color_perception_final.pth"
        }
    }

    loss_logs = {
        "resnet18": {
            "Curriculum Learning": "outputs/loss_logs/curriculum/resnet18_curriculum_losses.json",
            #"Reverse Curriculum Learning": "outputs/loss_logs/reverse_curriculum/resnet18_reverse_curriculum_losses.json",
            "No Curriculum": "outputs/loss_logs/no_curriculum/resnet18_no_curriculum_losses.json",
            "Visual Acuity Transform": "outputs/loss_logs/visual_acuity/resnet18_visual_acuity_losses.json",
            "Color Perception Transform": "outputs/loss_logs/color_perception/resnet18_color_perception_losses.json"
        },
        "alexnet": {
            "Curriculum Learning": "outputs/loss_logs/curriculum/alexnet_curriculum_losses.json",
            #"Reverse Curriculum Learning": "outputs/loss_logs/reverse_curriculum/alexnet_reverse_curriculum_losses.json",
            "No Curriculum": "outputs/loss_logs/no_curriculum/alexnet_no_curriculum_losses.json",
            "Visual Acuity Transform": "outputs/loss_logs/visual_acuity/alexnet_visual_acuity_losses.json",
            "Color Perception Transform": "outputs/loss_logs/color_perception/alexnet_color_perception_losses.json"
        },
        "vgg16": {
            "Curriculum Learning": "outputs/loss_logs/curriculum/vgg16_curriculum_losses.json",
            #"Reverse Curriculum Learning": "outputs/loss_logs/reverse_curriculum/vgg16_reverse_curriculum_losses.json",
            "No Curriculum": "outputs/loss_logs/no_curriculum/vgg16_no_curriculum_losses.json",
            "Visual Acuity Transform": "outputs/loss_logs/visual_acuity/vgg16_visual_acuity_losses.json",
            "Color Perception Transform": "outputs/loss_logs/color_perception/vgg16_color_perception_losses.json"
        }
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

    # Compare models and generate plots for each model
    results = {}
    for model_name in models_info.keys():
        print(f"\nComparing tasks for model: {model_name}")

        # Compare performance for the current model
        results[model_name] = compare_models_performance(
            models_info[model_name],
            val_dataloader,
            criterion,
            device
        )

        # Generate plots for the current model
        plot_learning_curves(model_name, loss_logs[model_name], f"outputs/plots/{model_name}_learning_curves.png")
        save_accuracy_vs_loss(results, model_name, f"outputs/plots/{model_name}_accuracy_vs_loss.png")
        save_accuracy_histogram(results, model_name, f"outputs/plots/{model_name}_accuracy_histogram.png")

    # Print results
    print("\nModel Results:")
    for model_name, tasks in results.items():
        print(f"\nResults for {model_name}:")
        for task_name, metrics in tasks.items():
            print(f"{task_name}: Accuracy = {metrics['accuracy']:.2%}, Final Loss = {metrics['final_loss']:.4f}")


if __name__ == "__main__":
    main()
