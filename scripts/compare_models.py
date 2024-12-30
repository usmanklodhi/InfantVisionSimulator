import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from src.models import get_model
from src.preprocessed_dataset import PreprocessedDataset
from setting import NUM_CLASSES, DEVICE, BATCH_SIZE

# Function to calculate accuracy
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Function to load loss data
def load_loss_data(loss_file):
    with open(loss_file, "r") as f:
        data = json.load(f)
    return data["train_losses"], data["val_losses"]

# Function to plot comparative loss and accuracy
def plot_comparisons(train_losses, val_losses, accuracies, model_name, stage_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Plot losses
    plt.figure()
    for stage_name, (train, val) in zip(stage_names, train_losses.items()):
        plt.plot(train, label=f"Train Loss ({stage_name})", linestyle="--")
        plt.plot(val, label=f"Validation Loss ({stage_name})", linestyle="-")
    plt.title(f"Loss Comparison - {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss_comparison.png"))
    plt.close()

    # Plot accuracy as a histogram
    plt.figure()
    plt.bar(stage_names, accuracies, color=["blue", "green", "orange", "purple"])
    plt.title(f"Accuracy Comparison - {model_name}")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join(output_dir, f"{model_name}_accuracy_comparison.png"))
    plt.close()

# Main function
def evaluate_comparisons(model_dirs, loss_dirs, dataset, batch_size, model_names, output_dir):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    preprocessed_dataset = PreprocessedDataset(dataset, transform=transform)
    dataloader = torch.utils.data.DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for model_name in model_names:
        stage_names = ["No Curriculum", "Curriculum", "Color Perception", "Visual Acuity"]
        train_losses = {}
        val_losses = {}
        accuracies = []

        for stage, dir_key in zip(stage_names, model_dirs.keys()):
            model_path = os.path.join(model_dirs[dir_key], f"{model_name}_{dir_key}_final.pth")
            loss_file = os.path.join(loss_dirs[dir_key], f"{model_name}_{dir_key}_losses.json")

            # Load model
            model = get_model(model_name, num_classes=NUM_CLASSES)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model = model.to(DEVICE)

            # Load loss data
            train, val = load_loss_data(loss_file)
            train_losses[stage] = train
            val_losses[stage] = val

            # Calculate accuracy
            acc = calculate_accuracy(model, dataloader)
            accuracies.append(acc)
            print(f"Accuracy for {stage} ({model_name}): {acc:.2f}%")

        # Plot comparisons
        plot_comparisons(train_losses, val_losses, accuracies, model_name, stage_names, output_dir)

if __name__ == "__main__":
    # Configuration
    model_dirs = {
        "no_curriculum": "outputs/models/no_curriculum/",
        "curriculum": "outputs/models/curriculum/",
        "color_perception": "outputs/models/color_perception/",
        "visual_acuity": "outputs/models/visual_acuity/"
    }
    loss_dirs = {
        "no_curriculum": "outputs/loss_logs/no_curriculum/",
        "curriculum": "outputs/loss_logs/curriculum/",
        "color_perception": "outputs/loss_logs/color_perception/",
        "visual_acuity": "outputs/loss_logs/visual_acuity/"
    }
    output_dir = "output/plots/"
    model_names = ["resnet18", "vgg16", "alexnet"]  # Add more models if needed

    # Dataset (replace with actual dataset)
    from datasets import load_dataset
    dataset = load_dataset("zh-plus/tiny-imagenet")["valid"]

    batch_size = BATCH_SIZE

    evaluate_comparisons(model_dirs, loss_dirs, dataset, batch_size, model_names, output_dir)
