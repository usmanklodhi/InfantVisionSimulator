import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from src.models import get_model
from src.preprocessed_dataset import PreprocessedDataset
from setting import NUM_CLASSES, DEVICE

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
def plot_comparisons(train_losses_nc, val_losses_nc, acc_nc, train_losses_c, val_losses_c, acc_c, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Plot losses
    plt.figure()
    plt.plot(train_losses_nc, label="Train Loss (No Curriculum)", linestyle="--")
    plt.plot(val_losses_nc, label="Validation Loss (No Curriculum)", linestyle="--")
    plt.plot(train_losses_c, label="Train Loss (Curriculum)", linestyle="-")
    plt.plot(val_losses_c, label="Validation Loss (Curriculum)", linestyle="-")
    plt.title(f"Loss Comparison - {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss_comparison.png"))
    plt.close()

    # Plot accuracy as a histogram
    plt.figure()
    plt.bar(["No Curriculum", "Curriculum"], [acc_nc, acc_c], color=["blue", "green"])
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
        # Paths for no curriculum
        model_path_nc = os.path.join(model_dirs["no_curriculum"], f"{model_name}_no_curriculum_final.pth")
        loss_file_nc = os.path.join(loss_dirs["no_curriculum"], f"{model_name}_no_curriculum_losses.json")

        # Paths for curriculum
        model_path_c = os.path.join(model_dirs["curriculum"], f"{model_name}_curriculum_final.pth")
        loss_file_c = os.path.join(loss_dirs["curriculum"], f"{model_name}_curriculum_losses.json")

        # Load models
        model_nc = get_model(model_name, num_classes=NUM_CLASSES)
        model_nc.load_state_dict(torch.load(model_path_nc, map_location=DEVICE, weights_only=True))
        model_nc = model_nc.to(DEVICE)

        model_c = get_model(model_name, num_classes=NUM_CLASSES)
        model_c.load_state_dict(torch.load(model_path_c, map_location=DEVICE, weights_only=True))
        model_c = model_c.to(DEVICE)

        # Load loss data
        train_losses_nc, val_losses_nc = load_loss_data(loss_file_nc)
        train_losses_c, val_losses_c = load_loss_data(loss_file_c)

        # Calculate accuracy
        acc_nc = calculate_accuracy(model_nc, dataloader)
        acc_c = calculate_accuracy(model_c, dataloader)

        # Plot comparisons
        plot_comparisons(train_losses_nc, val_losses_nc, acc_nc, train_losses_c, val_losses_c, acc_c, model_name, output_dir)

if __name__ == "__main__":
    # Configuration
    model_dirs = {
        "no_curriculum": "outputs/models/no_curriculum/",
        "curriculum": "outputs/models/curriculum/"
    }
    loss_dirs = {
        "no_curriculum": "outputs/loss_logs/no_curriculum/",
        "curriculum": "outputs/loss_logs/curriculum/"
    }
    output_dir = "outputs/plots/"
    model_names = ["resnet18"]  # Add more models if needed

    # Dataset (replace with actual dataset)
    from datasets import load_dataset
    dataset = load_dataset("zh-plus/tiny-imagenet")["valid"]

    batch_size = 32

    evaluate_comparisons(model_dirs, loss_dirs, dataset, batch_size, model_names, output_dir)
