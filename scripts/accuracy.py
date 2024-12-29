import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.tinyimagenet_dataloader import TestTinyImageNetDataset, id_dict
from torchvision.transforms import Normalize
from src.models import get_model
from setting import DEVICE, NUM_CLASSES, BATCH_SIZE


def evaluate_model(model_path, model_name, batch_size=BATCH_SIZE):
    """
    Evaluate the accuracy of a single model on the Tiny ImageNet test set.
    """
    # Transform for preprocessing
    transform = Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))

    # Load the test dataset
    testset = TestTinyImageNetDataset(id_dict=id_dict, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load the model
    model = get_model(model_name, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Define loss function and metrics
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0

    # Evaluate the model
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Predictions and accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * inputs.size(0)

    # Calculate metrics
    accuracy = correct / total
    avg_loss = total_loss / total

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Loss: {avg_loss:.4f}")

    return accuracy, avg_loss


if __name__ == "__main__":
    # Example Usage

    # Path to the model to evaluate
    model_path = "outputs/models/no_curriculum/resnet18_no_curriculum_final.pth"  # Update this path as needed
    model_name = "resnet18"  # Model name: resnet18, alexnet, or vgg16

    # Evaluate the model
    evaluate_model(model_path, model_name)
