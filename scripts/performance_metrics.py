import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from src.dataset import InfantVisionDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform
import time

def evaluate_model_performance(data_dir, model, batch_size=16, num_classes=10, ages=[0, 2, 6, 12]):
    """
    Evaluate the performance of a pre-trained model on transformed datasets.

    Args:
        data_dir (str): Path to the dataset directory.
        model (torch.nn.Module): Pre-trained model for evaluation.
        batch_size (int): Number of samples per batch.
        num_classes (int): Number of classes (adjust for your dataset).
        ages (list): List of ages (in months) for transformation testing.
    
    Returns:
        dict: Performance metrics (accuracy and time) for each age group.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Base dataset transform (original images)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    performance_metrics = {"Age (Months)": [], "Accuracy (Original)": [], "Accuracy (Transformed)": [],
                           "Time (Original)": [], "Time (Transformed)": []}

    # Evaluate for each age
    for age in ages:
        print(f"\nEvaluating performance for age: {age} months...")
        
        # Define the transform for this age
        transformed_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            VisualAcuityTransform(age),
            ColorPerceptionTransform(age),
            transforms.ToTensor(),
        ])

        # Original dataset
        dataset_original = InfantVisionDataset(data_dir, transform=base_transform)
        dataloader_original = DataLoader(dataset_original, batch_size=batch_size, shuffle=False)

        # Transformed dataset
        dataset_transformed = InfantVisionDataset(data_dir, transform=transformed_transform)
        dataloader_transformed = DataLoader(dataset_transformed, batch_size=batch_size, shuffle=False)

        # Helper function to evaluate
        def evaluate_loader(dataloader):
            correct = 0
            total = 0
            start_time = time.time()
            for images, _ in dataloader:
                images = images.to(device)
                labels = torch.randint(0, num_classes, (images.size(0),)).to(device)  # Simulated labels
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            end_time = time.time()
            accuracy = 100 * correct / total
            elapsed_time = end_time - start_time
            return accuracy, elapsed_time

        # Evaluate on original images
        accuracy_original, time_original = evaluate_loader(dataloader_original)
        print(f"Original Images - Accuracy: {accuracy_original:.2f}%, Time: {time_original:.2f}s")

        # Evaluate on transformed images
        accuracy_transformed, time_transformed = evaluate_loader(dataloader_transformed)
        print(f"Transformed Images - Accuracy: {accuracy_transformed:.2f}%, Time: {time_transformed:.2f}s")

        # Store results
        performance_metrics["Age (Months)"].append(age)
        performance_metrics["Accuracy (Original)"].append(accuracy_original)
        performance_metrics["Accuracy (Transformed)"].append(accuracy_transformed)
        performance_metrics["Time (Original)"].append(time_original)
        performance_metrics["Time (Transformed)"].append(time_transformed)

    return performance_metrics


def plot_performance_metrics(performance_metrics):
    """
    Plot performance metrics for comparison.
    
    Args:
        performance_metrics (dict): Metrics dictionary with accuracies and times.
    """
    import matplotlib.pyplot as plt

    ages = performance_metrics["Age (Months)"]

    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ages, performance_metrics["Accuracy (Original)"], label="Accuracy (Original)", marker='o')
    plt.plot(ages, performance_metrics["Accuracy (Transformed)"], label="Accuracy (Transformed)", marker='o')
    plt.xlabel("Age (Months)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Accuracy vs Age (Original vs Transformed)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Time Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ages, performance_metrics["Time (Original)"], label="Time (Original)", marker='o')
    plt.plot(ages, performance_metrics["Time (Transformed)"], label="Time (Transformed)", marker='o')
    plt.xlabel("Age (Months)", fontsize=12)
    plt.ylabel("Time (Seconds)", fontsize=12)
    plt.title("Time Taken vs Age (Original vs Transformed)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Path to dataset
    data_dir = "dataset/Test_image_100"

    # Pre-trained model (ResNet18 as an example)
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes

    # Evaluate performance
    metrics = evaluate_model_performance(data_dir, model, batch_size=16, ages=[0, 2, 6, 12])

    # Plot performance metrics
    plot_performance_metrics(metrics)
