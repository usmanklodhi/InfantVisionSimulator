import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.models import alexnet
from torch.utils.data import DataLoader
from src.dataset import InfantVisionDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform
import matplotlib.pyplot as plt
import time
import os


def evaluate_feature_extraction_quality(model, dataloader, device, model_type):
    """
    Evaluate feature extraction quality by calculating the mean and variance
    of features from an intermediate layer.

    Args:
        model (torch.nn.Module): Pre-trained model for feature extraction.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the evaluation on.

    Returns:
        dict: Mean and variance of extracted features.
    """
    feature_means = []
    feature_variances = []

    # Hook to extract features from an intermediate layer
    def hook(module, input, output):
        # Calculate mean and variance of the output
        mean = output.mean().item()
        variance = output.var().item()
        feature_means.append(mean)
        feature_variances.append(variance)

    # Register the hook
    # Dynamically register the hook based on model type
    if model_type == "ResNet":
        handle = model.layer4.register_forward_hook(hook)
    elif model_type == "AlexNet":
        handle = model.features.register_forward_hook(hook)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            model(images)  # Forward pass to extract features

    # Remove the hook
    handle.remove()

    # Calculate overall mean and variance
    mean_of_means = sum(feature_means) / len(feature_means)
    mean_of_variances = sum(feature_variances) / len(feature_variances)

    return {"mean": mean_of_means, "variance": mean_of_variances}


def evaluate_model_performance(data_dir, model, model_type, batch_size=16, num_classes=10, ages=[0, 2, 6, 12]):
    """
    Evaluate the performance of a pre-trained model on transformed datasets,
    including feature extraction quality.

    Args:
        data_dir (str): Path to the dataset directory.
        model (torch.nn.Module): Pre-trained model for evaluation.
        batch_size (int): Number of samples per batch.
        num_classes (int): Number of classes (adjust for your dataset).
        ages (list): List of ages (in months) for transformation testing.

    Returns:
        dict: Performance metrics (accuracy, time, feature quality) for each age group.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Base dataset transform (original images)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    performance_metrics = {
        "Age (Months)": [],
        "Accuracy (Original)": [],
        "Accuracy (Transformed)": [],
        "Time (Original)": [],
        "Time (Transformed)": [],
        "Feature Mean (Original)": [],
        "Feature Variance (Original)": [],
        "Feature Mean (Transformed)": [],
        "Feature Variance (Transformed)": [],
    }

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

        # Helper function to evaluate accuracy and time
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

        # Evaluate feature extraction quality for original images
        feature_quality_original = evaluate_feature_extraction_quality(model, dataloader_original, device, model_type)
        print(f"Original Images - Feature Mean: {feature_quality_original['mean']:.4f}, "
              f"Variance: {feature_quality_original['variance']:.4f}")

        # Evaluate on transformed images
        accuracy_transformed, time_transformed = evaluate_loader(dataloader_transformed)
        print(f"Transformed Images - Accuracy: {accuracy_transformed:.2f}%, Time: {time_transformed:.2f}s")

        # Evaluate feature extraction quality for transformed images
        feature_quality_transformed = evaluate_feature_extraction_quality(model, dataloader_transformed, device, model_type)
        print(f"Transformed Images - Feature Mean: {feature_quality_transformed['mean']:.4f}, "
              f"Variance: {feature_quality_transformed['variance']:.4f}")

        # Store results
        performance_metrics["Age (Months)"].append(age)
        performance_metrics["Accuracy (Original)"].append(accuracy_original)
        performance_metrics["Accuracy (Transformed)"].append(accuracy_transformed)
        performance_metrics["Time (Original)"].append(time_original)
        performance_metrics["Time (Transformed)"].append(time_transformed)
        performance_metrics["Feature Mean (Original)"].append(feature_quality_original["mean"])
        performance_metrics["Feature Variance (Original)"].append(feature_quality_original["variance"])
        performance_metrics["Feature Mean (Transformed)"].append(feature_quality_transformed["mean"])
        performance_metrics["Feature Variance (Transformed)"].append(feature_quality_transformed["variance"])

    return performance_metrics

import matplotlib.pyplot as plt


def plot_and_save_performance_metrics_with_model(performance_metrics, output_dir, model_name):
    """
    Plot and save performance metrics for comparison with model name in filenames.
    
    Args:
        performance_metrics (dict): Metrics dictionary with accuracies, times, and feature quality.
        output_dir (str): Directory to save the plots.
        model_name (str): Name of the model being evaluated.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    ages = performance_metrics["Age (Months)"]

    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ages, performance_metrics["Accuracy (Original)"], label="Accuracy (Original)", marker='o')
    plt.plot(ages, performance_metrics["Accuracy (Transformed)"], label="Accuracy (Transformed)", marker='o')
    plt.xlabel("Age (Months)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(f"Accuracy vs Age (Original vs Transformed) - {model_name}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{model_name}_accuracy_vs_age.png")
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.show()

    # Time Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ages, performance_metrics["Time (Original)"], label="Time (Original)", marker='o')
    plt.plot(ages, performance_metrics["Time (Transformed)"], label="Time (Transformed)", marker='o')
    plt.xlabel("Age (Months)", fontsize=12)
    plt.ylabel("Time (Seconds)", fontsize=12)
    plt.title(f"Time Taken vs Age (Original vs Transformed) - {model_name}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{model_name}_time_vs_age.png")
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.show()

    # Feature Mean Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ages, performance_metrics["Feature Mean (Original)"], label="Feature Mean (Original)", marker='o')
    plt.plot(ages, performance_metrics["Feature Mean (Transformed)"], label="Feature Mean (Transformed)", marker='o')
    plt.xlabel("Age (Months)", fontsize=12)
    plt.ylabel("Feature Mean", fontsize=12)
    plt.title(f"Feature Mean vs Age (Original vs Transformed) - {model_name}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{model_name}_feature_mean_vs_age.png")
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.show()

    # Feature Variance Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ages, performance_metrics["Feature Variance (Original)"], label="Feature Variance (Original)", marker='o')
    plt.plot(ages, performance_metrics["Feature Variance (Transformed)"], label="Feature Variance (Transformed)", marker='o')
    plt.xlabel("Age (Months)", fontsize=12)
    plt.ylabel("Feature Variance", fontsize=12)
    plt.title(f"Feature Variance vs Age (Original vs Transformed) - {model_name}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{model_name}_feature_variance_vs_age.png")
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.show()


def evaluate_and_plot_for_alexnet(data_dir, output_dir, batch_size, ages, model_type):
    """
    Evaluate performance metrics and plot results for AlexNet.
    
    Args:
        data_dir (str): Path to the dataset directory.
        output_dir (str): Directory to save the plots.
    """
    # Define the AlexNet model
    model_name = model_type
    model = alexnet(pretrained=True)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 10)  # Adjust for 10 classes

    # Evaluate performance
    metrics = evaluate_model_performance(data_dir, model, model_type, batch_size=batch_size, ages=ages)

    # Plot and save performance metrics
    plot_and_save_performance_metrics_with_model(metrics, output_dir, model_name)

    print(f"Plots saved in: {output_dir}")
    
def evaluate_and_plot_for_resnet18(data_dir, output_dir, batch_size, ages, model_type):
    """
    Evaluate performance metrics and plot results for AlexNet.
    
    Args:
        data_dir (str): Path to the dataset directory.
        output_dir (str): Directory to save the plots.
    """
    # Define the AlexNet model
    model_name = model_type
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes  

    # Evaluate performance
    metrics = evaluate_model_performance(data_dir, model, model_type, batch_size=batch_size, ages=ages)
    
    # Print metrics
    for key, values in metrics.items():
        print(f"{key}: {values}")

    # Plot and save performance metrics
    plot_and_save_performance_metrics_with_model(metrics, output_dir, model_name)

    print(f"Plots saved in: {output_dir}")

if __name__ == "__main__":
    # Path to dataset
    data_dir = "dataset/Test_image_100"
    
    # Output directory for plots
    output_dir = "output_images/performance_metrics_plots"
    
    # Batch size
    batch_size=16
    
    # The age in months 
    ages=[0, 2, 4, 6, 8, 12]

    # Pre-trained model 
    evaluate_and_plot_for_alexnet(data_dir, output_dir, batch_size, ages, model_type="AlexNet")
    
    evaluate_and_plot_for_resnet18(data_dir, output_dir, batch_size, ages, model_type="ResNet")