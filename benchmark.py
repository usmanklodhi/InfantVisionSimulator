import time
import logging
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset  # Import your custom ImageDataset class
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from torchvision.transforms import ToPILImage
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_time_benchmark(img_dir, transform, age, apply_transform):
    """
    Benchmark the performance of data loading without batching.

    Args:
        img_dir (str): Path to the dataset directory.
        transform (torchvision.transforms): Transformations to apply.
        age (int): Age in months for transformations.
        apply_transform (bool): Whether to apply transformations.

    Returns:
        float: Time taken to load all images in seconds.
    """
    dataset = ImageDataset(img_dir=img_dir, transform=transform, age_in_months=age, apply_transform=apply_transform)

    start_time = time.time()
    for idx in range(len(dataset)):
        image, label = dataset[idx]  # Load image one by one
        if image is None:  # Handle failed loads
            continue
    total_time = time.time() - start_time

    logging.info(f"Total time ({'with' if apply_transform else 'without'} transform): {total_time:.2f} seconds")
    return total_time



def create_dataset(img_dir, transform, age, apply_transform):
    """
    Create a dataset instance with or without transformations.

    Args:
        img_dir (str): Path to the dataset directory.
        transform (torchvision.transforms): Transformations to apply.
        age (int): Age in months for transformations.
        apply_transform (bool): Whether to apply transformations.

    Returns:
        ImageDataset: A dataset instance.
    """
    return ImageDataset(
        img_dir=img_dir,
        transform=transform,
        age_in_months=age if apply_transform else 0,
        verbose=False
    )


def create_dataloader(dataset, batch_size):
    """
    Create a DataLoader instance.

    Args:
        dataset (ImageDataset): Dataset instance.
        batch_size (int): Number of images per batch.

    Returns:
        DataLoader: A DataLoader instance.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def verify_transformations(img_dir, transform, age):
    """
    Verify transformations visually and quantitatively, and plot results
    with comparison between with and without transformations.

    Args:
        img_dir (str): Path to the dataset directory.
        transform (torchvision.transforms): Transformations to apply.
        age (int): Age in months for transformations.
    """
    logging.info(f"Verifying transformations for age {age} months...")

    # Metrics for with and without transformations
    metrics_with_transform = analyze_metrics(img_dir, transform, age, apply_transform=True)
    metrics_without_transform = analyze_metrics(img_dir, transform, age, apply_transform=False)

    # Plot comparison
    plot_comparison(metrics_with_transform, metrics_without_transform)


def calculate_metrics(original_image, transformed_image, image_idx):
    """
    Calculate and log quality metrics for transformed images.

    Args:
        original_image (PIL.Image): Original image.
        transformed_image (PIL.Image): Transformed image.
        image_idx (int): Index of the image.

    Returns:
        tuple: MSE and SSIM values.
    """
    original_np = np.array(original_image)
    transformed_np = np.array(transformed_image)

    # Ensure SSIM window size fits the image dimensions
    min_dimension = min(original_np.shape[:2])  # Smallest dimension of the image
    win_size = min(7, min_dimension)  # Use a window size of 7 or less

    # Calculate MSE
    mse = mean_squared_error(original_np, transformed_np)

    # Calculate SSIM
    try:
        sim_index = ssim(original_np, transformed_np, multichannel=True, win_size=win_size, channel_axis=-1)
    except ValueError as e:
        logging.error(f"Error calculating SSIM for image {image_idx + 1}: {e}")
        sim_index = None

    # Log metrics
    if sim_index is not None:
        logging.info(f"Image {image_idx + 1} - MSE: {mse:.2f}, SSIM: {sim_index:.2f}")
    else:
        logging.info(f"Image {image_idx + 1} - MSE: {mse:.2f}, SSIM: Not computed (Error)")

    return mse, sim_index

def plot_comparison(metrics_with, metrics_without):
    """
    Plot comparison of MSE and SSIM for with and without transformations.

    Args:
        metrics_with (list of tuple): Metrics with transformations.
        metrics_without (list of tuple): Metrics without transformations.
    """
    if not metrics_with or not metrics_without:
        logging.error("Metrics are empty or None. Cannot plot comparison.")
        return

    mse_with = [m[0] for m in metrics_with]
    ssim_with = [m[1] if m[1] is not None else 0 for m in metrics_with]
    mse_without = [m[0] for m in metrics_without]
    ssim_without = [m[1] if m[1] is not None else 0 for m in metrics_without]

    x = range(1, len(metrics_with) + 1)  # Image indices

    plt.figure(figsize=(12, 6))

    # Plot MSE comparison
    plt.subplot(1, 2, 1)
    plt.bar(x, mse_with, width=0.4, label="With Transform", color="blue", alpha=0.7)
    plt.bar([i + 0.4 for i in x], mse_without, width=0.4, label="Without Transform", color="green", alpha=0.7)
    plt.xlabel("Image Index")
    plt.ylabel("MSE")
    plt.title("MSE Comparison")
    plt.legend()

    # Plot SSIM comparison
    plt.subplot(1, 2, 2)
    plt.bar(x, ssim_with, width=0.4, label="With Transform", color="orange", alpha=0.7)
    plt.bar([i + 0.4 for i in x], ssim_without, width=0.4, label="Without Transform", color="purple", alpha=0.7)
    plt.xlabel("Image Index")
    plt.ylabel("SSIM")
    plt.title("SSIM Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

def analyze_metrics(img_dir, transform, age, apply_transform):
    """
    Analyze MSE and SSIM for images with or without transformations.

    Args:
        img_dir (str): Path to the dataset directory.
        transform (torchvision.transforms): Transformations to apply.
        age (int): Age in months for transformations.
        apply_transform (bool): Whether to apply transformations.

    Returns:
        list of tuple: List of (MSE, SSIM) for each image or an empty list if errors occur.
    """
    dataset = create_dataset(img_dir, transform, age, apply_transform=apply_transform)
    metrics = []
    resize_transform = transforms.Compose([transforms.Resize((128, 128))])

    for idx in range(len(dataset)):
        try:
            original_image_path = f"{img_dir}/{dataset.img_labels[idx][0]}"
            original_image = Image.open(original_image_path).convert("RGB")
            original_image_resized = resize_transform(original_image)

            transformed_image, _ = dataset[idx]
            transformed_image = transforms.ToPILImage()(transformed_image)

            # Calculate metrics
            mse, sim_index = calculate_metrics(original_image_resized, transformed_image, idx)
            metrics.append((mse, sim_index))

        except Exception as e:
            logging.error(f"Error processing image {idx + 1}: {e}")
            continue  # Skip invalid images

    if not metrics:
        logging.warning("No valid metrics were computed. Please check the dataset.")
    return metrics

def visualize_transformed_vs_original(img_dir, transform, age, num_images=5):
    """
    Visualize random transformed images alongside non-transformed images for comparison.

    Args:
        img_dir (str): Path to the dataset directory.
        transform (torchvision.transforms.Compose): Transformations to apply.
        age (int): Age in months for transformations.
        num_images (int): Number of random images to visualize.
    """
    # Create datasets for transformed and non-transformed images
    transformed_dataset = create_dataset(img_dir, transform, age, apply_transform=True)
    non_transformed_dataset = create_dataset(img_dir, transform, age, apply_transform=False)

    # Select random indices for visualization
    random_indices = random.sample(range(len(transformed_dataset)), num_images)

    # Prepare for visualization
    plt.figure(figsize=(15, 5 * num_images))

    for i, idx in enumerate(random_indices):
        # Load transformed and non-transformed images
        transformed_image, _ = transformed_dataset[idx]
        non_transformed_image, _ = non_transformed_dataset[idx]

        # Convert tensors to PIL images for display
        transformed_image = ToPILImage()(transformed_image)
        non_transformed_image = ToPILImage()(non_transformed_image)

        # Plot non-transformed image
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(non_transformed_image)
        plt.axis("off")
        plt.title(f"Original (Idx {idx})")

        # Plot transformed image
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(transformed_image)
        plt.axis("off")
        plt.title(f"Transformed (Age {age} months)")

    plt.tight_layout()
    plt.show()
    

def time_benchmark(img_dir, base_transform, age):
    logging.info("Running benchmark with transformations...")
    time_with_transform = run_time_benchmark(img_dir, base_transform, age, apply_transform=True)

    logging.info("Running benchmark without transformations...")
    time_without_transform = run_time_benchmark(img_dir, base_transform, age, apply_transform=False)
    
def visual_comparsion(img_dir, base_transform, age, num_images):
    logging.info("Running benchmark with transformations...")
    visualize_transformed_vs_original(img_dir, base_transform, age, num_images)





if __name__ == '__main__':
    # Dataset directory
    img_dir = "dataset/Test_image_100"

    # Define base transformation
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Age for testing transformations
    age = 2  # Example age for transformations

    # Benchmark for transformations
    time_benchmark(img_dir, base_transform, age)
    
    visual_comparsion(img_dir, base_transform, age, 2)

    
    # Verify transformations
    verify_transformations(img_dir, base_transform, age)
