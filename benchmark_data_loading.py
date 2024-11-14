import time
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset  # Import your existing dataset class
from util.plot_utils import plot_images

# Define a dummy transformation dataset without age-based modifications
class NoTransformImageDataset(ImageDataset):
    def apply_age_based_transformations(self, image):
        # This method is overridden to do nothing, so we get the raw images
        return image


def test_performance_and_plot(dataset_class, transform, img_dir, batch_size, age_in_months, max_images=None):
    """Test the time it takes to load images with a given dataset class and plot sample images."""
    # Initialize the dataset and dataloader
    dataset = dataset_class(img_dir=img_dir, transform=transform, age_in_months=age_in_months)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    start_time = time.time()
    images_loaded = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images_loaded += len(images)

        # Plot images for this batch
        plot_images(images, batch_idx, age_in_months, dataset_class.__name__)

        # Stop if max_images limit is reached
        if max_images and images_loaded >= max_images:
            break

    elapsed_time = time.time() - start_time
    return elapsed_time


if __name__ == '__main__':
    # Configuration
    img_dir = "dataset"  # Adjust to your dataset directory
    batch_size = 16
    age_in_months = 6  # Example age for transformations
    max_images = 100  # Maximum images to load, optional

    # Base transformation to resize and convert to tensor
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Test with transformations
    print("Testing with transformations:")
    time_with_transforms = test_performance_and_plot(
        ImageDataset, base_transform, img_dir, batch_size, age_in_months, max_images=max_images
    )
    print(f"Time to load up to {max_images} images with transformations: {time_with_transforms:.2f} seconds")

    # Test without transformations
    print("Testing without transformations:")
    time_without_transforms = test_performance_and_plot(
        NoTransformImageDataset, base_transform, img_dir, batch_size, age_in_months, max_images=max_images
    )
    print(f"Time to load up to {max_images} images without transformations: {time_without_transforms:.2f} seconds")
