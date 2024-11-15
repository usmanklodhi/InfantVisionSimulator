import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset  # Import existing dataset class
from util.plot_utils import plot_images

# Define a dummy transformation dataset without age-based modifications
class NoTransformImageDataset(ImageDataset):
    def apply_age_based_transformations(self, image):
        # This method is overridden to do nothing, so we get the raw images
        return image


def test_performance(dataset_class, transform, img_dir, batch_size, age_in_months, max_images=None):
    """Test time to load images with given dataset class, return elapsed time."""
    dataset = dataset_class(img_dir=img_dir, transform=transform, age_in_months=age_in_months)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    start_time = time.time()
    images_loaded = 0
    
    for _, (images, _) in enumerate(dataloader):
        images_loaded += len(images)
        if max_images and images_loaded >= max_images:
            break

    return time.time() - start_time

def benchmark_performance(img_dir, batch_size, age_in_months, max_images):
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Test with transformations
    print("Testing with transformations:")
    time_with_transforms = test_performance(
        ImageDataset, base_transform, img_dir, batch_size, age_in_months, max_images
    )
    print(f"Time to load up to {max_images} images with transformations: {time_with_transforms:.2f} seconds")
    
    # Test without transformations
    print("Testing without transformations:")
    time_without_transforms = test_performance(
        NoTransformImageDataset, base_transform, img_dir, batch_size, age_in_months, max_images
    )
    print(f"Time to load up to {max_images} images without transformations: {time_without_transforms:.2f} seconds")

if __name__ == '__main__':
    # Configuration
    img_dir = "dataset"  # Adjust to your dataset directory
    batch_size = 16
    age_in_months = 6  # Example age for transformations
    max_images = 100  # Maximum images to load, optional
    benchmark_performance(img_dir, batch_size, age_in_months, max_images)
    
