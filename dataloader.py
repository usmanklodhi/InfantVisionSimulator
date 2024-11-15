import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset  # Import your custom ImageDataset class
from util.plot_utils import save_images  # Assumed to be a utility function for saving images
import logging

# Configure logging to output to the console
logging.basicConfig(level=logging.INFO)

def main():
    # Define basic transformation for resizing and converting to tensor
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # List of ages to test for transformations
    test_ages = [0, 2, 3, 6, 12]

    for age in test_ages:
        logging.info(f"Testing transformations for age {age} months")

        # Initialize the dataset with specified age
        dataset = ImageDataset(img_dir="dataset", transform=base_transform, age_in_months=age)
        
        # Define the transformations applied for each age group
        applied_transforms = get_applied_transforms(age)
        
        # Initialize the DataLoader
        dataloader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=4)

        # Load one batch and display/save images
        for batch_idx, (images, _) in enumerate(dataloader):
            logging.info(f"Processing Batch {batch_idx} for age {age} months")
            logging.info(f"Image batch shape: {images.shape}")

            # Save images with transformations for inspection
            save_images(images, batch_idx, age_in_months=age, applied_transforms=', '.join(applied_transforms))
            break  # Process only the first batch for each age

def get_applied_transforms(age):
    """Determine the list of transformations based on the infant's age."""
    if 0 <= age < 2:
        return ["High Blur (Max Acuity)", "Grayscale"]
    elif 2 <= age < 3:
        return ["Blur (High)", "Grayscale + Red-Green Sensitivity"]
    elif 3 <= age < 6:
        return ["Reduced Blur", "Enhanced Red-Green Sensitivity"]
    elif 6 <= age < 12:
        return ["Minimal Blur", "Red-Green + Blue-Yellow Sensitivity"]
    else:
        return ["Minimal Blur", "Full Color"]

if __name__ == '__main__':
    main()
