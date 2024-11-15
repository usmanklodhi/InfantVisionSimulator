import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset  # Import your custom ImageDataset class
from util.plot_utils import save_images  # utility function for saving images
import logging


# Configure logging to output to the console
logging.basicConfig(level=logging.INFO)

def test_on_6_images():
    """
    Test transformations visually using Test_image_6.
    """
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    test_ages = [0, 2, 3, 6, 12]

    for age in test_ages:
        logging.info(f"Testing transformations for age {age} months on Test_image_6")

        dataset = ImageDataset(img_dir="dataset/Test_image_6", transform=base_transform, age_in_months=age, verbose=True)
        
        applied_transforms = get_applied_transforms(age)

        dataloader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=4)

        for batch_idx, (images, _) in enumerate(dataloader):
            logging.info(f"Processing Batch {batch_idx} for age {age} months")
            logging.info(f"Image batch shape: {images.shape}")
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
    # Uncomment the desired function to run
    test_on_6_images()  # For visual testing on Test_image_6
    # test_on_100_images()  # For performance testing on Test_image_100