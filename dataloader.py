import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import math
from dataset import ImageDataset  # Import your custom ImageDataset class
from util.plot_utils import save_images


if __name__ == '__main__':
    # Define transformations for the dataset
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Test at different ages
    for age in [0, 6, 12, 24, 36]:  # Testing for ages 0 months to 16 months
        print(f"Testing for age {age} months")

        # Initialize the dataset with different ages
        dataset = ImageDataset(img_dir="dataset", transform=base_transform, age_in_months=age)

        # List of transformations based on age
        applied_transforms = []
        if age == 0:
            applied_transforms.append("Max Blur (High)")
            applied_transforms.append("Grayscale")
        elif age < 4:
            applied_transforms.append("Blur (High)")
            applied_transforms.append("Grayscale")
        elif age < 12:
            applied_transforms.append("Reduced Blur")
            applied_transforms.append("Grayscale-Color Transition")
        else:
            applied_transforms.append("Minimal Blur")
            applied_transforms.append("Full Color")

        # Initialize the DataLoader
        dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=4)

        # Get one batch of images
        for batch_idx, (images, _) in enumerate(dataloader):
            print("Batch:", batch_idx)
            print("Images:", images.shape)

            # Plot the images in the batch with age and transformations applied
            save_images(images, batch_idx, age_in_months=age, applied_transforms=', '.join(applied_transforms))


            # Break after the first batch for each age
            break