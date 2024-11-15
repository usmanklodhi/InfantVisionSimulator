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
    for age in [0, 2, 3, 6, 12]:  # Testing for ages 0 months to 16 months
        print(f"Testing for age {age} months")

        # Initialize the dataset with different ages
        dataset = ImageDataset(img_dir="dataset", transform=base_transform, age_in_months=age)

        # List of transformations based on age
        applied_transforms = []
        if 0 <= age < 2:
            applied_transforms.append("High Blur (Max Acuity)")
            applied_transforms.append("Grayscale")
        elif 2 <= age < 3:
            applied_transforms.append("Blur (High)")
            applied_transforms.append("Grayscale + Red-Green Sensitivity")
        elif 3 <= age < 6:
            applied_transforms.append("Reduced Blur")
            applied_transforms.append("Enhanced Red-Green Sensitivity")
        elif 6 <= age < 12:
            applied_transforms.append("Minimal Blur")
            applied_transforms.append("Red-Green + Blue-Yellow Sensitivity")
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