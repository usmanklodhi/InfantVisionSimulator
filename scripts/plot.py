import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataset import InfantVisionDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform

def plot_transformed_images(data_dir, ages, save_dir):
    """
    Plot 6 images from the dataset after applying transformations for a specific age.

    Args:
        data_dir (str): Path to the image dataset.
        age_in_months (int): Age in months for which the transformations are applied.
    """
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    for age in ages:
        # Define the transform for the current age
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            VisualAcuityTransform(age),
            ColorPerceptionTransform(age),
            transforms.ToTensor()
        ])

        # Load the dataset
        dataset = InfantVisionDataset(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=6, shuffle=False)  # Load 6 images per batch

        # Get the first batch of transformed images
        images, _ = next(iter(dataloader))

        # Plot the 6 images
        fig, axes = plt.subplots(1, 6, figsize=(18, 6))
        fig.suptitle(f"Transformed Images at Age {age} Months", fontsize=16)

        for i, image in enumerate(images):
            image_np = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
            axes[i].imshow(image_np)
            axes[i].axis("off")
            axes[i].set_title(f"Image {i + 1}")

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(save_dir, f"transformed_images_age_{age}_months.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)  # Close the plot to avoid displaying it
        print(f"Saved transformed images for age {age} months to {save_path}")


if __name__ == "__main__":
    data_dir = "dataset/Test_image_6"  # Path to your dataset
    ages = [0, 2, 3, 6, 8, 10, 12]    # Change the age for plotting
    output_dir = "output_images/Transformed_images_plot"

    plot_transformed_images(data_dir, ages, output_dir)
