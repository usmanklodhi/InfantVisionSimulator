import os
from PIL import Image
from torchvision import transforms
from src.dataset import InfantVisionDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform
import matplotlib.pyplot as plt


def save_combined_images_by_age(data_dir, output_dir, num_images=5, ages=[0, 3, 6, 12]):
    """
    Save combined images for each age in a single image using Matplotlib.

    Args:
        data_dir (str): Path to the dataset directory.
        output_dir (str): Directory to save combined images.
        num_images (int): Number of images to process.
        ages (list): Ages (in months) for parameter settings.
    """
    # Load the dataset
    dataset = InfantVisionDataset(data_dir)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for age in ages:
        print(f"Processing images for age {age} months...")

        # Create a Matplotlib figure
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
        fig.suptitle(f"Transformations for Age {age} Months", fontsize=16)

        for idx in range(num_images):
            # Get the original image
            original_image, _ = dataset[idx]

            # Apply visual acuity transformation
            visual_acuity_transform = VisualAcuityTransform(age)
            acuity_image = visual_acuity_transform(original_image)

            # Apply color perception transformation
            color_perception_transform = ColorPerceptionTransform(age)
            color_image = color_perception_transform(original_image)

            # Display images
            images = [original_image, acuity_image, color_image]
            titles = ["Original", "Visual Acuity", "Color Perception"]

            for col, (img, title) in enumerate(zip(images, titles)):
                ax = axes[idx, col]
                ax.imshow(img)
                ax.set_title(title, fontsize=12)
                ax.axis("off")

        # Save the combined image
        save_path = os.path.join(output_dir, f"combined_age_{age}_months.png")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined image: {save_path}")


if __name__ == "__main__":
    # Path to the dataset directory
    data_dir = "dataset/Test_image_6"

    # Output directory to save combined images
    output_dir = "output_images/combined_transformed_examples"

    # Save combined images for each age
    save_combined_images_by_age(data_dir, output_dir, num_images=5, ages=[0, 3, 6, 12])
