import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.dataset import InfantVisionDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform

def create_comparison_image(original, transformed_1, transformed_2, age, details_1, details_2, save_path):
    """
    Create and save a comparison image using Matplotlib with titles and margins.

    Args:
        original (PIL.Image): Original image.
        transformed_1 (PIL.Image): Image after first transformation.
        transformed_2 (PIL.Image): Image after second transformation.
        age (int): Age in months.
        details_1 (str): Details of the first transformation.
        details_2 (str): Details of the second transformation.
        save_path (str): Path to save the combined image.
    """
    # Convert images to Matplotlib-friendly format (NumPy arrays)
    original_np = transforms.ToTensor()(original).permute(1, 2, 0).numpy()
    transformed_1_np = transforms.ToTensor()(transformed_1).permute(1, 2, 0).numpy()
    transformed_2_np = transforms.ToTensor()(transformed_2).permute(1, 2, 0).numpy()

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f"Age: {age} Months", fontsize=18, fontweight='bold', y=0.95)

    # Plot the images
    axes[0].imshow(original_np)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(transformed_1_np)
    axes[1].set_title(f"Visual Acuity\n{details_1}", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(transformed_2_np)
    axes[2].set_title(f"Color Perception\n{details_2}", fontsize=12)
    axes[2].axis("off")

    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison image to {save_path}")

def test_dataloader(data_dir, ages, output_dir):
    """
    Test the data loader and create comparison images using Matplotlib.

    Args:
        data_dir (str): Path to the dataset directory.
        ages (list): List of ages (in months) to test.
        output_dir (str): Directory to save the output images.
    """
    dataset = InfantVisionDataset(data_dir)

    for idx, (original_image, _) in enumerate(dataset):
        for age in ages:
            print(f"Processing image {idx + 1} for age {age} months...")

            # Apply transformations
            visual_acuity = VisualAcuityTransform(age)
            color_perception = ColorPerceptionTransform(age)

            transformed_1 = visual_acuity(original_image)
            transformed_2 = color_perception(transformed_1)

            # Details for the transformations
            details_1 = f"Blur Radius: {visual_acuity.blur_radius:.2f}"
            details_2 = (f"RG Sensitivity: {color_perception.red_green_sensitivity:.2f}, "
                         f"BY Sensitivity: {color_perception.blue_yellow_sensitivity:.2f}, "
                         f"Saturation: {color_perception.saturation_factor:.2f}")

            # display the details
            print(details_1)
            print(details_2)

            # Save the comparison image
            age_dir = os.path.join(output_dir, f"age_{age}_months")
            save_path = os.path.join(age_dir, f"image_{idx + 1:04d}.png")
            create_comparison_image(
                original_image, transformed_1, transformed_2, age, details_1, details_2, save_path
            )

if __name__ == "__main__":
    # Test parameters
    test_data_dir = "dataset/Test_image_6"  # Path to your dataset
    test_ages = [0, 1, 2, 3, 6, 8, 12]  # Ages to test (in months)
    output_directory = "output_images/test_dataloader_results"  # Directory to save results

    # Run the test
    test_dataloader(test_data_dir, test_ages, output_directory)
