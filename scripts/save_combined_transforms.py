import os
from PIL import Image
from torchvision import transforms
from src.dataset import InfantVisionDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform
import matplotlib.pyplot as plt
from config import DATA_DIR, AGES, NUM_IMAGES, OUTPUT_DIR


def apply_transformations(image, age):
    visual_acuity_transform = VisualAcuityTransform(age)
    acuity_image = visual_acuity_transform(image)

    color_perception_transform = ColorPerceptionTransform(age)
    color_image = color_perception_transform(image)

    return [image, acuity_image, color_image] # Return individual transformations applied to the image in a list


def save_combined_image(images, titles, output_path, age):
    num_images = len(images) // 3
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    fig.suptitle(f"Transformations for Age {age} Months", fontsize=16)

    for idx in range(num_images):
        for col in range(3):
            img_idx = idx * 3 + col
            axes[idx, col].imshow(images[img_idx])
            axes[idx, col].set_title(titles[img_idx % 3], fontsize=12)
            axes[idx, col].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined image: {output_path}")


def save_combined_images_by_age(data_dir, output_dir, num_images=5, ages=[0, 3, 6, 12]):
    # Create a transform that resizes the image to 256x256 and converts it to a tensor
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.ToTensor()
    ])

    # Load the dataset
    dataset = InfantVisionDataset(data_dir, transform=transform)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for age in ages:
        print(f"Processing images for age {age} months...")
        combined_images = []
        titles = ["Original", "Visual Acuity", "Color Perception"] * num_images

        for idx in range(num_images):
            original_image, _ = dataset[idx]
            combined_images.extend(apply_transformations(original_image, age))

        save_path = os.path.join(output_dir, f"combined_age_{age}_months.png")
        save_combined_image(combined_images, titles, save_path, age)


if __name__ == "__main__":
    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR

    save_combined_images_by_age(data_dir, output_dir, num_images=5, ages=AGES)
