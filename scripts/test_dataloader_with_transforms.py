import os
from PIL import Image
from src.dataset import InfantVisionDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform
from util.utils import save_image_as_plot
from config import DATA_DIR, AGES, NUM_IMAGES, OUTPUT_DIR

# This approach might be marked as redundant. We may not need individual transforms parameterized by age.
def process_single_image(original_image, age):
    # Apply visual acuity transformation
    visual_acuity = VisualAcuityTransform(age)
    acuity_image = visual_acuity(original_image)

    # Apply color perception transformation
    color_perception = ColorPerceptionTransform(age)
    color_image = color_perception(acuity_image)

    # Transformation details
    details_1 = f"Blur Radius: {visual_acuity.blur_radius:.2f}"
    details_2 = (f"RG Sensitivity: {color_perception.red_green_sensitivity:.2f}, "
                 f"BY Sensitivity: {color_perception.blue_yellow_sensitivity:.2f}, "
                 f"Saturation: {color_perception.saturation_factor:.2f}")

    return [original_image, acuity_image, color_image], details_1, details_2


def test_dataloader(data_dir, ages, output_dir, num_images=5):
    dataset = InfantVisionDataset(data_dir)

    for idx, (original_image, _) in enumerate(dataset):
        if idx >= num_images:  # Limit to the specified number of images
            break

        for age in ages:
            print(f"Processing image {idx + 1} for age {age} months...")

            # Process the image with transformations
            transformed_images, details_1, details_2 = process_single_image(original_image, age)

            # Define save path
            age_dir = os.path.join(output_dir, f"age_{age}_months")
            save_path = os.path.join(age_dir, f"image_{idx + 1:04d}.png")

            # Save the comparison image using the utility function
            titles = ["Original", f"Visual Acuity\n{details_1}", f"Color Perception\n{details_2}"]
            save_image_as_plot(transformed_images, titles, f"Age: {age} Months", save_path)


if __name__ == "__main__":
    test_data_dir = DATA_DIR
    test_ages = AGES
    output_directory = OUTPUT_DIR
    num_images_to_process = 5  # Limit the number of images to process

    test_dataloader(test_data_dir, test_ages, output_directory, num_images=num_images_to_process)
