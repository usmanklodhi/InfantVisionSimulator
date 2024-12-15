# import os
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from src.dataset import InfantVisionDataset
# from transforms.visual_acuity import VisualAcuityTransform
# from transforms.color_perception import ColorPerceptionTransform
# from config.__init__ import DATA_DIR, AGES, OUTPUT_DIR


# def create_age_based_transform(age):
#     return transforms.Compose([
#         transforms.Resize((256, 256)),
#         VisualAcuityTransform(age),
#         ColorPerceptionTransform(age),
#         transforms.ToTensor()
#     ])


# def save_transformed_images_grid(images, age, save_dir, grid_title="Transformed Images"):
#     # Create output directory if not exists
#     os.makedirs(save_dir, exist_ok=True)

#     # Create a Matplotlib figure
#     fig, axes = plt.subplots(1, len(images), figsize=(18, 6))
#     fig.suptitle(f"{grid_title} at Age {age} Months", fontsize=16)

#     for i, image in enumerate(images):
#         image_np = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
#         axes[i].imshow(image_np)
#         axes[i].axis("off")
#         axes[i].set_title(f"Image {i + 1}")

#     plt.tight_layout()

#     # Save the plot
#     save_path = os.path.join(save_dir, f"transformed_images_age_{age}_months.png")
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved transformed images for age {age} months to {save_path}")


# def plot_transformed_images(data_dir, ages, save_dir):
#     for age in ages:
#         print(f"Processing images for age {age} months...")

#         # Create transform for the current age
#         transform = create_age_based_transform(age)

#         # Load my_datasets and dataloader
#         dataset = InfantVisionDataset(data_dir, transform=transform)
#         dataloader = DataLoader(dataset, batch_size=5, shuffle=False)  # Load 6 images per batch

#         # Get the first batch of transformed images
#         images, _ = next(iter(dataloader))

#         # Save the transformed images grid
#         save_transformed_images_grid(images, age, save_dir)

# # Testing script: 5 images for each of the two properties, with at least 3 different parameter settings
# if __name__ == "__main__":
#     data_dir = DATA_DIR
#     ages = AGES
#     output_dir = OUTPUT_DIR

#     plot_transformed_images(data_dir, ages, output_dir)
