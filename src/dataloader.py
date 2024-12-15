from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataset import InfantVisionDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform
import matplotlib.pyplot as plt
import numpy as np
from config.__init__ import DATA_DIR, AGES


# Wrapper function for Part 1 Task 4
def create_dataloader_v2(data_dir, batch_size=1, age_in_months=0, use_visual_transform=False, use_colour_transform=False, img_size=(256, 256)):
    if use_visual_transform:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            VisualAcuityTransform(age_in_months),
            transforms.ToTensor()
        ])
    elif use_colour_transform:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            ColorPerceptionTransform(age_in_months),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    dataset = InfantVisionDataset(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def create_dataloader(data_dir, batch_size=1, age_in_months=0, use_transform=True, img_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        VisualAcuityTransform(age_in_months),
        ColorPerceptionTransform(age_in_months),
        transforms.ToTensor()
    ]) if use_transform else transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    dataset = InfantVisionDataset(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Create a dataloader provided the dataset and batch size
def create_dataloader_v3(dataset, batch_size=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def visualize_images(dataloader, age_in_months, max_batches=1):
    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        fig.suptitle(f"Transformed Images - Age {age_in_months} Months", fontsize=14)

        for i, image in enumerate(images):
            image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            axes[i].imshow(image_np)
            axes[i].axis("off")

        plt.show()

if __name__ == "__main__":
    for age in AGES:
        dataloader = create_dataloader(DATA_DIR, batch_size=4, age_in_months=age)
        visualize_images(dataloader, age_in_months=age)