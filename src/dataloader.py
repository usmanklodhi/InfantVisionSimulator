from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataset import InfantVisionDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform
import matplotlib.pyplot as plt

def create_dataloader(data_dir, batch_size=1, age_in_months=0, use_transform=True):
    """
    Create a DataLoader for the Infant Vision Dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        age_in_months (int): Age in months to parameterize transformations.
        use_transform (bool): Whether to apply transformations.

    Returns:
        DataLoader: DataLoader object.
    """
    if use_transform:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            VisualAcuityTransform(age_in_months),
            ColorPerceptionTransform(age_in_months),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    dataset = InfantVisionDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def visualize_images(dataloader, age_in_months):
    """
    Visualize a batch of images after applying transformations.

    Args:
        dataloader (DataLoader): Dataloader object.
        age_in_months (int): Age in months for visualization title.
    """
    for batch_idx, (images, _) in enumerate(dataloader):
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        fig.suptitle(f"Transformed Images - Age {age_in_months} Months", fontsize=14)

        for i, image in enumerate(images):
            image_np = image.permute(1, 2, 0).numpy()
            axes[i].imshow(image_np)
            axes[i].axis("off")

        plt.show()
        break  # Only visualize the first batch

if __name__ == "__main__":
    data_dir = "dataset/Test_image_6"  # Path to the image directory
    batch_size = 4
    ages = [0, 2, 3, 6, 12]  # Ages to simulate

    for age in ages:
        dataloader = create_dataloader(data_dir, batch_size, age_in_months=age)
        visualize_images(dataloader, age_in_months=age)
