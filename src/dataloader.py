import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.preprocessed_dataset import PreprocessedDataset
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform


# Define curriculum learning transformations
def create_age_based_transform(age):
    return transforms.Compose([
        transforms.Resize((64, 64)),
        VisualAcuityTransform(age),
        ColorPerceptionTransform(age),
        transforms.ToTensor()
    ])

def create_curriculum_dataloaders(dataset, ages, batch_size):
    dataloaders = {}
    for age in ages:
        transform = create_age_based_transform(age)
        preprocessed_dataset = PreprocessedDataset(dataset, transform=transform)
        dataloaders[age] = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloaders

def create_image_progression_dataloader(dataset, ages, batch_size):
    """
    Create a DataLoader where each image is transformed for all ages sequentially.
    """
    transformed_dataset = []

    for image, label in dataset:
        for age in ages:
            # Apply age-specific transformations to the same image
            transform = create_age_based_transform(age)
            transformed_image = transform(image)
            transformed_dataset.append((transformed_image, label))

    return DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Define a static transform for all images (no curriculum learning)
def create_no_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

def create_no_curriculum_dataloader(dataset, batch_size):
    transform = create_no_transform()
    preprocessed_dataset = PreprocessedDataset(dataset, transform=transform)
    return DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Define separate transformations
def create_acuity_transform(age):
    return transforms.Compose([
        transforms.Resize((64, 64)),
        VisualAcuityTransform(age),
        transforms.ToTensor()
    ])


def create_color_transform(age):
    return transforms.Compose([
        transforms.Resize((64, 64)),
        ColorPerceptionTransform(age),
        transforms.ToTensor()
    ])

# Create dataloaders for each transformation
def create_acuity_dataloader(dataset, ages, batch_size):
    dataloaders = {}
    for age in ages:
        transform = create_acuity_transform(age)
        preprocessed_dataset = PreprocessedDataset(dataset, transform=transform)
        dataloaders[age] = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloaders

def create_color_dataloader(dataset, ages, batch_size):
    dataloaders = {}
    for age in ages:
        transform = create_color_transform(age)
        preprocessed_dataset = PreprocessedDataset(dataset, transform=transform)
        dataloaders[age] = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloaders
