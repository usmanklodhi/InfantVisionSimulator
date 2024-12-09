# import torch
from scripts import plot_transformed_images as pt
from src import dataloader as dl, preprocessed_dataset as ppd
from datasets import load_dataset

# The following lines load the dataset from HuggingFace
# data = load_dataset("zh-plus/tiny-imagenet")
# train_data, val_data = (data['train'], data['valid'])

# Define transformations for different stages of infant development
# 3 stages: 1 month, 6 months, 12 months
young_transform = pt.create_age_based_transform(1)
mid_transform = pt.create_age_based_transform(6)
old_transform = pt.create_age_based_transform(12)

# Load my_datasets and dataloaders for each stage
def get_data_loader(stage, train_data, batch_size=128):
    if stage == 'young':
        transform = young_transform
    elif stage == 'mid':
        transform = mid_transform
    elif stage == 'old':
        transform = old_transform
    else:
        raise ValueError(f"Invalid stage: {stage}")

    dataset = ppd.PreprocessedDataset(train_data, transform=transform)
    dataloader = dl.create_dataloader_v3(dataset, batch_size=batch_size)
    return dataloader

if __name__ == '__main__':
    data = load_dataset("zh-plus/tiny-imagenet")
    train_data, val_data = (data['train'], data['valid'])

    # # Define transformations
    # young_transform = pt.create_age_based_transform(1)
    # mid_transform = pt.create_age_based_transform(6)
    # old_transform = pt.create_age_based_transform(12)

    young_dataloader = get_data_loader('young', train_data)
    mid_dataloader = get_data_loader('mid', train_data)
    old_dataloader = get_data_loader('old', train_data)

    # Validation dataloader
    val_dataset = ppd.PreprocessedDataset(val_data, transform=pt.create_age_based_transform(12))
    val_dataloader = dl.create_dataloader_v3(val_dataset)

    # Example loop to test the dataloaders
    for i, (images, labels) in enumerate(young_dataloader):
        if i >= 1:
            break
        print(f"Batch {i}: {images.shape}, {labels.shape}")
