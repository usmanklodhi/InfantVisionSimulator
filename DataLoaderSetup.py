def get_dataloaders(image_paths, batch_size=32):
    # Create datasets for grayscale and red-green stages
    grayscale_dataset = InfantVisionDataset(image_paths, stage='grayscale')
    red_green_dataset = InfantVisionDataset(image_paths, stage='red-green')

    # Set up DataLoaders
    grayscale_loader = DataLoader(grayscale_dataset, batch_size=batch_size, shuffle=True)
    red_green_loader = DataLoader(red_green_dataset, batch_size=batch_size, shuffle=True)

    return grayscale_loader, red_green_loader