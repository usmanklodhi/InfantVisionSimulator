import os
from PIL import Image
from torch.utils.data import Dataset

class InfantVisionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Custom Dataset for Infant Vision project.

        Args:
            data_dir (str): Path to the directory containing images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.lower().endswith(('png', 'jpg', 'jpeg'))
        ]

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve an image and apply transformations.

        Args:
            idx (int): Index of the image.

        Returns:
            Transformed image and its path.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path
