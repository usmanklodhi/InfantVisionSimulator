import os
from PIL import Image
from torch.utils.data import Dataset
from PIL import UnidentifiedImageError


class InfantVisionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(data_dir, file_name)
                try:
                    Image.open(img_path).verify()
                    self.image_paths.append(img_path)
                except (UnidentifiedImageError, OSError):
                    print(f"Skipping corrupted file: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path
