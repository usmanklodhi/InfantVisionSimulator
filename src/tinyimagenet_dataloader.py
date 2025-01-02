# import os
# import glob
# import torch
# from torch.utils.data import Dataset
# from torchvision.io import read_image, ImageReadMode
# from torchvision.transforms import Normalize
# from configuration.setting import DATASET_PATH
#
#
# # Load ID mappings
# id_dict = {}
# for i, line in enumerate(open(os.path.join(DATASET_PATH, "wnids.txt"), 'r')):
#     id_dict[line.strip()] = i
#
#
# class TrainTinyImageNetDataset(Dataset):
#     def __init__(self, id_dict, transform=None):
#         self.filenames = glob.glob(os.path.join(DATASET_PATH, "train/*/*/*.JPEG"))
#         self.transform = transform
#         self.id_dict = id_dict
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         img_path = self.filenames[idx]
#         image = read_image(img_path, ImageReadMode.RGB)
#         label = self.id_dict[img_path.split(os.sep)[-3]]  # Extract class name from path
#         if self.transform:
#             image = self.transform(image.type(torch.FloatTensor))
#         return image, label
#
#
# class TestTinyImageNetDataset(Dataset):
#     def __init__(self, id_dict, transform=None):
#         self.filenames = glob.glob(os.path.join(DATASET_PATH, "val/images/*.JPEG"))
#         self.transform = transform
#         self.id_dict = id_dict
#         self.cls_dict = {}
#         val_annotations_path = os.path.join(DATASET_PATH, "val/val_annotations.txt")
#         with open(val_annotations_path, 'r') as f:
#             for line in f:
#                 img, cls_id = line.split('\t')[:2]
#                 self.cls_dict[img] = self.id_dict[cls_id]
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         img_path = self.filenames[idx]
#         image = read_image(img_path, ImageReadMode.RGB)
#         label = self.cls_dict[os.path.basename(img_path)]
#         if self.transform:
#             image = self.transform(image.type(torch.FloatTensor))
#         return image, label
#
#
# # Normalization Transform
# transform = Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
#
# # Dataset Instances
# trainset = TrainTinyImageNetDataset(id_dict=id_dict, transform=transform)
# testset = TestTinyImageNetDataset(id_dict=id_dict, transform=transform)
