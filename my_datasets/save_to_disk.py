from datasets import load_dataset

dataset = load_dataset("zh-plus/tiny-imagenet")
dataset.save_to_disk("./tiny-imagenet")
