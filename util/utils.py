# utils.py
import os
import matplotlib.pyplot as plt
from torchvision import transforms

def save_image_as_plot(images, titles, suptitle, save_path, figsize=(15, 6), title_fontsize=12):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold')

    for ax, img, title in zip(axes, images, titles):
        img_np = transforms.ToTensor()(img).permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(title, fontsize=title_fontsize)
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image grid to {save_path}")
