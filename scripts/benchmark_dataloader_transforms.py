import time
from src.dataloader import create_dataloader
import matplotlib.pyplot as plt
from config import DATA_DIR, AGES, NUM_IMAGES

def measure_dataloader_time(data_dir, age, num_images, use_transform):
    dataloader = create_dataloader(data_dir, batch_size=1, age_in_months=age, use_transform=use_transform)
    start_time = time.time()
    for i, (image, _) in enumerate(dataloader):
        if i >= num_images - 1:
            break
    return time.time() - start_time # Return the time taken

def test_performance(data_dir, ages, num_images=100):
    performance = {"ages": [], "without_transform": [], "with_transform": []}

    print(f"Testing performance for {num_images} images...\n")

    for age in ages:
        print(f"=== Testing for Age: {age} months ===")
        # Measure time without transformations
        no_transform_time = measure_dataloader_time(data_dir, age, num_images, use_transform=False)
        print(f"Time Taken (No Transform): {no_transform_time:.4f} seconds")

        # Measure time with transformations
        with_transform_time = measure_dataloader_time(data_dir, age, num_images, use_transform=True)
        print(f"Time Taken (With Transform): {with_transform_time:.4f} seconds")

        # Store performance data
        performance["ages"].append(age)
        performance["without_transform"].append(no_transform_time)
        performance["with_transform"].append(with_transform_time)

    return performance # Returns the performance data as a dictionary

def plot_performance(performance, num_images):
    ages = performance["ages"]
    without_transform = performance["without_transform"]
    with_transform = performance["with_transform"]

    plt.figure(figsize=(10, 6))
    plt.plot(ages, without_transform, label="Without Transform", marker='o')
    plt.plot(ages, with_transform, label="With Transform", marker='o')

    plt.title(f"Performance Comparison by Age ({num_images} Images)", fontsize=14)
    plt.xlabel("Age (Months)", fontsize=12)
    plt.ylabel("Time (Seconds)", fontsize=12)
    plt.xticks(ages)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = DATA_DIR
    ages = AGES
    num_images = NUM_IMAGES

    performance = test_performance(data_dir, ages, num_images=num_images)
    plot_performance(performance, num_images)
