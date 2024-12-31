# config.py
# DATA_DIR for mac should be 'my_datasets' and for windows '../my_datasets'
## Project PART 1
# DATA_DIR = "../my_datasets"
# OUTPUT_DIR = "output_images/"
# NUM_IMAGES = 100
AGES = [3, 6, 9, 12]

## Project PART 2
DATASET_PATH="my_datasets/tiny-imagenet-200"
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_CLASSES = 200  # Tiny ImageNet has 200 classes
DEVICE = "cpu"