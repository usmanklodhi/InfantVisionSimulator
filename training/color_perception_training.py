from src.dataloader import create_color_dataloader
from configuration.setting import AGES, EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES
from training import utils as ut
from training import curriculum as c

def main():
    batch_size = BATCH_SIZE
    total_epochs = EPOCHS
    learning_rate = LEARNING_RATE
    num_classes = NUM_CLASSES
    ages = AGES
    model_names = ["resnet18"]
    model_output_dir = "outputs/models/color_perception/"
    loss_output_dir = "outputs/loss_logs/color_perception/"
    user_epoch_map = None

    c.train_curriculum_model(
        batch_size=batch_size,
        total_epochs=total_epochs,
        learning_rate=learning_rate,
        num_classes=num_classes,
        model_output_dir=model_output_dir,
        loss_output_dir=loss_output_dir,
        model_names=model_names,
        ages=ages,
        user_epoch_map=user_epoch_map,
        curriculum_type="color_perception",
        dataset_loader=ut.load_data,
        dataloader_creator=create_color_dataloader
    )

if __name__ == "__main__":
    main()

