from backend.services.model_repository import (
    create_model,
    get_model_by_checkpoint,
)


MODEL_NAME = "Change-Aware U-Net"
ARCHITECTURE = "U-Net"
CHECKPOINT = "best_model_change_aware.pth"

INPUT_CHANNELS = 9
OUTPUT_CLASSES = 5

EPOCH = 3
VALIDATION_LOSS = 0.4699871812547956


def main():
    existing_model = get_model_by_checkpoint(CHECKPOINT)

    if existing_model:
        print("Model already exists.")
        print("Model ID:", existing_model["id"])
        print("Model:", existing_model)
        return

    model = create_model(
        name=MODEL_NAME,
        architecture=ARCHITECTURE,
        checkpoint=CHECKPOINT,
        input_channels=INPUT_CHANNELS,
        output_classes=OUTPUT_CLASSES,
        epoch=EPOCH,
        validation_loss=VALIDATION_LOSS,
    )

    print("Model registered successfully.")
    print("Model ID:", model["id"])
    print("Model:", model)


if __name__ == "__main__":
    main()
    