import os

from backend.services.preprocessing import (
    prepare_model_input
)

from backend.services.model_service import (
    damage_model
)


DATASET_ROOT = r"D:\Projects\Datasets\xBD"

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "train",
    "images"
)


def main():

    before_path = os.path.join(
        IMAGE_DIR,
        "hurricane-michael_00000341_pre_disaster.png"
    )

    after_path = os.path.join(
        IMAGE_DIR,
        "hurricane-michael_00000341_post_disaster.png"
    )

    with open(
        before_path,
        "rb"
    ) as f:

        before_bytes = f.read()

    with open(
        after_path,
        "rb"
    ) as f:

        after_bytes = f.read()

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_tensor = prepare_model_input(
        before_bytes,
        after_bytes
    )

    print()
    print(
        "Input shape:",
        input_tensor.shape
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction_mask = damage_model.predict(
        input_tensor
    )

    print(
        "Prediction shape:",
        prediction_mask.shape
    )

    print(
        "Predicted classes:",
        sorted(
            set(
                prediction_mask.flatten()
                .tolist()
            )
        )
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    statistics = (
        damage_model.calculate_statistics(
            prediction_mask
        )
    )

    print()
    print("=" * 70)
    print("MODEL PREDICTION")
    print("=" * 70)

    print()

    print(
        "Damage pixels:",
        statistics["damage_pixels"]
    )

    print(
        "Damage percentage:",
        statistics["damage_percentage"],
        "%"
    )

    print(
        "Damage level:",
        statistics["damage_level"]
    )

    print()

    print("Class statistics:")

    for class_name, info in (
        statistics["classes"].items()
    ):

        print(
            f"  {class_name:<15}"
            f"{info['percentage']:>8.4f}%"
            f"  ({info['pixels']:,} pixels)"
        )


if __name__ == "__main__":
    main()