import os

from backend.services.preprocessing import (
    prepare_model_input,
    get_model_input_info
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

    tensor = prepare_model_input(
        before_bytes,
        after_bytes
    )

    info = get_model_input_info(
        tensor
    )

    print("=" * 70)
    print("PREPROCESSING TEST")
    print("=" * 70)

    print()
    print("Tensor shape :", info["shape"])
    print("Tensor dtype :", info["dtype"])
    print("Minimum      :", info["min"])
    print("Maximum      :", info["max"])

    print()
    print("=" * 70)
    print("PREPROCESSING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()