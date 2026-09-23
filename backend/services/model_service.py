import os

import numpy as np
import torch

from ai.unet_change_aware import UNetChangeAware


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "checkpoints",
    "best_model_change_aware.pth"
)

NUM_CLASSES = 5

CLASS_NAMES = {
    0: "background",
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed"
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL SERVICE
# ============================================================

class DamageModelService:

    def __init__(self):

        print(
            "Loading damage assessment model..."
        )

        print(
            f"Device: {DEVICE}"
        )

        print(
            f"Checkpoint: {CHECKPOINT_PATH}"
        )

        if not os.path.exists(
            CHECKPOINT_PATH
        ):

            raise FileNotFoundError(
                "Model checkpoint not found: "
                f"{CHECKPOINT_PATH}"
            )

        # Create model
        self.model = UNetChangeAware(
            in_channels=9,
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        # Load checkpoint
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        self.device = DEVICE

        self.epoch = checkpoint.get(
            "epoch"
        )

        self.validation_loss = checkpoint.get(
            "val_loss"
        )

        print(
            "Model loaded successfully."
        )

        print(
            f"Checkpoint epoch: {self.epoch}"
        )

        print(
            f"Validation loss: "
            f"{self.validation_loss}"
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        input_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Run model inference.

        Expected input:
        [1, 9, 256, 256]

        Returns:
        [256, 256] class mask.
        """

        input_tensor = input_tensor.to(
            self.device
        )

        with torch.no_grad():

            outputs = self.model(
                input_tensor
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

        prediction_mask = (
            predictions[0]
            .cpu()
            .numpy()
        )

        return prediction_mask

    # ========================================================
    # STATISTICS
    # ========================================================

    def calculate_statistics(
        self,
        prediction_mask: np.ndarray
    ) -> dict:

        total_pixels = (
            prediction_mask.size
        )

        class_counts = {}

        for class_id, class_name in (
            CLASS_NAMES.items()
        ):

            count = int(
                np.sum(
                    prediction_mask
                    == class_id
                )
            )

            percentage = (
                count
                /
                total_pixels
                *
                100
            )

            class_counts[
                class_name
            ] = {
                "pixels": count,
                "percentage": round(
                    float(
                        percentage
                    ),
                    4
                )
            }

        # Damage classes:
        # Minor + Major + Destroyed

        damage_mask = (
            (prediction_mask == 2)
            |
            (prediction_mask == 3)
            |
            (prediction_mask == 4)
        )

        damage_pixels = int(
            np.sum(
                damage_mask
            )
        )

        damage_percentage = (
            damage_pixels
            /
            total_pixels
            *
            100
        )

        # Determine simple overall level
        if damage_percentage == 0:
            damage_level = "NONE"

        elif damage_percentage < 5:
            damage_level = "LOW"

        elif damage_percentage < 15:
            damage_level = "MODERATE"

        elif damage_percentage < 30:
            damage_level = "HIGH"

        else:
            damage_level = "SEVERE"

        return {
            "total_pixels": total_pixels,
            "damage_pixels": damage_pixels,
            "damage_percentage": round(
                float(
                    damage_percentage
                ),
                4
            ),
            "damage_level": damage_level,
            "classes": class_counts
        }


# ============================================================
# SINGLE MODEL INSTANCE
# ============================================================

damage_model = DamageModelService()