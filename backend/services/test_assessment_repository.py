from backend.services.model_repository import get_model_by_checkpoint
from backend.services.assessment_repository import (
    create_assessment,
    add_assessment_image,
    add_prediction,
    add_class_statistics,
)


CHECKPOINT = "best_model_change_aware.pth"


def main():
    # ---------------------------------------------------------
    # 1. Get the registered model
    # ---------------------------------------------------------

    model = get_model_by_checkpoint(CHECKPOINT)

    if not model:
        raise RuntimeError(
            "Change-Aware U-Net model was not found."
        )

    model_id = model["id"]

    print("Model found:")
    print(model)
    print()


    # ---------------------------------------------------------
    # 2. Create assessment
    # ---------------------------------------------------------

    assessment = create_assessment(
        user_id=None,
        model_id=model_id,
        status="completed",
        damage_level="LOW",
        damage_percentage=0.1846,
        total_pixels=65536,
        damage_pixels=121,
    )

    assessment_id = assessment["id"]

    print("Assessment created:")
    print(assessment)
    print()


    # ---------------------------------------------------------
    # 3. Add before image metadata
    # ---------------------------------------------------------

    before_image = add_assessment_image(
        assessment_id=assessment_id,
        image_type="before",
        filename="test_before.png",
        storage_path=None,
        content_type="image/png",
        image_format="PNG",
        width=1024,
        height=1024,
        size_bytes=1000000,
    )

    print("Before image created:")
    print(before_image)
    print()


    # ---------------------------------------------------------
    # 4. Add after image metadata
    # ---------------------------------------------------------

    after_image = add_assessment_image(
        assessment_id=assessment_id,
        image_type="after",
        filename="test_after.png",
        storage_path=None,
        content_type="image/png",
        image_format="PNG",
        width=1024,
        height=1024,
        size_bytes=1000000,
    )

    print("After image created:")
    print(after_image)
    print()


    # ---------------------------------------------------------
    # 5. Add prediction
    # ---------------------------------------------------------

    prediction = add_prediction(
        assessment_id=assessment_id,
        mask_width=256,
        mask_height=256,
        predicted_classes=[0, 1, 3, 4],
    )

    print("Prediction created:")
    print(prediction)
    print()


    # ---------------------------------------------------------
    # 6. Add class statistics
    # ---------------------------------------------------------

    statistics = {
        0: {
            "class_name": "background",
            "pixels": 64200,
            "percentage": 97.9614,
        },
        1: {
            "class_name": "no_damage",
            "pixels": 1215,
            "percentage": 1.8539,
        },
        2: {
            "class_name": "minor_damage",
            "pixels": 0,
            "percentage": 0.0,
        },
        3: {
            "class_name": "major_damage",
            "pixels": 68,
            "percentage": 0.1038,
        },
        4: {
            "class_name": "destroyed",
            "pixels": 53,
            "percentage": 0.0809,
        },
    }

    class_statistics = add_class_statistics(
        assessment_id=assessment_id,
        statistics=statistics,
    )

    print("Class statistics created:")
    print(class_statistics)
    print()

    print("=" * 60)
    print("ASSESSMENT REPOSITORY TEST PASSED")
    print("=" * 60)
    print()
    print("Assessment ID:", assessment_id)
    print("Model ID:", model_id)


if __name__ == "__main__":
    main()