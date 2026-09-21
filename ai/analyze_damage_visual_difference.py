import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"D:\Projects\Datasets\xBD"

VAL_CSV = os.path.join(
    DATASET_ROOT,
    "splits",
    "val.csv"
)

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "train",
    "images"
)

TARGET_DIR = os.path.join(
    DATASET_ROOT,
    "train",
    "targets"
)

OUTPUT_DIR = os.path.join(
    "outputs",
    "damage_visual_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


MODEL_SIZE = 256

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed"
}


# ============================================================
# HELPERS
# ============================================================

def find_image(image_dir, sample_id, suffix):
    """
    Find image using sample ID and suffix.
    """

    filename = f"{sample_id}_{suffix}.png"

    path = os.path.join(
        image_dir,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    return path


def find_target(target_dir, sample_id):
    """
    Find post-disaster target mask.
    """

    filename = f"{sample_id}_post_disaster_target.png"

    path = os.path.join(
        target_dir,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Target not found: {path}"
        )

    return path


def load_rgb(path):
    """
    Load RGB image and resize to model resolution.
    """

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (MODEL_SIZE, MODEL_SIZE),
        Image.Resampling.BILINEAR
    )

    return np.asarray(
        image,
        dtype=np.float32
    ) / 255.0


def load_mask(path):
    """
    Load target mask and resize using nearest neighbor.
    """

    mask = Image.open(path)

    mask = mask.resize(
        (MODEL_SIZE, MODEL_SIZE),
        Image.Resampling.NEAREST
    )

    return np.asarray(mask)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 75)
    print("PRE / POST VISUAL DIFFERENCE ANALYSIS")
    print("=" * 75)

    print()
    print("Dataset root :", DATASET_ROOT)
    print("Validation CSV:", VAL_CSV)
    print("Image size   :", f"{MODEL_SIZE}x{MODEL_SIZE}")
    print()

    df = pd.read_csv(VAL_CSV)

    print("Validation samples:", len(df))
    print()

    results = []

    # --------------------------------------------------------
    # Process each validation sample
    # --------------------------------------------------------

    for index, row in df.iterrows():

        sample_id = str(row.iloc[0])

        pre_path = find_image(
            IMAGE_DIR,
            sample_id,
            "pre_disaster"
        )

        post_path = find_image(
            IMAGE_DIR,
            sample_id,
            "post_disaster"
        )

        target_path = find_target(
            TARGET_DIR,
            sample_id
        )

        pre = load_rgb(pre_path)
        post = load_rgb(post_path)
        target = load_mask(target_path)

        # ----------------------------------------------------
        # Absolute RGB difference
        # ----------------------------------------------------

        difference = np.abs(post - pre)

        # Convert RGB difference into a single magnitude
        difference_magnitude = np.mean(
            difference,
            axis=2
        )

        # ----------------------------------------------------
        # Analyze every class
        # ----------------------------------------------------

        for class_id, class_name in CLASS_NAMES.items():

            class_mask = target == class_id

            pixel_count = int(
                np.sum(class_mask)
            )

            if pixel_count == 0:
                continue

            class_difference = difference_magnitude[
                class_mask
            ]

            result = {
                "sample_id": sample_id,
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": pixel_count,
                "mean_difference": float(
                    np.mean(class_difference)
                ),
                "median_difference": float(
                    np.median(class_difference)
                ),
                "std_difference": float(
                    np.std(class_difference)
                ),
                "min_difference": float(
                    np.min(class_difference)
                ),
                "max_difference": float(
                    np.max(class_difference)
                ),
                "p90_difference": float(
                    np.percentile(
                        class_difference,
                        90
                    )
                ),
            }

            results.append(result)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(df)} samples"
            )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    detailed_path = os.path.join(
        OUTPUT_DIR,
        "damage_visual_difference.csv"
    )

    results_df.to_csv(
        detailed_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
        .groupby(
            ["class_id", "class_name"]
        )
        .agg(
            samples=("sample_id", "count"),
            total_pixels=("pixel_count", "sum"),
            mean_difference=("mean_difference", "mean"),
            median_difference=("median_difference", "mean"),
            std_difference=("std_difference", "mean"),
            p90_difference=("p90_difference", "mean"),
            min_difference=("min_difference", "min"),
            max_difference=("max_difference", "max")
        )
        .reset_index()
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "damage_visual_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("VISUAL DIFFERENCE SUMMARY")
    print("=" * 75)

    for _, row in summary.iterrows():

        print()
        print(row["class_name"])

        print(
            f"  Samples           : {int(row['samples'])}"
        )

        print(
            f"  Total pixels      : {int(row['total_pixels']):,}"
        )

        print(
            f"  Mean difference   : "
            f"{row['mean_difference']:.4f}"
        )

        print(
            f"  Median difference : "
            f"{row['median_difference']:.4f}"
        )

        print(
            f"  P90 difference    : "
            f"{row['p90_difference']:.4f}"
        )

        print(
            f"  Std difference    : "
            f"{row['std_difference']:.4f}"
        )

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(10, 6))

    class_order = [
        "Background",
        "No Damage",
        "Minor Damage",
        "Major Damage",
        "Destroyed"
    ]

    plot_data = []

    for class_name in class_order:

        values = results_df[
            results_df["class_name"] == class_name
        ]["mean_difference"].values

        plot_data.append(values)

    plt.boxplot(
        plot_data,
        labels=class_order
    )

    plt.title(
        "Pre/Post Visual Difference by Ground-Truth Class"
    )

    plt.ylabel(
        "Mean Absolute RGB Difference"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "damage_difference_distribution.png"
    )

    plt.savefig(
        plot_path,
        dpi=200
    )

    plt.close()

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)

    print()
    print("Saved:")
    print(detailed_path)

    print("Saved:")
    print(summary_path)

    print("Saved:")
    print(plot_path)


if __name__ == "__main__":
    main()