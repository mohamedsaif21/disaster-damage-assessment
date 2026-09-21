import os
import sys
import csv
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


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
    PROJECT_ROOT,
    "outputs",
    "damage_region_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# Configuration
# =========================================================

ORIGINAL_SIZE = 1024
MODEL_SIZE = 256

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed"
}

DAMAGE_CLASSES = [2, 3, 4]


# =========================================================
# Connected component analysis
# =========================================================

def connected_components(mask):
    """
    Find connected regions using 8-connectivity.

    Returns:
        list of region sizes in pixels
    """

    height, width = mask.shape

    visited = np.zeros(
        (height, width),
        dtype=bool
    )

    region_sizes = []

    # 8-connected neighbors
    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    ]

    for y in range(height):

        for x in range(width):

            if not mask[y, x]:
                continue

            if visited[y, x]:
                continue

            # Start new component
            stack = [(y, x)]
            visited[y, x] = True

            size = 0

            while stack:

                cy, cx = stack.pop()

                size += 1

                for dy, dx in neighbors:

                    ny = cy + dy
                    nx = cx + dx

                    if ny < 0 or ny >= height:
                        continue

                    if nx < 0 or nx >= width:
                        continue

                    if visited[ny, nx]:
                        continue

                    if not mask[ny, nx]:
                        continue

                    visited[ny, nx] = True
                    stack.append((ny, nx))

            region_sizes.append(size)

    return region_sizes


# =========================================================
# Analyze one mask
# =========================================================

def analyze_mask(
    target_array,
    sample_id
):

    results = []

    for class_id in DAMAGE_CLASSES:

        class_mask = (
            target_array == class_id
        )

        pixel_count = int(
            class_mask.sum()
        )

        percentage = (
            pixel_count
            / target_array.size
            * 100.0
        )

        # Connected regions
        if pixel_count > 0:

            regions = connected_components(
                class_mask
            )

        else:

            regions = []

        if regions:

            regions_sorted = sorted(
                regions,
                reverse=True
            )

            region_count = len(
                regions_sorted
            )

            largest_region = (
                regions_sorted[0]
            )

            smallest_region = (
                regions_sorted[-1]
            )

            mean_region = (
                np.mean(regions_sorted)
            )

            median_region = (
                np.median(regions_sorted)
            )

            # Region thresholds
            regions_1px = sum(
                size <= 1
                for size in regions_sorted
            )

            regions_5px = sum(
                size <= 5
                for size in regions_sorted
            )

            regions_10px = sum(
                size <= 10
                for size in regions_sorted
            )

            regions_25px = sum(
                size <= 25
                for size in regions_sorted
            )

            regions_50px = sum(
                size <= 50
                for size in regions_sorted
            )

        else:

            region_count = 0
            largest_region = 0
            smallest_region = 0
            mean_region = 0
            median_region = 0

            regions_1px = 0
            regions_5px = 0
            regions_10px = 0
            regions_25px = 0
            regions_50px = 0

        results.append(
            {
                "sample_id": sample_id,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "pixel_count": pixel_count,
                "percentage": percentage,
                "region_count": region_count,
                "largest_region": largest_region,
                "smallest_region": smallest_region,
                "mean_region": mean_region,
                "median_region": median_region,
                "regions_1px_or_less": regions_1px,
                "regions_5px_or_less": regions_5px,
                "regions_10px_or_less": regions_10px,
                "regions_25px_or_less": regions_25px,
                "regions_50px_or_less": regions_50px
            }
        )

    return results


# =========================================================
# Resize comparison
# =========================================================

def analyze_resize_effect(
    target_array
):
    """
    Compare original 1024x1024 target against
    256x256 nearest-neighbor resized target.

    Returns pixel counts before and after resizing.
    """

    original_counts = {}

    for class_id in range(5):

        original_counts[class_id] = int(
            np.sum(
                target_array == class_id
            )
        )

    # PIL resize with nearest-neighbor
    target_image = Image.fromarray(
        target_array.astype(np.uint8)
    )

    resized_image = target_image.resize(
        (MODEL_SIZE, MODEL_SIZE),
        Image.Resampling.NEAREST
    )

    resized_array = np.array(
        resized_image,
        dtype=np.int64
    )

    resized_counts = {}

    for class_id in range(5):

        resized_counts[class_id] = int(
            np.sum(
                resized_array == class_id
            )
        )

    results = {}

    for class_id in range(5):

        original = original_counts[class_id]
        resized = resized_counts[class_id]

        # Convert resized pixels back to an
        # equivalent number at original scale.
        scaled_resized = resized * 16

        if original > 0:

            retention = (
                scaled_resized
                / original
                * 100.0
            )

        else:

            retention = 0.0

        results[class_id] = {
            "original_pixels": original,
            "resized_pixels": resized,
            "scaled_resized_pixels": scaled_resized,
            "retention_percent": retention
        }

    return results


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 75)
    print("GROUND-TRUTH DAMAGE REGION ANALYSIS")
    print("=" * 75)

    print()
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Validation CSV: {VAL_CSV}")
    print(f"Original size: {ORIGINAL_SIZE}x{ORIGINAL_SIZE}")
    print(f"Model size   : {MODEL_SIZE}x{MODEL_SIZE}")
    print()

    # -----------------------------------------------------
    # Load validation split
    # -----------------------------------------------------

    samples = pd.read_csv(
        VAL_CSV
    )

    print(
        f"Validation samples: {len(samples)}"
    )

    print()

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    all_results = []

    resize_results = []

    class_summary = defaultdict(
        lambda: {
            "samples_with_class": 0,
            "total_pixels": 0,
            "region_count": 0,
            "largest_regions": [],
            "mean_regions": [],
            "median_regions": [],
            "regions_1px_or_less": 0,
            "regions_5px_or_less": 0,
            "regions_10px_or_less": 0,
            "regions_25px_or_less": 0,
            "regions_50px_or_less": 0
        }
    )

    # -----------------------------------------------------
    # Process validation set
    # -----------------------------------------------------

    for index, row in samples.iterrows():

        sample_id = row["sample_id"]

        target_path = os.path.join(
            TARGET_DIR,
            f"{sample_id}_post_disaster_target.png"
        )

        if not os.path.exists(
            target_path
        ):

            print(
                f"WARNING: Missing target: "
                f"{sample_id}"
            )

            continue

        target_image = Image.open(
            target_path
        )

        target_array = np.array(
            target_image,
            dtype=np.int64
        )

        # -------------------------------------------------
        # Check original size
        # -------------------------------------------------

        if target_array.shape != (
            ORIGINAL_SIZE,
            ORIGINAL_SIZE
        ):

            print(
                f"WARNING: Unexpected shape "
                f"for {sample_id}: "
                f"{target_array.shape}"
            )

        # -------------------------------------------------
        # Region analysis
        # -------------------------------------------------

        results = analyze_mask(
            target_array,
            sample_id
        )

        all_results.extend(
            results
        )

        # -------------------------------------------------
        # Aggregate class statistics
        # -------------------------------------------------

        for result in results:

            class_id = result["class_id"]

            summary = class_summary[
                class_id
            ]

            if result["pixel_count"] > 0:

                summary[
                    "samples_with_class"
                ] += 1

            summary[
                "total_pixels"
            ] += result["pixel_count"]

            summary[
                "region_count"
            ] += result["region_count"]

            if result["largest_region"] > 0:

                summary[
                    "largest_regions"
                ].append(
                    result["largest_region"]
                )

            if result["mean_region"] > 0:

                summary[
                    "mean_regions"
                ].append(
                    result["mean_region"]
                )

            if result["median_region"] > 0:

                summary[
                    "median_regions"
                ].append(
                    result["median_region"]
                )

            summary[
                "regions_1px_or_less"
            ] += result[
                "regions_1px_or_less"
            ]

            summary[
                "regions_5px_or_less"
            ] += result[
                "regions_5px_or_less"
            ]

            summary[
                "regions_10px_or_less"
            ] += result[
                "regions_10px_or_less"
            ]

            summary[
                "regions_25px_or_less"
            ] += result[
                "regions_25px_or_less"
            ]

            summary[
                "regions_50px_or_less"
            ] += result[
                "regions_50px_or_less"
            ]

        # -------------------------------------------------
        # Resize analysis
        # -------------------------------------------------

        resize_data = analyze_resize_effect(
            target_array
        )

        for class_id in range(5):

            resize_results.append(
                {
                    "sample_id": sample_id,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    **resize_data[class_id]
                }
            )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (index + 1) % 25 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(samples)} samples"
            )

    # =====================================================
    # Save detailed region results
    # =====================================================

    region_csv = os.path.join(
        OUTPUT_DIR,
        "damage_regions.csv"
    )

    region_fields = [
        "sample_id",
        "class_id",
        "class_name",
        "pixel_count",
        "percentage",
        "region_count",
        "largest_region",
        "smallest_region",
        "mean_region",
        "median_region",
        "regions_1px_or_less",
        "regions_5px_or_less",
        "regions_10px_or_less",
        "regions_25px_or_less",
        "regions_50px_or_less"
    ]

    with open(
        region_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=region_fields
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # =====================================================
    # Save resize results
    # =====================================================

    resize_csv = os.path.join(
        OUTPUT_DIR,
        "resize_effect.csv"
    )

    resize_fields = [
        "sample_id",
        "class_id",
        "class_name",
        "original_pixels",
        "resized_pixels",
        "scaled_resized_pixels",
        "retention_percent"
    ]

    with open(
        resize_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=resize_fields
        )

        writer.writeheader()

        writer.writerows(
            resize_results
        )

    # =====================================================
    # Save class summary
    # =====================================================

    summary_rows = []

    total_image_pixels = (
        ORIGINAL_SIZE * ORIGINAL_SIZE
    )

    for class_id in DAMAGE_CLASSES:

        summary = class_summary[
            class_id
        ]

        total_pixels = summary[
            "total_pixels"
        ]

        percentage_of_all_pixels = (
            total_pixels
            / (
                len(samples)
                * total_image_pixels
            )
            * 100.0
        )

        largest_regions = summary[
            "largest_regions"
        ]

        mean_regions = summary[
            "mean_regions"
        ]

        median_regions = summary[
            "median_regions"
        ]

        summary_rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "samples_with_class": summary[
                    "samples_with_class"
                ],
                "total_pixels": total_pixels,
                "percentage_of_all_pixels":
                    percentage_of_all_pixels,
                "total_regions": summary[
                    "region_count"
                ],
                "average_largest_region":
                    np.mean(largest_regions)
                    if largest_regions
                    else 0,
                "maximum_region":
                    max(largest_regions)
                    if largest_regions
                    else 0,
                "average_region_size":
                    np.mean(mean_regions)
                    if mean_regions
                    else 0,
                "average_median_region":
                    np.mean(median_regions)
                    if median_regions
                    else 0,
                "regions_1px_or_less":
                    summary[
                        "regions_1px_or_less"
                    ],
                "regions_5px_or_less":
                    summary[
                        "regions_5px_or_less"
                    ],
                "regions_10px_or_less":
                    summary[
                        "regions_10px_or_less"
                    ],
                "regions_25px_or_less":
                    summary[
                        "regions_25px_or_less"
                    ],
                "regions_50px_or_less":
                    summary[
                        "regions_50px_or_less"
                    ]
            }
        )

    summary_csv = os.path.join(
        OUTPUT_DIR,
        "damage_region_summary.csv"
    )

    summary_fields = list(
        summary_rows[0].keys()
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # =====================================================
    # Print summary
    # =====================================================

    print()
    print("=" * 75)
    print("DAMAGE REGION SUMMARY")
    print("=" * 75)

    for row in summary_rows:

        print()
        print(
            f"{row['class_name']}"
        )

        print(
            f"  Samples containing class : "
            f"{row['samples_with_class']}"
        )

        print(
            f"  Total pixels             : "
            f"{row['total_pixels']:,}"
        )

        print(
            f"  % of validation pixels  : "
            f"{row['percentage_of_all_pixels']:.4f}%"
        )

        print(
            f"  Total regions            : "
            f"{row['total_regions']:,}"
        )

        print(
            f"  Average region size     : "
            f"{row['average_region_size']:.2f}"
        )

        print(
            f"  Average median region   : "
            f"{row['average_median_region']:.2f}"
        )

        print(
            f"  Maximum region           : "
            f"{row['maximum_region']:,}"
        )

        print(
            f"  Regions <= 1 pixel       : "
            f"{row['regions_1px_or_less']:,}"
        )

        print(
            f"  Regions <= 5 pixels      : "
            f"{row['regions_5px_or_less']:,}"
        )

        print(
            f"  Regions <= 10 pixels     : "
            f"{row['regions_10px_or_less']:,}"
        )

        print(
            f"  Regions <= 25 pixels     : "
            f"{row['regions_25px_or_less']:,}"
        )

        print(
            f"  Regions <= 50 pixels     : "
            f"{row['regions_50px_or_less']:,}"
        )

    # =====================================================
    # Resize summary
    # =====================================================

    print()
    print("=" * 75)
    print("1024 → 256 RESIZE RETENTION")
    print("=" * 75)

    for class_id in DAMAGE_CLASSES:

        class_rows = [
            row
            for row in resize_results
            if row["class_id"] == class_id
            and row["original_pixels"] > 0
        ]

        if not class_rows:
            continue

        retentions = [
            row["retention_percent"]
            for row in class_rows
        ]

        print()
        print(
            f"{CLASS_NAMES[class_id]}"
        )

        print(
            f"  Samples containing class : "
            f"{len(class_rows)}"
        )

        print(
            f"  Average retention        : "
            f"{np.mean(retentions):.2f}%"
        )

        print(
            f"  Minimum retention        : "
            f"{np.min(retentions):.2f}%"
        )

        print(
            f"  Maximum retention        : "
            f"{np.max(retentions):.2f}%"
        )

        print(
            f"  Samples with 0 resized "
            f"pixels: "
            f"{sum(r == 0 for r in retentions)}"
        )

    # =====================================================
    # Finish
    # =====================================================

    print()
    print("=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)

    print()
    print(
        f"Saved:\n{region_csv}"
    )

    print(
        f"Saved:\n{summary_csv}"
    )

    print(
        f"Saved:\n{resize_csv}"
    )


if __name__ == "__main__":
    main()