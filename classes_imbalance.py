import argparse
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def analyze_class_imbalance(data_dir: str, split: str = "train"):
    """
    Analyze class imbalance for target categories in COCO dataset.
    
    Args:
        data_dir: Path to COCO data directory
        split: "train" or "val" to analyze train or validation set
    """
    # Target categories
    target_cats_names = [
     'cat',
    'dog',
    'horse',
    'cow',
    'sheep',
    'bus',
    'truck',
    'train',
    'motorcycle',
    'airplane',
    ]
    
    # Load COCO annotations
    ann_file = os.path.join(data_dir, "annotations", f"instances_{split}2017.json")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    
    coco = COCO(ann_file)
    
    # Get target category IDs
    target_cat_ids = coco.getCatIds(catNms=target_cats_names)
    target_cats = coco.loadCats(target_cat_ids)
    
    # Create mapping from original COCO ID to category name
    cat_id_to_name = {cat["id"]: cat["name"] for cat in target_cats}
    
    # Count instances per class
    class_counts = defaultdict(int)
    
    # Get all image IDs that contain target categories
    img_ids = []
    for cat_id in target_cat_ids:
        img_ids.extend(coco.getImgIds(catIds=[cat_id]))
    img_ids = list(set(img_ids))
    
    print(f"\nAnalyzing {split.upper()} set...")
    print(f"Total images with target categories: {len(img_ids)}")
    
    # Count annotations per class
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=target_cat_ids)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id in cat_id_to_name:
                class_counts[cat_id_to_name[cat_id]] += 1
    
    # Calculate statistics
    total_instances = sum(class_counts.values())
    class_stats = {}
    
    for cat_name in target_cats_names:
        count = class_counts[cat_name]
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        class_stats[cat_name] = {
            "count": count,
            "percentage": percentage,
        }
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Class Imbalance Analysis - {split.upper()} Set")
    print(f"{'='*60}")
    print(f"{'Class':<15} {'Count':<10} {'Percentage':<12} {'Ratio'}")
    print(f"{'-'*60}")
    
    max_count = max(class_stats.values(), key=lambda x: x["count"])["count"]
    
    for cat_name in target_cats_names:
        stats = class_stats[cat_name]
        ratio = stats["count"] / max_count if max_count > 0 else 0
        print(
            f"{cat_name:<15} {stats['count']:<10} {stats['percentage']:>6.2f}%     "
            f"{ratio:.3f}"
        )
    
    print(f"{'-'*60}")
    print(f"{'TOTAL':<15} {total_instances:<10} {'100.00%':<12}")
    
    # Calculate imbalance metrics
    counts = [stats["count"] for stats in class_stats.values()]
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
        
        print(f"\nImbalance Metrics:")
        print(f"  Min instances: {min_count}")
        print(f"  Max instances: {max_count}")
        print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        # Standard deviation
        mean_count = sum(counts) / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        std_dev = variance ** 0.5
        cv = (std_dev / mean_count * 100) if mean_count > 0 else 0
        print(f"  Mean: {mean_count:.1f}")
        print(f"  Std Dev: {std_dev:.1f}")
        print(f"  Coefficient of Variation: {cv:.2f}%")
    
    return class_stats, target_cats_names


def plot_class_imbalance(class_stats: dict, class_names: list, output_path: str, split: str):
    """
    Plot class imbalance as a bar chart.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    counts = [class_stats[name]["count"] for name in class_names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), counts, color="steelblue", alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylabel("Number of Instances")
    plt.title(f"Class Imbalance - {split.upper()} Set")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=150)
    plt.close()
    print(f"\nSaved plot to {output_path}")


def plot_percentage_distribution(class_stats: dict, class_names: list, output_path: str, split: str):
    """
    Plot class distribution as percentages.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    percentages = [class_stats[name]["percentage"] for name in class_names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), percentages, color="coral", alpha=0.7)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylabel("Percentage (%)")
    plt.title(f"Class Distribution (Percentage) - {split.upper()} Set")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=150)
    plt.close()
    print(f"Saved percentage plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze class imbalance for target categories in COCO dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/alisa/homework2/data/coco",
        help="Path to COCO data directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to analyze (train or val).",
    )
    parser.add_argument(
        "--output_count",
        type=str,
        default=None,
        help="Output path for count bar chart SVG (default: classes_imbalance_{split}.svg).",
    )
    parser.add_argument(
        "--output_percentage",
        type=str,
        default=None,
        help="Output path for percentage bar chart SVG (default: classes_imbalance_percentage_{split}.svg).",
    )
    parser.add_argument(
        "--both_splits",
        action="store_true",
        help="Analyze both train and validation splits.",
    )
    
    args = parser.parse_args()
    
    splits_to_analyze = ["train", "val"] if args.both_splits else [args.split]
    
    for split in splits_to_analyze:
        class_stats, class_names = analyze_class_imbalance(args.data_dir, split=split)
        
        # Generate output paths
        if args.output_count is None:
            output_count = f"/home/alisa/cv_hw2/classes_imbalance_{split}.svg"
        else:
            output_count = args.output_count.replace("{split}", split)
        
        if args.output_percentage is None:
            output_percentage = f"/home/alisa/cv_hw2/classes_imbalance_percentage_{split}.svg"
        else:
            output_percentage = args.output_percentage.replace("{split}", split)
        
        # Plot results
        plot_class_imbalance(class_stats, class_names, output_count, split)
        plot_percentage_distribution(class_stats, class_names, output_percentage, split)


if __name__ == "__main__":
    main()

