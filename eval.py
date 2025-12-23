import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DetrForObjectDetection

from dataset import CocoSubsetDataset, collate_fn, processor
from pycocotools.coco import COCO


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from center-x, center-y, width, height (cxcywh)
    to x_min, y_min, x_max, y_max (xyxy). Boxes are expected to be normalized.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes in xyxy format.
    boxes1: [N, 4], boxes2: [M, 4]
    Returns: [N, M] IoU matrix.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def evaluate_per_class_accuracy(
    model: DetrForObjectDetection,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    iou_threshold: float = 0.5,
) -> Tuple[List[float], List[int], List[int], torch.Tensor, Dict[str, Dict[str, int]]]:
    """
    Compute per-class accuracy for object detection.

    For each ground-truth box of a given class, we check if there exists
    a predicted box of the same class with IoU >= iou_threshold.

    Returns:
        per_class_accuracy: list of length num_classes
        correct_per_class: list of length num_classes
        total_per_class: list of length num_classes
        confusion: [num_classes, num_classes] tensor (rows = GT, cols = Pred)
        error_stats: dict with FP/FN breakdown per class
    """
    model.eval()
    correct_per_class = [0 for _ in range(num_classes)]
    total_per_class = [0 for _ in range(num_classes)]

    # confusion[gt, pred]
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    # detailed error stats per class
    error_stats: Dict[str, Dict[str, int]] = {
        str(c): {
            "tp": 0,
            "fn_missed": 0,          # no prediction at all for this GT
            "fn_loc": 0,             # prediction with IoU < threshold but correct class
            "fn_misclass": 0,        # best prediction IoU>=thr but wrong class
            "fp": 0,                 # prediction with no matching GT
        }
        for c in range(num_classes)
    }

    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels_batch = batch["labels"]

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits  # [B, num_queries, num_classes+1]
            pred_boxes = outputs.pred_boxes  # [B, num_queries, 4] in cxcywh, normalized

            probs = softmax(logits)
            scores, pred_classes = probs[..., :-1].max(-1)  # ignore "no-object" class

            for b in range(pixel_values.shape[0]):
                labels = labels_batch[b]
                gt_classes = labels["class_labels"].to(device)  # [N_gt]
                gt_boxes_cxcywh = labels["boxes"].to(device)  # [N_gt, 4], normalized

                if gt_boxes_cxcywh.numel() == 0:
                    continue

                gt_boxes_xyxy = cxcywh_to_xyxy(gt_boxes_cxcywh)

                pred_cls_b = pred_classes[b]
                pred_scores_b = scores[b]
                pred_boxes_b_cxcywh = pred_boxes[b]
                pred_boxes_b_xyxy = cxcywh_to_xyxy(pred_boxes_b_cxcywh)

                # Filter low-confidence predictions
                keep = pred_scores_b > 0.5
                pred_cls_b = pred_cls_b[keep]
                pred_scores_b = pred_scores_b[keep]
                pred_boxes_b_xyxy = pred_boxes_b_xyxy[keep]

                # If no predictions survive threshold, all GTs become missed
                if pred_boxes_b_xyxy.numel() == 0:
                    for idx_gt in range(gt_classes.shape[0]):
                        c_gt = int(gt_classes[idx_gt].item())
                        total_per_class[c_gt] += 1
                        error_stats[str(c_gt)]["fn_missed"] += 1
                    continue

                # Compute IoU between all GT and all predictions
                ious = box_iou(gt_boxes_xyxy, pred_boxes_b_xyxy)  # [N_gt, N_pred]

                # For matching, track which predictions are already used
                matched_pred = torch.zeros(pred_boxes_b_xyxy.shape[0], dtype=torch.bool, device=device)

                # Per-GT matching and error type
                for idx_gt in range(gt_classes.shape[0]):
                    c_gt = int(gt_classes[idx_gt].item())
                    total_per_class[c_gt] += 1

                    iou_row = ious[idx_gt]  # [N_pred]
                    best_iou, best_pred_idx = (iou_row.max(dim=0))

                    if best_iou < 0.001:
                        # No overlap with any prediction
                        error_stats[str(c_gt)]["fn_missed"] += 1
                        continue

                    c_pred = int(pred_cls_b[best_pred_idx].item())
                    confusion[c_gt, c_pred] += 1

                    if best_iou >= iou_threshold:
                        if c_pred == c_gt:
                            correct_per_class[c_gt] += 1
                            error_stats[str(c_gt)]["tp"] += 1
                            matched_pred[best_pred_idx] = True
                        else:
                            # Good IoU but wrong class
                            error_stats[str(c_gt)]["fn_misclass"] += 1
                            matched_pred[best_pred_idx] = True
                    else:
                        # Some overlap but IoU too low
                        if c_pred == c_gt:
                            error_stats[str(c_gt)]["fn_loc"] += 1
                        else:
                            error_stats[str(c_gt)]["fn_misclass"] += 1

                # Remaining unmatched predictions are false positives
                for idx_pred in range(pred_boxes_b_xyxy.shape[0]):
                    if matched_pred[idx_pred]:
                        continue
                    c_pred = int(pred_cls_b[idx_pred].item())
                    if 0 <= c_pred < num_classes:
                        error_stats[str(c_pred)]["fp"] += 1

    per_class_accuracy = []
    for c in range(num_classes):
        if total_per_class[c] > 0:
            per_class_accuracy.append(correct_per_class[c] / total_per_class[c])
        else:
            per_class_accuracy.append(0.0)

    return per_class_accuracy, correct_per_class, total_per_class, confusion, error_stats


def plot_per_class_accuracy(
    accuracies: List[float],
    class_names: List[str],
    output_path: str,
) -> None:
    """
    Plot per-class accuracy as a bar chart and save to an SVG file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    x = range(len(class_names))
    plt.bar(x, accuracies)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Per-class Detection Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_confusion_matrix_all_classes(
    confusion: torch.Tensor,
    class_names: List[str],
    output_path: str,
) -> None:
    """
    Plot full confusion matrix for all classes and save to an SVG file.

    confusion: [num_classes, num_classes] where rows = GT, cols = Pred.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize rows so each GT row sums to 1 for readability
    cm = confusion.to(torch.float32)
    row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
    cm_norm = cm / row_sums

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar()

    num_classes = len(class_names)
    ticks = list(range(num_classes))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix (All Classes, IoU-based matches)")

    # Optionally annotate diagonal and main off-diagonal entries
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j].item()
            if val > 0.05:  # only show reasonably large values
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate DETR model and compute per-class accuracy.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/alisa/homework2/data/coco",
        help="Path to COCO data directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/alisa/cv_hw2/ckpts/detr_epoch_10.pt",
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cuda or cpu).",
    )
    parser.add_argument(
        "--output_svg",
        type=str,
        default="/home/alisa/cv_hw2/per_class_accuracy.svg",
        help="Output path for per-class accuracy SVG plot.",
    )
    parser.add_argument(
        "--output_confusion_all_svg",
        type=str,
        default="/home/alisa/cv_hw2/confusion_all_classes.svg",
        help="Output path for confusion matrix (all classes) SVG plot.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # COCO annotations and category mapping (same as in train.py)
    ann_file_train = os.path.join(args.data_dir, "annotations", "instances_train2017.json")
    ann_file_val = os.path.join(args.data_dir, "annotations", "instances_val2017.json")

    coco_train = COCO(ann_file_train)

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
    'airplane'
    ]
    target_cat_ids = coco_train.getCatIds(catNms=target_cats_names)
    target_cats = coco_train.loadCats(target_cat_ids)

    id2label = {k: v["name"] for k, v in zip(range(10), target_cats)}
    label2id = {v: k for k, v in id2label.items()}
    original_id_to_new_id = {cat_id: i for i, cat_id in enumerate(target_cat_ids)}

    # Model
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=10,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if os.path.isfile(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Checkpoint {args.checkpoint} not found, evaluating base pretrained model.")

    model.to(device)

    # Validation dataset and dataloader
    val_dataset = CocoSubsetDataset(
        root=os.path.join(args.data_dir, "val2017"),
        annFile=ann_file_val,
        processor=processor,
        target_cat_ids=target_cat_ids,
        original_id_to_new_id=original_id_to_new_id,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Evaluating on {len(val_dataset)} validation images...")
    per_class_accuracy, correct_per_class, total_per_class, confusion, error_stats = evaluate_per_class_accuracy(
        model=model,
        dataloader=val_dataloader,
        num_classes=10,
        device=device,
        iou_threshold=0.5,
    )

    print("Per-class accuracy:")
    for i, name in id2label.items():
        acc = per_class_accuracy[i]
        stats = error_stats[str(i)]
        print(
            f"{i} ({name}): acc={acc:.4f} "
            f"(tp={stats['tp']}, "
            f"fn_missed={stats['fn_missed']}, "
            f"fn_loc={stats['fn_loc']}, "
            f"fn_misclass={stats['fn_misclass']}, "
            f"fp={stats['fp']}, "
            f"total_gt={total_per_class[i]})"
        )

    # Plot overall per-class accuracy
    plot_per_class_accuracy(
        accuracies=per_class_accuracy,
        class_names=[id2label[i] for i in range(10)],
        output_path=args.output_svg,
    )
    print(f"Saved per-class accuracy plot to {args.output_svg}")

    # Plot full confusion matrix for all classes
    plot_confusion_matrix_all_classes(
        confusion=confusion,
        class_names=[id2label[i] for i in range(10)],
        output_path=args.output_confusion_all_svg,
    )
    print(f"Saved full confusion matrix plot to {args.output_confusion_all_svg}")


if __name__ == "__main__":
    main()


