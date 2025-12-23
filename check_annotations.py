import json
from pathlib import Path


def main():
    """
    Small helper script to peek into the augmented COCO annotations.

    It will:
    - Load `instances_augmented.json`
    - Print a few sample images and their annotations so you can see
      how augmented images were saved.
    """

    # Adjust this path if your augmented annotations live somewhere else
    ann_path = Path('/home/alisa/homework2/data/coco/annotations/instances_train2017.json')

    if not ann_path.exists():
        print(f"Annotation file not found at: {ann_path}")
        return

    print(f"Loading annotations from: {ann_path}")
    with ann_path.open("r") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    if not images:
        print("No images found in annotations file.")
        return

    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")

    # Index annotations by image_id for quick lookup
    anns_by_image = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is not None:
            anns_by_image.setdefault(img_id, []).append(ann)

    # Print a few sample images and their annotations
    num_samples = 5
    print(f"\nShowing first {num_samples} images and their annotations:\n")

    for img in images[-num_samples:]:
        img_id = img.get("id")
        file_name = img.get("file_name")

        print("=" * 80)
        print(f"Image ID: {img_id}")
        print(f"File name: {file_name}")

        img_anns = anns_by_image.get(img_id, [])
        print(f"Number of annotations: {len(img_anns)}")

        for i, ann in enumerate(img_anns[:10]):  # limit annotations per image
            print(f"  Annotation {i + 1}:")
            print(f"    category_id: {ann.get('category_id')}")
            print(f"    bbox: {ann.get('bbox')}")
            print(f"    area: {ann.get('area')}")
            print(f"    iscrowd: {ann.get('iscrowd')}")

    print("\nDone.")


if __name__ == "__main__":
    main()


