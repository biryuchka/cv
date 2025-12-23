import os
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor
from PIL import Image
import torch



class CocoSubsetDataset(Dataset):
    def __init__(self, root, annFile, processor, target_cat_ids, original_id_to_new_id, train=True):
        self.root = root
        self.coco = COCO(annFile)
        self.processor = processor
        self.target_cat_ids = target_cat_ids
        self.original_id_to_new_id = original_id_to_new_id
        
        self.img_ids = []
        for cat_id in target_cat_ids:
            self.img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
        self.img_ids = list(set(self.img_ids))
        print(f"Found {len(self.img_ids)} images with target classes.")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        file_name = img_info["file_name"]

        if file_name.startswith("synthetic_placeholder/"):
            image_path = os.path.join(
                self.root.replace("train2017", "train2017_synthetic"),
                file_name.replace("synthetic_placeholder/", "")
            )
        else:
            image_path = os.path.join(self.root, file_name)

        image = Image.open(image_path).convert("RGB")
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        target_anns = [ann for ann in anns if ann['category_id'] in self.target_cat_ids]
        
        coco_annotations = []
        for ann in target_anns:
            x, y, w, h = ann['bbox']
            coco_annotations.append({
                'bbox': [x, y, w, h],
                'category_id': ann['category_id'], 
                'area': w * h,
                'iscrowd': ann.get('iscrowd', 0)
            })
        
        target = {
            'image_id': img_id,
            'annotations': coco_annotations
        }
        
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        if len(labels['class_labels']) > 0:
            remapped_labels = torch.tensor([
                self.original_id_to_new_id[cid.item()] 
                for cid in labels['class_labels']
            ], dtype=torch.int64)
            labels['class_labels'] = remapped_labels
        
        return pixel_values, labels

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    max_h = max([img.shape[1] for img in pixel_values])
    max_w = max([img.shape[2] for img in pixel_values])
    
    padded_images = []
    for img in pixel_values:
        c, h, w = img.shape
        padded = torch.zeros((c, max_h, max_w), dtype=img.dtype)
        padded[:, :h, :w] = img
        padded_images.append(padded)
    
    return {
        'pixel_values': torch.stack(padded_images),
        'labels': labels
    }

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
