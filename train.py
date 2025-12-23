from transformers import DetrForObjectDetection, DeformableDetrForObjectDetection, DetrImageProcessor
from transformers import AutoModelForObjectDetection
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import numpy as np
import os
import time
from dataset import CocoSubsetDataset, collate_fn, processor
from pycocotools.coco import COCO
from transformers import Trainer, TrainingArguments

data_dir = "/home/alisa/homework2/data/coco"

ann_file_train = os.path.join(data_dir, "annotations", "instances_train2017.json")
ann_file_val = os.path.join(data_dir, "annotations", "instances_val2017.json")

coco_train = COCO(ann_file_train)
coco_val = COCO(ann_file_val)

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
target_cat_ids = coco_train.getCatIds(catNms=target_cats_names)
target_cats = coco_train.loadCats(target_cat_ids)

for cat in target_cats:
    print(f"ID: {cat['id']}, Name: {cat['name']}")

id2label = {k: v['name'] for k, v in zip(range(10), target_cats)}
label2id = {v: k for k, v in id2label.items()}
original_id_to_new_id = {cat_id: i for i, cat_id in enumerate(target_cat_ids)}

print(f"ID Mapping: {original_id_to_new_id}")

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=10,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

model.to('cuda')

train_dataset = CocoSubsetDataset(
    root=os.path.join(data_dir, "train2017"),
    annFile=ann_file_train,
    processor=processor,
    target_cat_ids=target_cat_ids,
    original_id_to_new_id=original_id_to_new_id
)

val_dataset = CocoSubsetDataset(
    root=os.path.join(data_dir, "val2017"),
    annFile=ann_file_val,
    processor=processor,
    target_cat_ids=target_cat_ids,
    original_id_to_new_id=original_id_to_new_id
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

from trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    device='cuda',
    epochs=20,
    checkpoint_dir="ckpts_new_targets",
    log_dir="logs_new_targets",
)

trainer.train()
