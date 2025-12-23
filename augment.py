# augment.py
import torch
import cv2
import numpy as np
import os
import json
import copy
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from pycocotools.coco import COCO
from tqdm.auto import tqdm

# === КОНФИГУРАЦИЯ ===
DATASET_ROOT = '/home/alisa/homework2/data/coco'
IMG_DIR = os.path.join(DATASET_ROOT, 'train2017')
ANN_FILE = os.path.join(DATASET_ROOT, 'annotations', 'instances_train2017.json')

# Папка для синтетики
SYNTH_DIR = os.path.join(DATASET_ROOT, 'train2017_synthetic')
# Новый JSON, который объединит реальные и синтетические данные
OUTPUT_ANN = os.path.join(DATASET_ROOT, 'annotations', 'instances_augmented.json')

TARGET_CLASSES = ['cat',
    'dog',
    'horse',
    'cow',
    'sheep',
    'bus',
    'truck',
    'train',
    'motorcycle',
    'airplane',]

# Сколько картинок генерировать на 1 реальную (если класс редкий, без явной цели)
AUG_FACTOR = 0.4
# Генерируем только для классов, где меньше N картинок (или для всех, если поставить большое число)
RARE_THRESHOLD = 2000

def get_canny_image(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def main():
    os.makedirs(SYNTH_DIR, exist_ok=True)
    
    print("Loading ControlNet & SD...")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    coco = COCO(ANN_FILE)
    cat_ids = coco.getCatIds(catNms=TARGET_CLASSES)
    cats = coco.loadCats(cat_ids)
    
    with open(ANN_FILE, 'r') as f:
        coco_data = json.load(f)
    
    start_img_id = 10_000_000
    start_ann_id = 10_000_000
    new_images = []
    new_annotations = []
    
    # === Статистика по целевым классам и план синтетики ===
    print("\n=== Class statistics (TARGET_CLASSES) ===")
    img_count = {}
    ann_count = {}
    for cat in cats:
        cat_name = cat["name"]
        img_ids_for_cat = coco.getImgIds(catIds=[cat["id"]])
        ann_ids_for_cat = coco.getAnnIds(catIds=[cat["id"]])
        img_count[cat_name] = len(img_ids_for_cat)
        ann_count[cat_name] = len(ann_ids_for_cat)
        print(
            f"{cat_name}: {len(img_ids_for_cat)} images, "
            f"{len(ann_ids_for_cat)} objects"
        )
    print("=========================================")
    
    desired_total_imgs = {}
    
    print("\n=== Planned synthetic images per class ===")
    for cls_name, cur_n in img_count.items():
        print(f"{cls_name}: no specific target, will use rare-threshold logic.", RARE_THRESHOLD, cur_n)
    print("=========================================\n")
    
    for cat in cats:
        cat_name = cat['name']
        img_ids = coco.getImgIds(catIds=[cat['id']])
        
            # Для остальных классов оставляем старую логику по порогу редкости
        if len(img_ids) >= RARE_THRESHOLD:
            print(f"Skipping {cat_name} (count: {len(img_ids)})")
            continue
        # Если класс редкий, создаём по AUG_FACTOR на каждую реальную
        need_synth = int((RARE_THRESHOLD - len(img_ids)) * AUG_FACTOR)
        print(f"Augmenting rare class {cat_name}: generating {need_synth} synthetic images.")
        
        generated_for_cat = 0
        # Идём по изображениям, пока не добьём нужное количество синтетики
        for img_id in tqdm(img_ids, desc=f"{cat_name}"):
            if generated_for_cat >= need_synth: 
                break
            img_info = coco.loadImgs(img_id)[0]
            path = os.path.join(IMG_DIR, img_info['file_name'])
            try:
                orig_img = Image.open(path).convert("RGB")
            except:
                continue

            canny = get_canny_image(orig_img)
            # Немного особый промпт для велосипеда и машины
            prompt = (
                    f"a photo of a {cat_name}, high quality, realistic "
                    f"the {cat_name} is located in the center of the image"
                )
        
            w, h = orig_img.size
            gen_img = pipe(prompt, image=canny.resize((512,512)), num_inference_steps=20).images[0]
            gen_img = gen_img.resize((w, h)) # Возвращаем размер, чтобы боксы совпали!
            
            filename = f"synth_{start_img_id}.jpg"
            gen_img.save(os.path.join(SYNTH_DIR, filename))
            
            new_img_info = copy.deepcopy(img_info)
            new_img_info['id'] = start_img_id
            new_img_info['file_name'] = f"synthetic_placeholder/{filename}" # Маркер для Dataset класса
            new_images.append(new_img_info)
            
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = start_ann_id
                new_ann['image_id'] = start_img_id
                new_annotations.append(new_ann)
                start_ann_id += 1
            
            start_img_id += 1
            generated_for_cat += 1

    coco_data['images'].extend(new_images)
    coco_data['annotations'].extend(new_annotations)
    
    with open(OUTPUT_ANN, 'w') as f:
        json.dump(coco_data, f)
    print(f"Done. Saved to {OUTPUT_ANN}")

if __name__ == "__main__":
    main()