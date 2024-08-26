import os
import json
import random
from collections import defaultdict
from pycocotools.coco import COCO

def split_dataset(data_dir, dataType, split_ratio=0.8, seed=0):
    annFile = f'{data_dir}/instances_{dataType}_subset.json'
    coco = COCO(annFile)
    
    # Split ratios
    train_ratio = split_ratio
    val_ratio = 1 - train_ratio
    
    # Output directories
    output_dirs = {
        "train": "coco_train_split",
        "val": "coco_val_split",
    }
    
    # Get all image IDs and create a dictionary to track annotations by image
    imgIds = coco.getImgIds()
    img_annots = defaultdict(list)
    for imgId in imgIds:
        annIds = coco.getAnnIds(imgIds=[imgId])
        for annId in annIds:
            ann = coco.loadAnns(annId)[0]
            img_annots[imgId].append(ann)
    
    # Random seed for reproducibility
    random.seed(seed)
    
    # Shuffle image IDs
    random.shuffle(imgIds)
    
    # Split image IDs by ratio
    num_imgs = len(imgIds)
    num_val = int(val_ratio * num_imgs)
    num_train = num_imgs - num_val
    
    train_imgIds = imgIds[:num_train]
    val_imgIds = imgIds[num_train:]
    
    # Datasets to store annotations
    datasets = {
        "train": {"images": [], "annotations": [], "categories": coco.loadCats(coco.getCatIds())},
        "val": {"images": [], "annotations": [], "categories": coco.loadCats(coco.getCatIds())},
    }
    
    # Category counts
    category_counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
    }
    
    # Assign images and annotations to each set
    for split_name, imgIds in zip(["train", "val"], [train_imgIds, val_imgIds]):
        for imgId in imgIds:
            img = coco.loadImgs(imgId)[0]
            datasets[split_name]['images'].append(img)
            for ann in img_annots[imgId]:
                datasets[split_name]['annotations'].append(ann)
                category_counts[split_name][ann['category_id']] += 1
    
    # Save the split datasets
    for split_name, data in datasets.items():
        output_dir = os.path.join(data_dir, "annotations")
        with open(os.path.join(output_dir, f'instances_{dataType}_{split_name}.json'), 'w') as f:
            json.dump(data, f)
    
    # Print category counts
    print("Category counts in training set:")
    for cat_id, count in category_counts["train"].items():
        cat_name = coco.loadCats(cat_id)[0]['name']
        print(f"{cat_name}: {count}")
    
    print("Category counts in validation set:")
    for cat_id, count in category_counts["val"].items():
        cat_name = coco.loadCats(cat_id)[0]['name']
        print(f"{cat_name}: {count}")
    
    print("Dataset successfully split into training and validation sets.")
    print(f"Training set: {len(datasets['train']['images'])} images, {len(datasets['train']['annotations'])} annotations")
    print(f"Validation set: {len(datasets['val']['images'])} images, {len(datasets['val']['annotations'])} annotations")

# Usage
data_dir = 'coco'
dataType = 'train2017'
split_ratio = 0.8  # 80% train, 20% val
split_dataset(data_dir, dataType, split_ratio)