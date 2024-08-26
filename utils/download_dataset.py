import os
import requests
import json
import random
from collections import defaultdict
from pycocotools.coco import COCO
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(img, image_dir):
    img_url = img['coco_url']
    try:
        img_data = requests.get(img_url).content
        with open(os.path.join(image_dir, img['file_name']), 'wb') as f:
            f.write(img_data)
        return img['file_name'], None
    except Exception as e:
        return img['file_name'], e

def download_balanced_coco_subset(dataType, categories, max_instances=110):
    annFile = f'annotations/instances_{dataType}.json'
    coco = COCO(annFile)

    output_dir = 'coco'
    image_dir = os.path.join(output_dir, 'images')
    ann_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Get category IDs
    catIds = coco.getCatIds(catNms=categories)
    
    # Dictionary to keep track of the number of instances per category
    cat_instances = defaultdict(int)
    selected_imgIds = set()

    # Filter images and balance the dataset
    for catId in catIds:
        imgIds = coco.getImgIds(catIds=[catId])

        random.seed(0)  # Set a random seed for deterministic shuffle
        random.shuffle(imgIds)

        for imgId in imgIds:
            if cat_instances[catId] >= max_instances:
                break

            anns = coco.loadAnns(coco.getAnnIds(imgIds=[imgId], catIds=[catId], iscrowd=None))
            num_anns = len(anns)
            if cat_instances[catId] + num_anns > max_instances:
                continue  # Skip this image if it would push the category over the max

            selected_imgIds.add(imgId)
            cat_instances[catId] += num_anns

    print(f"Total unique images selected: {len(selected_imgIds)}")
    print(f"Category distribution: {dict(cat_instances)}")

    subset_ann = {'images': [], 'categories': [], 'annotations': []}

    # Download images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_img = {executor.submit(download_image, coco.loadImgs(imgId)[0], image_dir): imgId for imgId in selected_imgIds}

        for future in as_completed(future_to_img):
            imgId = future_to_img[future]
            img = coco.loadImgs(imgId)[0]
            try:
                file_name, error = future.result()
                if error:
                    print(f"Error downloading {file_name}: {error}")
                else:
                    # Save annotation
                    anns = coco.loadAnns(coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None))
                    # ann_file = os.path.join(ann_dir, f"{img['file_name'].split('.')[0]}.json")
                    # with open(ann_file, 'w') as f:
                    #     json.dump(anns, f)

                    # Add to subset annotation
                    subset_ann['images'].append(img)
                    subset_ann['annotations'].extend(anns)
            except Exception as e:
                print(f"Exception for image ID {imgId}: {e}")

    # Add categories to subset annotation
    subset_ann['categories'] = coco.loadCats(catIds)

    # Save subset annotation file
    with open(os.path.join(output_dir, f'instances_{dataType}_subset.json'), 'w') as f:
        json.dump(subset_ann, f)

    print(f"Created subset annotation file: instances_{dataType}_subset.json")
    print(f"Total images downloaded: {len(subset_ann['images'])}")
    print("Category distribution:")
    for cat in subset_ann['categories']:
        cat_anns = [ann for ann in subset_ann['annotations'] if ann['category_id'] == cat['id']]
        print(f"  {cat['name']}: {len(cat_anns)} annotations")

# Usage
dataType = 'train2017'
categories = ['person', 'bicycle', 'car']
max_instances = 6000

download_balanced_coco_subset(dataType, categories, max_instances)
