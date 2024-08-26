import numpy as np
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import albumentations as A
import torch
import utils.config as config
from utils.utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image,
    seed_everything
)

seed_everything()

class CustomCocoDataset(Dataset):
    """
    This class is a custom dataset class that inherits from torch.utils.data.Dataset. 
    It is used to load the COCO dataset format and prepare the data for the YOLOv3 model training.
    It applies augmentations to the images and converts the annotations to YOLOv3 format.
    """
     
    def __init__(self, img_folder, ann_file, augmentations=None, anchors=None, S=[13, 26, 52], C=5):
        self.coco_dataset = CocoDetection(root=img_folder, annFile=ann_file, transform=None)
        self.augmentations = augmentations
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C # number of classes
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, annotations = self.coco_dataset[idx]
        image = np.array(image)
        
        bboxes = []
        category_ids = []

        # Extract bounding boxes and category IDs
        for ann in annotations:
            category_id = ann['category_id'] - 1  # YOLO class IDs start from 0
            bboxes.append(ann['bbox'])
            category_ids.append(category_id)
        
        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image, bboxes=bboxes, category_id=category_ids)
            image = augmented['image']
            bboxes = augmented['bboxes']
            category_ids = augmented['category_id']

        # Convert annotations to YOLO format
        yolo_annotations = self.convert_coco_annotations_to_yolo(bboxes, category_ids, image.shape)
        
        # Prepare targets for YOLOv3 model
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]
        for box in yolo_annotations:
            # Extract the x_center, y_center, width, height, and category_id
            category_id, x_center, y_center, bbox_width, bbox_height = box

            # Calculate iou of bounding box with anchor boxes to determine the best anchor box
            iou_anchors = iou(torch.tensor([bbox_width, bbox_height]), self.anchors)

            # Selecting the best anchor box 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            # At each scale, assigning the bounding box to the best matching anchor box 
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                # Identifying the grid size for the scale 
                S = self.S[scale_idx]

                # Identifying the cell to which the bounding box belongs 
                i, j = int(S * y_center), int(S * x_center) 
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # Check if the anchor box is already assigned 
                if not anchor_taken and not has_anchor[scale_idx]:

                    # Set the probability to 1 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding box relative to the cell 
                    x_cell, y_cell = S * x_center - j, S * y_center - i 

                    # Calculating the width and height of the bounding box relative to the cell 
                    width_cell, height_cell = bbox_width * S, bbox_height * S  

                    # Idnetify the box coordinates 
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    # Assigning the box coordinates to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates

                    # Assigning the class label to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(category_id)

                    # Set the anchor box as assigned for the scale 
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the IoU is greater than the threshold
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)
    

    def convert_coco_annotations_to_yolo(self, bboxes, category_ids, img_size):
        _, height, width = img_size
        yolo_annotations = []
        for bbox, category_id in zip(bboxes, category_ids):
            x_min, y_min, bbox_width, bbox_height = bbox
            x_center = (x_min + bbox_width / 2) / width
            y_center = (y_min + bbox_height / 2) / height
            bbox_width /= width
            bbox_height /= height
            yolo_annotations.append([category_id, x_center, y_center, bbox_width, bbox_height])
        return yolo_annotations

def test_dataloader():
    """
    Visualize the ground truth bounding boxes
    """
    anchors = config.ANCHORS

    val_dataset = CustomCocoDataset(
        img_folder=config.IMG_DIR, 
        ann_file=config.TRAIN_ANNOTATION_FILE,
        augmentations=config.train_transforms,
        anchors=config.ANCHORS,
        S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        C=config.NUM_CLASSES,)
    
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    # Visualize the ground truth bounding boxes
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test_dataloader()