import torch
import utils.config as config
from utils.model import YOLOv3
from utils.utils import seed_everything, plot_couple_examples, get_evaluation_bboxes, mean_average_precision
from utils.dataloader import CustomCocoDataset
from torch.utils.data import DataLoader, Subset

seed_everything()


def main():
    """ 
    Given a trained model, evaluate the model on the validation set and calculate the mean average precision (mAP)
    Visualize the model output bounding boxes and classes
    """

    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    val_dataset = CustomCocoDataset(
        img_folder=config.IMG_DIR, 
        ann_file=config.VAL_ANNOTATION_FILE,
        augmentations=config.test_transforms,
        anchors=config.ANCHORS,
        S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        C=config.NUM_CLASSES,)
    
    S = [13, 26, 52]

    scaled_anchors = torch.tensor(config.ANCHORS).to(config.DEVICE) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2).to(config.DEVICE)
    )

    # val_dataset = Subset(val_dataset, list(range(32)))
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

    pred_boxes, true_boxes = get_evaluation_bboxes(
    val_loader,
    model,
    iou_threshold=config.NMS_IOU_THRESH,
    anchors=config.ANCHORS,
    threshold=config.CONF_THRESHOLD,
    )

    # If one wants to save precision recall curve, set visualize_PR=True
    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
        visualize_PR=False
    )
    print(f"MAP: {mapval.item()}")

    #### Visualize the model output bounding boxes and classes
    plot_couple_examples(model, val_loader, thresh=config.CONF_THRESHOLD, iou_thresh=config.NMS_IOU_THRESH, anchors=scaled_anchors)

if __name__ == "__main__":
    main()