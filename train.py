"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from torch.optim.lr_scheduler import MultiStepLR

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    load_pretrained_weights,
    get_loaders,
    seed_everything
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

seed_everything()

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    Function to train the model for one epoch
    Returns the mean loss for the epoch
    """
    model.train()
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

        # Each iteration can be logged to MLflow
        # mlflow.log_metric("train_iteration_loss", mean_loss, step=(epoch+1)*batch_idx)
    return mean_loss
    
    
def evaluate_fn(val_loader, model, loss_fn, scaled_anchors, threshold):
    """
    Function to evaluate the model on the validation set
    Additionally, calculates the class accuracy, object accuracy and no object accuracy
    """
    model.eval()
    loop = tqdm(val_loader, leave=True)
    losses = []

    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.no_grad():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)
    class_acc = (correct_class / (tot_class_preds + 1e-16))*100
    no_obj_acc = (correct_noobj / (tot_noobj + 1e-16))*100
    obj_acc = (correct_obj / (tot_obj + 1e-16))*100

    print(f"Class accuracy is: {class_acc:2f}%")
    print(f"No obj accuracy is: {no_obj_acc:2f}%")
    print(f"Obj accuracy is: {obj_acc:2f}%")
    model.train()
        
    return class_acc, no_obj_acc, obj_acc, mean_loss
     

def main():

    # Start MLflow run
    mlflow.start_run(run_name=config.RUN_NAME)

    # Create the model and sent it to the device
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Load pretrained weights and initialize the model
    load_pretrained_weights(model, config.BASE_FILE, config.NUM_CLASSES)

    # Create the optimizer and scheduler, concentrate only on the parameters that require gradients
    optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=config.LEARNING_RATE, 
    weight_decay=config.WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=[10, 25, 35, 45, 55], gamma=0.7) #

    # Define the loss function and the scaler for mixed precision training
    loss_fn = YoloLoss()
    scaler = torch.amp.GradScaler('cuda')

    # Load the train and validation data
    train_loader, val_loader = get_loaders(subset=False)

    # If it is a resuming training, load the checkpoint
    if config.RESUME_TRAINING:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    # Calculate scaled anchors
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # Define a loss variable to keep track of the least loss, update it when the loss decreases
    loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n Currently epoch {epoch}")

        mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        mlflow.log_metric("train_loss", mean_loss, step=epoch)

        scheduler.step()
        # Log learning rate to MLflow
        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)

        if epoch % 2 == 0:
            print("On Validation evaluation:")
            class_acc, no_obj_acc, obj_acc, eval_loss = evaluate_fn(
                val_loader, model, loss_fn, scaled_anchors, threshold=config.CONF_THRESHOLD
                )
            mlflow.log_metric("val_loss", eval_loss, step=epoch)
            mlflow.log_metric("class_accuracy", class_acc, step=epoch)
            mlflow.log_metric("no_obj_accuracy", no_obj_acc, step=epoch)
            mlflow.log_metric("obj_accuracy", obj_acc, step=epoch)

        if config.SAVE_MODEL and eval_loss < loss:
           print(f"Loss decreased {loss} -> {eval_loss}")
           save_checkpoint(model, optimizer, filename=f"least_loss_checkpoint.pth.tar")
        #    mlflow.pytorch.log_model(model, "least_loss_model") # One other way is to save the model to mlflow
           loss = eval_loss
        
        if config.SAVE_MODEL and epoch > 0 and epoch % 10 == 0:
           save_checkpoint(model, optimizer, filename=f"epoch{epoch}_checkpoint.pth.tar")
        #    mlflow.pytorch.log_model(model, f"epoch{epoch}_model") # One other way is to save the model to mlflow

        if epoch > 10 and epoch % 4 == 0:
            print("On Validation mAP calculation:")
            
            pred_boxes, true_boxes = get_evaluation_bboxes(
                val_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            mlflow.log_metric("mAP 0.5", mapval.item(), step=epoch)
            model.train()


if __name__ == "__main__":
    main()