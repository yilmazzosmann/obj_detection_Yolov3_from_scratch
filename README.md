# Object Detection From Scratch with Yolov3
This is a Python implementation of Yolov3 finetuning on a small-sized COCO dataset. The main goal was to reproduce the existing network, fine-tune it, and comment on the outcome. Please see the [WIKI](https://github.com/yilmazzosmann/obj_detection_Yolov3_from_scratch/wiki/Project-Details-and-Design-Decisions) page for details.

Start with creating the environment 

`conda create --name yolov3 python==3.10`

`conda activate yolov3`

`pip install -r requirements.txt `

**Step 1**

Go to the [COCO dataset download](https://cocodataset.org/#download) page and download "2017 Train/Val annotations [241MB]". Extract the zip file and move the annotations folder inside the working directory.

**Step 2**

`python download_dataset.py ` 

`python split_dataset.py `

Now, we have images, training, and validation set with annotations ready. If you want more classes or more images, check _download_dataset.py_ script.

**Step 3**

If you want to do finetuning on the Yolov3 model trained with the Pascal VOC dataset, download it from [HERE](https://www.kaggle.com/datasets/1cf520aba05e023f2f80099ef497a8f3668516c39e6f673531e3e47407c46694). It is recommended to do fine-tuning.

**Step 4**

Check the config.py file with your preferences, give your run a name, and start training.

`python train.py `

**Step 5**

Check training and validation variables by running 
`mlflow ui` in the terminal.

Check a couple of examples outputs with 

`python evaluate.py`

If you want to save precision-recall curves of each class, go to _mean_average_precision_ function and change _visualize_PR__ to True

**If you want to play with the model I fine-tuned, please download it from the [LINK](https://www.kaggle.com/models/yilmazzosmann/custom-yolov3)**

## Training and Validation Metrics saved with MLflow
![finetuning-yolov3](https://github.com/user-attachments/assets/6064d9ab-f450-4fa1-aaa8-d33c40088b56)

## Precision-Recall Curve Plot of 3 classes
![pr-curve-all](https://github.com/user-attachments/assets/4eb1be5d-c170-4a1c-93a8-84a786ae2fa9)

## Example Outputs from Validation Set
![Figure_1](https://github.com/user-attachments/assets/a18c2d11-4f37-482d-822d-ba370511df4c)

![Figure_2](https://github.com/user-attachments/assets/89206ba7-9bb6-4a35-8337-c465afaf6a97)

![Figure_4](https://github.com/user-attachments/assets/595fb5a4-2d66-4a00-a36c-294f33bf4641)

![Figure_5](https://github.com/user-attachments/assets/53f37e16-e6a3-4940-9b53-f3185f87b048)
