# Pytorch

### Image Classification

Using VGG & Resnet in PyTorch <b>torchvision.models</b>, and train CIFAR10 dataset for image classification


- image_classification.ipynb
    - the notebook is using GPU on <b>Google colab</b>

- Learning rate adjustment by <b>StepLR</b> from <b>torch.optim.lr_scheduler</b>


 
- Result
    - Accuracy

        |   Model  | Accuracy |
        |:--------:|:--------:|
        |   [VGG16](https://arxiv.org/abs/1409.1556)  |  85 %       |
        | [ResNet18](https://arxiv.org/abs/1512.03385)  |    80 %      |
    - Accuracy for each class
        |       | VGG16 | Resnet18 |
        |:-----:|-------|----------|
        | plane |  82 % | 86 %     |
        |  car  | 100 % | 85 %    |
        |  bird |    90 %   | 75 %  |
        |  cat  |    58 %   | 58 %  |
        |  deer |    85 %    | 88 %  |
        |  dog  |   84 %    | 72 %  |
        |  frog |   86 %    | 83 %  |
        | horse |    76 %   | 92 %  |
        | ship  |     96 %   | 96 %  |
        | truck  |    89 %  | 87 %  |

### Object Detection

Using Pre-trained Faster RCNN in PyTorch <b>torchvision.models.detection</b> for object detection

- object_detection_image.ipynb
    - the notebook is for image and the workflow for the detection 
- object_detection_video.py
    - object detection for the video
 
| Image | Video |
|:-----:|:-----:|
|   <img src="https://github.com/Silence1995/Pytorch/blob/master/figure/object_detection.JPG" width="200" height="300" />    | [<img src="https://github.com/Silence1995/Pytorch/blob/master/figure/object_detection_video.JPG"  width="200" height="300" >](https://drive.google.com/open?id=1_jY_eLx5o5wkypJac8GXYjyraJVIdFd_)|

### Instance Segmentation

Using Pre-trained Mask RCNN in PyTorch <b>torchvision.models.detection</b> for instance segmentation
- instance_segmentation_image.ipynb
    - the notebook is for image and the workflow for the instance_segmentation
- instance_segmentation_video.py
    - instance_segmentation for the video

| Image | Video |
|:-----:|:-----:|
|  <img src="https://github.com/Silence1995/Pytorch/blob/master/figure/instance_segmentation.JPG" width="200" height="300" /> |[<img src="https://github.com/Silence1995/Pytorch/blob/master/figure/instance_segmentation_video.JPG"  width="200" height="300" >](https://drive.google.com/open?id=1SeIvhCkkf8fnAm5DcgPfDfwGRFYzOOMz)       |

### Reference
- [Faster R-CNN Tutorial](https://zhuanlan.zhihu.com/p/93829453)
- [Classfier Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Instance-Segmentation Tutorial](https://www.learnopencv.com/mask-r-cnn-instance-segmentation-with-pytorch/)
