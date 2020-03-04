# Pytorch

### Image Classification

Using VGG & Resnet in PyTorch <b>torchvision.models</b>, and train CIFAR10 dataset for image classification


- image_classification.ipynb
    - the notebook is using GPU on <b>Google colab</b>

- Learning rate adjustment by <b>StepLR</b> from <b>torch.optim.lr_scheduler</b>
    - optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    - scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        - `lr=0.1` for epoch `[0,step_size=50)`
        - `lr=0.01` for epoch `[50,100)`
        - `lr=0.001` for epoch `[100,150)`

 
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
|   <img src="https://github.com/Silence1995/Pytorch/blob/master/figure/object_detection.JPG" width="200" height="300" />    | [<img src="https://github.com/Silence1995/Pytorch/blob/master/figure/object_detection_video.JPG"  width="200" height="300" >](https://drive.google.com/file/d/1KNA_cTJh8C-tvww7oN8UsnDjCUyHWxsY/view?usp=sharing)|


### Reference
- [Faster R-CNN](https://zhuanlan.zhihu.com/p/93829453)
- [TRAINING A CLASSIFIER](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
