import torchvision
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as T
import random

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

CATEGORY_COLOR = dict()
COLORS = []

def  generate_color():

    number_of_category=len(COCO_INSTANCE_CATEGORY_NAMES )
    for i in range(number_of_category):
        r = random.randint(60,255)
        g = random.randint(70,255)
        b = random.randint(80,255)
        COLORS.append((r,g,b)) 


def get_prediction(frame, threshold):

    img = Image.fromarray(frame)
    
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    return pred_boxes, pred_class


def video_object_detection_api(video_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('fps = ' + str(fps))
    print('Total number of frames = ' + str(total_frame))
  
    
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))



    # Creat VideoWriter , Output to  output.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))
    
    while(cap.isOpened()):
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print('current frame:', current_frame, flush=True)
        ret, frame = cap.read()
        boxes, pred_cls = get_prediction(frame, threshold) # Get predictions
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB

        for i in range(len(boxes)):
            cv2.rectangle(frame, boxes[i][0], boxes[i][1],color=CATEGORY_COLOR[pred_cls[i]], thickness=rect_th) # Draw Rectangle with the coordinates
            cv2.putText(frame,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_DUPLEX, text_size, (255,255,255),thickness=text_th) # Write the prediction class
        
        out.write(frame)

        # cv2.namedWindow('Faster RCNN',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Faster RCNN', 300,600)
        # cv2.imshow('Faster RCNN',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or total_frame == current_frame:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    generate_color()

    CATEGORY_COLOR = dict(zip(COCO_INSTANCE_CATEGORY_NAMES, COLORS))
    
    video_object_detection_api('test.mp4', threshold=0.9, rect_th=5, text_size=1, text_th=3)

    print('Done')