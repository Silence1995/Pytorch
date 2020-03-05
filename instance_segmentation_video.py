import torchvision
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as T
import random

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
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

def colour_masks(image,pred_cls):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
 
    r[image == 1], g[image == 1], b[image == 1] = list(CATEGORY_COLOR[pred_cls])
    coloured_mask = np.stack([r, g, b], axis=2)
    
    return coloured_mask

def get_prediction(img, threshold):
    
    img = Image.fromarray(img)#＃From numpy array to PIL image

    transform = T.Compose([T.ToTensor()])#the image is converted to image tensor using PyTorch’s Transforms
    img = transform(img)
    pred = model([img])#mage is passed through the model to get the predictions
    
    # prediction classes and bounding box coordinates are obtained from the model 
    # soft masks are made binary(0 or 1) ie: eg. segment of cat is made 1 and rest of the image is made 0

    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]

    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    return masks, pred_boxes, pred_class



def video_instance_segmentation_api(video_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    
  
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
        print('current frame/total frame=', current_frame,"/",total_frame, flush=True)
        ret, frame = cap.read()
        masks, boxes, pred_cls = get_prediction(frame, threshold)# Get predictions
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB

        for i in range(len(masks)):

            cv2.rectangle(img, boxes[i][0], boxes[i][1],color=CATEGORY_COLOR[pred_cls[i]], thickness=rect_th)        
            text_width, text_height = cv2.getTextSize(pred_cls[i],cv2.FONT_HERSHEY_DUPLEX, text_size,thickness=text_th)[0]
            cv2.rectangle(img,  boxes[i][0],  (int(boxes[i][0][0]+text_width), int(boxes[i][0][1]-text_height)), CATEGORY_COLOR[pred_cls[i]], cv2.FILLED)
            cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_DUPLEX, text_size, (255,255,255), thickness=text_th)


            rgb_mask = colour_masks(masks[i],pred_cls[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.9, 0)
    
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
    
    video_instance_segmentation_api('test.mp4', threshold=0.9, rect_th=5, text_size=1, text_th=3)

    print('Done')