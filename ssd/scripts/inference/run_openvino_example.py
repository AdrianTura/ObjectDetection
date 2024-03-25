import argparse
import cv2 as cv
import numpy as np
from sqlalchemy import union
import torch
from vision.utils import box_utils
from vision.ssd.config import mobilenetv1_ssd_config as config

import torch.nn.functional as F

def iou(box1, box2):
    # Get coordinates
    x_top1, y_top1, x_bot1, y_bot1 = box1
    x_top2, y_top2, x_bot2, y_bot2 = box2
    
    # Compute area of each bbox
    box1_area = (x_bot1 - x_top1) * (y_bot1 - y_top1)
    box2_area = (x_bot2 - x_top2) * (y_bot2 - x_top2)

    # Compute area of intersection
    x_top = max(x_top1, x_top2)
    y_top = max(y_top1, y_top2)

    x_bot = max(x_bot1, x_bot2)
    y_bot = max(y_bot1, y_bot2)

    inter_area = (x_bot - x_top) * (y_bot - y_top)

    # Compute area of union
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou


def nms(boxes, scores, iou_treshold):
    output = {}

    output['boxes'] = []
    output['scores'] = []

    # Get the indexes of the sorted scores in descending order
    sorted_indexes = np.argsort(scores)[::-1]
    
    while len(sorted_indexes) > 0:
        # Append to output the first element (it always has the highest score)
        index = sorted_indexes[0]

        current_box, current_score = boxes[index], scores[index]

        output['boxes'].append(current_box)
        output['scores'].append(current_score)

        # Remove item from array
        sorted_indexes = np.delete(sorted_indexes, 0)
        print(sorted_indexes)

        # Remove other items with IOU > treshold
        other = 0
        while other < len(sorted_indexes):
            if(iou(current_box, boxes[other]) > iou_treshold):
                print(sorted_indexes.shape)
                sorted_indexes = np.delete(sorted_indexes, other)
                other = other - 1
            other = other + 1

    
    return output

def process_output(boxes, scores, prob_treshold, iou_treshold):
    boxes = boxes[0]
    scores = scores[0]

    out_boxes = []
    out_scores = []
    
    face_index = 1
    for i in range(0, len(scores)):
        if scores[i][face_index] > prob_treshold:
            out_boxes.append(boxes[i])
            out_scores.append(scores[i][face_index])

    print('calling nms')
    output = nms(out_boxes, out_scores, iou_treshold)
    print('done')

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)

    args = parser.parse_args()

    # Load the model.
    net = cv.dnn.readNetFromModelOptimizer(
                args.model_path + '.xml',
                args.model_path + '.bin')

    # Read an image.
    frame = cv.imread('giri.jpg')
    orig_frame = cv.resize(frame, (300, 300))

    frame = np.divide(orig_frame, 255)
    frame = frame.astype(np.float32)
    frame = cv.normalize(frame, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    #frame = np.expand_dims(frame.transpose(2, 0, 1), 0)

    # Prepare input blob and perform an inference.
    blob = cv.dnn.blobFromImage(frame, size=(300, 300), ddepth = cv.CV_32F)

    net.setInput(blob)

    boxes, scores = net.forward(net.getUnconnectedOutLayersNames())
    
    print('calling process output!')
    output = process_output(boxes, scores, prob_treshold=0.5, iou_treshold=0.3)
    print('done!')
    for box in output['boxes']:
        box[0] *= 300
        box[1] *= 300
        box[2] *= 300
        box[3] *= 300
        cv.rectangle(orig_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

    # Save the frame to an image file.
    cv.imwrite('out.png', orig_frame)

    print('Finished!')
