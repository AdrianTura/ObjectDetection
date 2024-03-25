import cv2
import numpy as np
import onnx
import onnxruntime as onnxrt
import torch
from vision.utils import box_utils
from vision.ssd.config import mobilenetv1_ssd_config as config

import torch.nn.functional as F


# Read image and resize
image = cv2.imread('giri.jpg')
resized_image = cv2.resize(image, (300, 300))

resized_image_ = np.divide(resized_image, 255)
input_image = np.expand_dims(resized_image_.transpose(2, 0, 1), 0)

print('IMAGE SUM', np.sum(input_image))
model = onnx.load("models/mb1-ssd/14.onnx")
onnx.helper.printable_graph(model.graph)
# Run inference
onnx_session= onnxrt.InferenceSession("models/mb1-ssd/14.onnx")

onnx_inputs= {onnx_session.get_inputs()[0].name:input_image.astype(np.float32)}
onnx_output = onnx_session.run(None, onnx_inputs)

scores = torch.Tensor(onnx_output[0])
locations = torch.Tensor(onnx_output[1])

print('Initial: ', torch.sum(locations))

# Process output

scores= F.softmax(scores, dim=2)
boxes = box_utils.convert_locations_to_boxes(
                locations, config.priors, config.center_variance, config.size_variance)

print('After convert locations to boxes:',torch.sum(boxes))
boxes = box_utils.center_form_to_corner_form(boxes)

print('After converting to corner form:',torch.sum(boxes))

boxes = boxes[0]
scores = scores[0]

picked_box_probs = []
picked_labels = []

prob_threshold = 0.4

for class_index in range(1, scores.size(1)):
    probs = scores[:, class_index]
    mask = probs > prob_threshold
    print(class_index)
    probs = probs[mask]
    if probs.size(0) == 0:
        continue
    subset_boxes = boxes[mask, :]
    box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
    box_probs = box_utils.nms(box_probs, None,
                                score_threshold=prob_threshold,
                                iou_threshold=0.35,
                                sigma=0.5,
                                top_k=10,
                                candidate_size=200)
    picked_box_probs.append(box_probs)
    picked_labels.extend([class_index] * box_probs.size(0))
    
if not picked_box_probs:
    print('No faces detected!')
    import sys
    sys.exit()

picked_box_probs = torch.cat(picked_box_probs)
picked_box_probs[:, 0] *= 300
picked_box_probs[:, 1] *= 300
picked_box_probs[:, 2] *= 300
picked_box_probs[:, 3] *= 300

out_boxes, out_scores = picked_box_probs[:, :4], picked_box_probs[:, 4]

print('After all the stuff:', torch.sum(out_boxes))

no_faces = out_boxes.size(0)
print('Number of faces: ', no_faces)

for i in range(out_boxes.size(0)):
    box = out_boxes[i, :]
    box = [int(x) for x in box]

    if not(box[0] < 0 or box[1] < 0 or box[2] > 300 or box[3] > 300):
        cv2.rectangle(resized_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0))
    print('Face: [',box[0], box[1], box[2], box[3], ']')

# Save the frame to an image file.
cv2.imwrite('out.png', resized_image)

print('Finished!')