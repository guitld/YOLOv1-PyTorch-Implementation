import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def bounding_box(box):
    x1 = box[..., 0:1] - box[..., 2:3] / 2
    y1 = box[..., 1:2] - box[..., 3:4] / 2
    x2 = box[..., 0:1] + box[..., 2:3] / 2
    y2 = box[..., 1:2] + box[..., 3:4] / 2

    return x1, y1, x2, y2

def area(coords):
    return abs((coords[2] - coords[0]) * (coords[3] - coords[1]))


def iou(pred_box, true_box):
    pred_bounds = bounding_box(pred_box)
    true_bounds = bounding_box(true_box)

    # intersection retangle
    x1 = torch.max(pred_bounds[0], true_bounds[0])
    y1 = torch.max(pred_bounds[1], true_bounds[1])
    x2 = torch.min(pred_bounds[2], true_bounds[2])
    y2 = torch.min(pred_bounds[3], true_bounds[3])

    intersection_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    pred_area = area(pred_bounds)
    true_area = area(true_bounds)

    return intersection_area / (pred_area + true_area - intersection_area) 