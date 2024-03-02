import cv2
import torch
import random
import logging
import numpy as np
import torch.nn.functional as F

def nms(hmap, kernel=21):
    hmax = F.max_pool3d(hmap, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == hmap).float()
    return hmap * keep

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian3D(shape, sigma=1):
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-m:m + 1, -n:n + 1, -p:p + 1]

    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_umich_gaussian_3D(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian3D((diameter, diameter, diameter), sigma=diameter / 6)

    z, y, x = int(center[0]), int(center[1]), int(center[2])

    depth, height, width = heatmap.shape[0:3]

    front, back = min(z, radius), min(depth - z, radius + 1)
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[z - front:z+back, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - front:radius + back, radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

