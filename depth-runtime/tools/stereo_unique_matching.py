"""Explicit offline local-uniqueness gate after ROI-normalized stereo matching.

Both directions require the best ZNCC peak to lie within two pixels of the
refined point and exceed alternatives outside that radius by 0.05. Searches
retain 2-D freedom; no epipolar row, depth, tags or calibration are inputs.
This rejects repeated local patterns, not all physical correspondence biases.
"""
import cv2
import numpy as np
from stereo_photometric_matching import refine_matches as roi_refine


def unique_patch(source, target, first, second):
    for image, point in ((source, first), (target, second)):
        if (point[0]-20 < 0 or point[1]-20 < 0 or point[0]+20 > image.shape[1]-1 or
                point[1]+20 > image.shape[0]-1):
            return False
    template = cv2.getRectSubPix(source, (15, 15), tuple(map(float, first)))
    search = cv2.getRectSubPix(target, (41, 41), tuple(map(float, second)))
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(scores).all():
        return False
    y, x = np.indices(scores.shape); centered = (x-13)**2+(y-13)**2 <= 4
    best_center = float(np.max(scores[centered]))
    alternative = float(np.max(scores[~centered]))
    return best_center >= .8 and best_center-alternative >= .05


def refine_matches(images, masks, matches, return_indices=False):
    points, indices = roi_refine(images, masks, matches, return_indices=True)
    images = [np.asarray(image, dtype=np.float32) for image in images]
    keep = [i for i, p in enumerate(points) if unique_patch(images[0], images[1], p[:2], p[2:]) and
            unique_patch(images[1], images[0], p[2:], p[:2])]
    selected = points[keep]
    return (selected, indices[keep]) if return_indices else selected
