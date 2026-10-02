"""Compose fixed resize, blur, latitude and tile averaging operators.

This is the same linear map in real arithmetic, evaluated without allocating
the 4096x2048 intermediate image. Floating point order changes; exact class
decisions must therefore be checked against reference_scores on actual runs.
"""
from __future__ import annotations
import numpy as np
from core import area_weights, tile_edges


def projection_matrices(width=4096, height=2048, low_width=192, low_height=144):
    import cv2
    xe, ye = tile_edges(width, height)
    # Each basis image measures an axis of OpenCV's actual interpolation and
    # border rule. GaussianBlur's 5x5 sigma=0 kernel is separable.
    bx = cv2.resize(np.eye(low_width, dtype=np.float32), (width, low_width),
                    interpolation=cv2.INTER_LINEAR)
    bx = cv2.GaussianBlur(bx, (5, 1), 0, borderType=cv2.BORDER_DEFAULT)
    by = cv2.resize(np.eye(low_height, dtype=np.float32), (low_height, height),
                    interpolation=cv2.INTER_LINEAR)
    by = cv2.GaussianBlur(by, (1, 5), 0, borderType=cv2.BORDER_DEFAULT)
    weights = area_weights(height).astype(np.float32)
    left = np.stack([(by[a:b].astype(np.float64)*weights[a:b,None]).mean(axis=0)
                     for a,b in zip(ye[:-1],ye[1:])])
    right = np.stack([bx[:,a:b].astype(np.float64).mean(axis=1)
                      for a,b in zip(xe[:-1],xe[1:])],axis=1)
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def reference_scores(normalized_map, width=4096, height=2048):
    """The previous full image implementation, retained as the oracle."""
    import cv2
    sal = cv2.resize(np.asarray(normalized_map,np.float32), (width,height),
                     interpolation=cv2.INTER_LINEAR)
    sal = cv2.GaussianBlur(sal, (5,5), 0)
    sal *= area_weights(height).astype(np.float32)[:,None]
    sums = cv2.integral(sal,sdepth=cv2.CV_64F)
    xe,ye=tile_edges(width,height)
    scores=np.empty((5,9),np.float64)
    for r,(a,b) in enumerate(zip(ye[:-1],ye[1:])):
        for c,(x,y) in enumerate(zip(xe[:-1],xe[1:])):
            scores[r,c]=(sums[b,y]-sums[a,y]-sums[b,x]+sums[a,x])/((b-a)*(y-x))
    return scores


def compact_scores(normalized_map, matrices):
    left,right=matrices
    return (left @ np.asarray(normalized_map,np.float64)) @ right
