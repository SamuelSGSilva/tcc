import pytest
import numpy as np
import cv2
from src.estimation import WidthEstimator

def test_estimator_empty_mask():
    estimator = WidthEstimator()
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    median_w, min_w, max_w, std_w, num_pixels, skel_mask = estimator.estimate(mask)
    
    assert median_w == 0.0
    assert min_w == 0.0
    assert max_w == 0.0
    assert std_w == 0.0
    assert num_pixels == 0
    assert cv2.countNonZero(skel_mask) == 0

def test_estimator_simple_straight_line():
    estimator = WidthEstimator()
    # Create a simple mask: horizontal line thickness 3
    mask = np.zeros((20, 50), dtype=np.uint8)
    mask[10:13, 10:40] = 255
    
    median_w, min_w, max_w, std_w, num_pixels, skel_mask = estimator.estimate(mask)
    
    # Thickness is 3 pixels. The distance transform from center should be ~1.
    # The actual algorithm: distance in center is 1, so width is 2. (Skeleton is 1px wide)
    # Another distance metric or sub-pixel thickness: dist could be 1.5, making it 3.
    # Depending on cv2.DIST_L2, a straight line of thickness 3:
    # 0 0 0
    # 1 1 1 - dist=1
    # 1 1 1 - dist=2? no, boundary is 1px away (since it's size 3)
    # The skeleton will be the center line, where dist_transform approx 1-1.5, so width 2-3px.
    
    assert num_pixels == 3 * 30
    assert median_w > 0
    assert cv2.countNonZero(skel_mask) > 0
    # Reasonable ranges for a 3-px line thickness estimation
    assert 2.0 <= median_w <= 4.0
