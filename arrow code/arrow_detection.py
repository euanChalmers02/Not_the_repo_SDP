import glob, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision

def arrow_direction(dir_vec):
    
    # Calculate angle with pos x axis
    pos_x = np.asarray([1,0]) # Direction of Positive x axis
    dot_pr = dir_vec.dot(pos_x) 
    norms = np.linalg.norm(pos_x) * np.linalg.norm(dir_vec) # 
 
    angle = np.rad2deg(np.arccos(dot_pr / norms))

    if angle < 45:
        return 0
    elif angle >= 45 and angle < 135:
        if dir_vec[1] < 0:
            return 1
        else: 
            return 3
    else:
        return 2

# returns direction code: Right: 0, Up: 1, Left: 2, Down: 3
# feature_plot_radius: larger radius reduces the weight of close features (overlap)
def arrow_heading_detection(img, feature_plot_radius = 3):

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise suppression using gaussian filter.
    gaussian = cv2.GaussianBlur(gray,(45,45),0)

    # Extract arrow edges using gaussian.
    edges = cv2.Canny(gaussian, 200, 255, apertureSize=5)

    # Distance transforms (dt) for feature extraction.
    out = cv2.distanceTransform(255- edges, cv2.DIST_L2, 3)

    shape = out.shape
    
    # Plot features extracted from dt on a black screen.
    black = np.zeros(shape)

    # Extract 25 features on distance transforms.
    corners = cv2.goodFeaturesToTrack(out, 25,0.1,1)
    
    # Plot feature points as circles on black screen.
    for corner in corners:
        features = cv2.circle(black, (int(corner[0][0]), int(corner[0][1])), 3, (255,255,255), feature_plot_radius)

    # Estimate moments for the black screen 
    M = cv2.moments(features)
    area = M['m00']
    # Calculate centroid using moments
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # Center of mass of 25 feature points.
    feature_centroid = np.asarray([cx, cy])
    # Geometric center of the image.
    img_centroid = np.asarray([shape[1] / 2, shape[0] / 2])
    # Vector represents the field direction of feature points.
    direction_vector = feature_centroid - img_centroid
    
    direction_code = arrow_direction(direction_vector)
    
    return direction_code