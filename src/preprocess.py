import cv2
import numpy as np

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    return edges

def region_of_interest(edges):
    # Create a mask
    mask = np.zeros_like(edges)
    height, width = edges.shape
    polygon = np.array([[(200, height), (1100, height), (550, 250)]])
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    return masked_edges
