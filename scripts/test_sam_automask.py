# Script for testing SAM with prompting

import pandas as pd
from glob import glob
import numpy as np
import datetime
import time
import cv2
import json
from pprint import pprint
import os

from SAMDetector import SAMDetector

if __name__ == '__main__':
    # Configure detector
    device = 'cuda'
    detector = SAMDetector.SAMDetector("/home/amaldonado/Code/sam-test/models/sam_vit_l_0b3195.pth","vit_l")
    
    # Read image
    img_addr = "/home/amaldonado/Datasets/MB/DS1/img_0.jpg"

    # Process image with automatica mask
    image = cv2.imread(img_addr)
    t0 = time.time()
    detections = detector.detect(image)
    print("Detection time: ", time.time()-t0)
    
    # draw detections
    results_image = detector.draw_results(image, detections)
    cv2.imshow("Results", results_image)
    cv2.waitKey(0)


