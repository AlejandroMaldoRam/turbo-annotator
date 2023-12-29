# Script for detecting all the possible objects using SAM in all the images inside a folder. 
# It saves the objects in a JSON format

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
    
    # Configure dataset
    base_addr = "/home/amaldonado/Datasets/MB/DS1/*.jpg"
    files = glob(base_addr)

    # Process each image from the dataset
    all_detections = []
    offset = 250
    end = 390
    for i, f in enumerate(files[offset:], start=offset):
        if i == end:
            break
        print("{}/{}".format(i, len(files)))
        image = cv2.imread(f)
        detections = detector.detect(image)
        all_detections.append({'image_addr':f, 'detections':detections})
        
    
    output_addr = "detections.json"
    if os.path.exists(output_addr):
        f = open(output_addr, 'r')
        initial_list = json.load(f)
        f.close()
        #json_detections = json.dumps(initial_list+all_detections)
        f = open(output_addr, 'w')
        json.dump(initial_list+all_detections, f, indent=4)
        f.close()
    else:
        f = open(output_addr, 'w')
        #initial_list = json.load(f)
        #json_detections = json.dumps(initial_list+all_detections)
        json.dump(all_detections, f, indent=4)
        f.close()



