"""
Object detector based on the use of SAM for product candidates detection and Mediapipe for template image matching using embeddings.
"""
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as  np
import cv2

class SAMDetector:
    def __init__(self, sam_model_dir, sam_model_type, device='cuda') -> None:
        
        # Initialize SAM
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_model_dir)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)


    """
    This function returns a list of dictionaries with the detected objects.
    It will return only the rotated rect and the bounding box.
    """
    def detect(self, image):
        # Get masks
        masks = self.mask_generator.generate(image)
        h,w,c = image.shape
        results = []
        for mask in masks:
            bin_mask = np.zeros(mask['segmentation'].shape, dtype=np.uint8)
            bin_mask[mask['segmentation']] = 255
            contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours)>1:
                max_area_rect = 0
                contour = contours[0]    
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > max_area_rect:
                        contour = c
            else:
                contour = contours[0]

            rotated_rect = cv2.minAreaRect(contour)
            rect = cv2.boundingRect(contour)
            area = rect[1][0]*rect[1][1]
            result = {'rotated_rect':rotated_rect, 'rect': rect, 'min_area': area}
            results.append(result)
        
        # sort rectangles by area
        results = sorted(result, key=lambda x: x['area'], reverse=True)

        return results
