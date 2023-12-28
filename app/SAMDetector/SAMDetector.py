"""
Object detector based on the use of SAM for product candidates detection and Mediapipe for template image matching using embeddings.
"""
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as  np
import cv2

class SAMDetector:
    def __init__(self, sam_model_dir, sam_model_type, device='cuda') -> None:
        self.initialize_model(sam_model_dir, sam_model_type, device)

    """
    This method initialze SAM.
    """
    def initialize_model(self, sam_model_dir, sam_model_type, device='cuda'):
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
            #print("rect: ", rect)
            #print("rrect: ", rotated_rect)
            area = rect[0]*rect[1]
            result = {'rotated_rect':rotated_rect, 'rect': rect, 'min_area': area}
            results.append(result)
        
        # sort rectangles by area
        results = sorted(results, key=lambda x: x['min_area'], reverse=True)

        return results
    
    """
    This function draws the detected objects in the input image
    """
    def draw_results(self, image, results):
        BB_COLOR_1 = (16,164,14)
        BB_COLOR_2 = (13,112,218)
        BB_FONT = cv2.FONT_HERSHEY_DUPLEX
        FONT_SCALE = 0.5
        print(results)
        for r in results:
            print(r)
            rect = r['rotated_rect']
            #score = r[2]
            #BB_COLOR = BB_COLOR_1 if score>0.5 else BB_COLOR_2
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            image = cv2.drawContours(image, [box], 0, BB_COLOR_1, 2)
            
            #image = cv2.putText(image, "{}".format(label), box[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 1)
        return image
