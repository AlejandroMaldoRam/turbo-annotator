"""
Object detector based on the use of SAM for product candidates detection and Mediapipe for template image matching using embeddings.
"""
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as  np
import math
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
            area = rect[2]*rect[3]
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
        #print(results)
        for r in results:
            #print(r)
            rect = r['rotated_rect']
            #score = r[2]
            #BB_COLOR = BB_COLOR_1 if score>0.5 else BB_COLOR_2
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            image = cv2.drawContours(image, [box], 0, BB_COLOR_1, 2)
            
            #image = cv2.putText(image, "{}".format(label), box[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 1)
        return image
    
    """
    This function extracts the image of a given detection. 
    """
    def extract_object(self, image, detection):
        box = cv2.boxPoints(detection['rotated_rect'])
        box = np.intp(box)
        warped = self.four_point_transform(image, box)
        return warped

    def distance(self, pt1, pt2):
        return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)


    def order_points(self, pts):
        """
        Return a set of 4 points clockwise-ordered starting from the top-left point.
        """
        
        rect = np.zeros((4, 2), dtype="float32")
        
        # the top-left point will have the smallest sum, whereas
        # the bottom.right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # now compute the difference between the two points,
        # the top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # return the ordered coordinates
        return rect

    """
    This funtion get the warped image given a rotated rectangle
    """
    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)

        #print("four rect: ", rect)
        dst_width = int(self.distance(rect[0],rect[1]))
        dst_height = int(self.distance(rect[1],rect[2]))
        
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left order
        dst = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]], dtype="float32")
        
        #print("dst: ", dst)
        
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (dst_width, dst_height))
        
        return warped


