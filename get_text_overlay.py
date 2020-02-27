"""
Appending solution on standard template
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


#IS algorithm
def r_c(hist ):
    ''' The method returns the best threshold value for segregation of text region from background.'''
    Ng = 255
    tc = 255
    index = 0

    while index <= tc and index <= Ng:
        bck_sum = 0
        hist_sum_bck = 0
        fr_sum = 0
        hist_sum_fr = 0
        b = 0 
        f = 0
               
        for i in range(0, index+1):
            bck_sum += i * hist[i][0]
            hist_sum_bck += hist[i][0]
            
        for j in range(index+1, Ng+1):
            fr_sum += j * hist[j][0]
            hist_sum_fr += hist[j][0]

        try:
            b = (bck_sum / hist_sum_bck)   
            f = (fr_sum / hist_sum_fr)
        
            if hist_sum_bck == 0.0 : 
                raise ZeroDivisionError  
            if hist_sum_fr == 0.0:
                raise ZeroDivisionError 
        except ZeroDivisionError:
            index += 1
            continue

        tc = (b  + f) // 2 

        index += 1
        
    return tc



def getTextOverlay(input_image):
    # output = np.zeros(input_image.shape, dtype=np.uint8)

    # Convert to gray (for histogram calculation)
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # histogram calculation
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # optimum threshold value based on Iterative selection algorithm
    thresh_val = int(r_c(hist))
    print(f'threshold value {thresh_val}')
    # threshold    
    _, bin_img = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
    
    # morphological opening to remove minor errors
    kernel = np.ones((7,7), np.uint8)
    enhance_bin = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    return enhance_bin

if __name__ == '__main__':
    image = cv2.imread('simpsons_frame0.png')
    output = getTextOverlay(image)
    # cv2.imwrite('simpons_text.png', output)
    

    org_minm = cv2.resize(image, None, fx=0.6, fy=0.6)
    cv2.imshow("original", org_minm)

    out_minn = cv2.resize(output, None, fx=0.6, fy=0.6)
    cv2.imshow("result", out_minn)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


