import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from thresholding import vision
from filter import HSV_FILTER

# Initialize WindowCapture class (example)
wincap = WindowCapture('Clash Royale')


c = cv.imread('crproj\\images\\opps.png', cv.IMREAD_UNCHANGED)
if c.shape[2] == 4:  # If the template has an alpha channel
    c = cv.cvtColor(c, cv.COLOR_BGRA2BGR)  # Convert to BGR (3 channels)
if c.dtype != np.uint8:
    c = c.astype(np.uint8)  # Convert to 8-bit unsigned if necessary


vis_inst = vision(c)

loop_time = time()

vis_inst.init_control_gui()

# HSF Filter for that troop i guess?
hsv_filter = HSV_FILTER(165,0,0,179,255,255,0,0,0,0)


while True:
    screenshot = wincap.get_screenshot()
    # cv.imwrite(temp_filename, screenshot)

    # p = cv.imread(temp_filename, cv.IMREAD_UNCHANGED)
    



    #-------------------PREPROCESSING THE SCREENSHOT?-----------------------
    p =screenshot
    if p.shape[2] == 4:  # If the image has an alpha channel
        p = cv.cvtColor(p, cv.COLOR_BGRA2BGR)  # Convert to BGR (3 channels)

    if p.dtype != np.uint8:
        p = p.astype(np.uint8)  # Convert to 8-bit unsigned if necessary


    processed_output_image = vis_inst.apply_hsv_filter(p,hsv_filter)







    # Find matching positions (you should implement or define findClickPos)
    rects = vis_inst.findClickPos(processed_output_image,0.5,10)

    output_image = vis_inst.show_rectanges(p,rects)

    # Display Images
    cv.imshow("matches",output_image)
    #cv.imshow("processed",processed_output_image)

    # Debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    #

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break


# Preprocessing
'''
    -> Remove evey other colors then needed
        Hue is like a color wheel bettwn Hmax and Hmin, evey color is accomodated
        Saturation is well saturaion
        V is the value like say from balckiest balck to whitest white
    -> Oversatura?
'''