import cv2 as cv
import numpy as np


main_img = cv.imread('crproj\cr1.png', cv.IMREAD_UNCHANGED)
to_check = cv.imread('crproj\sparky.png', cv.IMREAD_UNCHANGED)

result  =cv.matchTemplate(main_img,to_check,cv.TM_CCOEFF_NORMED)

# Get the best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print("Best match top left : %s" % str(max_loc))
print("Best match confidence : %s" % max_val)

threshold = 0.8

if max_val >= threshold:
    print("Found")

    w = to_check.shape[1]
    h=to_check.shape[0]

    top_left = max_loc
    botton_right = (top_left[0]+w,top_left[1]+h)

    cv.rectangle(main_img,top_left,botton_right,color=(0,255,0),thickness=2,lineType=cv.LINE_4)
    cv.imshow('Result',main_img)
    # Can also be save with
    #cv.imwrite('result.jpg',main_img)
    cv.waitKey()
else:
    print("Not Found")


