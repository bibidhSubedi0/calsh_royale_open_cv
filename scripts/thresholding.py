import cv2 as cv
import numpy as np
from filter import HSV_FILTER

class vision:
    # const
    TRACK_BAR = "Trackbars"


    child_image = None
    child_w =0 
    child_h =0
    method =None

    def __init__(self,child_image,method=cv.TM_CCOEFF_NORMED):
        self.child_image = child_image
        self.child_w = self.child_image.shape[1]
        self.child_h = self.child_image.shape[0]
        

        self.method = method
        pass

    def findClickPos(self,parent_image,threshold=0.5,max_results = 4):
        main_img = parent_image
        to_check = self.child_image
        w = to_check.shape[1]
        h = to_check.shape[0]

        result  =cv.matchTemplate(main_img,to_check,self.method)

        locations = np.where(result>=threshold)
        loc = list(zip(*locations[::-1]))

        

        rects = []
        for locs in loc:
            rect = [int(locs[0]),int(locs[1]),w,h]
            rects.append(rect)
            rects.append(rect)
            # This is done twice because, as it happens if there is only 1 rectange
            # then it will be discarded by the groupRectangles Function

        rects, wx = cv.groupRectangles(rects,1,0.5)
        # print(rects)

        if len(rects) > max_results:
            print('Warning: too many results, raise the threshold.')
            rects = rects[:max_results]
        return rects


    def get_click_pos(self,rects):
        # Save the points
        points = []

        for (x,y,wi,hi) in rects:

            # Determining the center position
            cx = x+int(wi/2)
            cy =y+ int(hi/2)

            points.append((cx,cy))
        return points

    def show_rectanges(self,main_img,rects):
        lin_col = (0,255,0)
        lin_type = cv.LINE_4
        for(x,y,w,h) in rects:
            tl = (x,y)
            br = (x+w,y+h)
            cv.rectangle(main_img,tl,br,lin_col,lin_type)
        return main_img
            
    def show_crosshairs(self,main_img,points):
        marker_col = (255,0,0)
        marker_type = cv.MARKER_CROSS
        for(cx,cy) in points:
            cv.drawMarker(main_img,(cx,cy),marker_col,marker_type)
        return main_img

    def init_control_gui(self):
        cv.namedWindow(self.TRACK_BAR, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.TRACK_BAR, 350, 700)

        # required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass

        # create trackbars for bracketing.
        # OpenCV scale for HSV is H: 0-179, S: 0-255, V: 0-255
        cv.createTrackbar('HMin', self.TRACK_BAR, 0, 179, nothing)
        cv.createTrackbar('SMin', self.TRACK_BAR, 0, 255, nothing)
        cv.createTrackbar('VMin', self.TRACK_BAR, 0, 255, nothing)
        cv.createTrackbar('HMax', self.TRACK_BAR, 0, 179, nothing)
        cv.createTrackbar('SMax', self.TRACK_BAR, 0, 255, nothing)
        cv.createTrackbar('VMax', self.TRACK_BAR, 0, 255, nothing)
        # Set default value for Max HSV trackbars
        cv.setTrackbarPos('HMax', self.TRACK_BAR, 179)
        cv.setTrackbarPos('SMax', self.TRACK_BAR, 255)
        cv.setTrackbarPos('VMax', self.TRACK_BAR, 255)

        # trackbars for increasing/decreasing saturation and value
        cv.createTrackbar('SAdd', self.TRACK_BAR, 0, 255, nothing)
        cv.createTrackbar('SSub', self.TRACK_BAR, 0, 255, nothing)
        cv.createTrackbar('VAdd', self.TRACK_BAR, 0, 255, nothing)
        cv.createTrackbar('VSub', self.TRACK_BAR, 0, 255, nothing)

    def curret_hsv_filters(self):
        hsv_filter = HSV_FILTER()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', self.TRACK_BAR)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', self.TRACK_BAR)
        hsv_filter.vMin = cv.getTrackbarPos('VMin', self.TRACK_BAR)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', self.TRACK_BAR)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', self.TRACK_BAR)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', self.TRACK_BAR)
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', self.TRACK_BAR)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', self.TRACK_BAR)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', self.TRACK_BAR)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', self.TRACK_BAR)
        return hsv_filter
        
    def apply_hsv_filter(self, img, hsv_filter=None):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # if we haven't been given a defined filter, use the filter values from the GUI
        if not hsv_filter:
            hsv_filter = self.curret_hsv_filters()

        # add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, -hsv_filter.vSub)
        hsv = cv.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])
        # Apply the thresholds
        mask = cv.inRange(hsv, lower, upper)

        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # convert back to BGR for imshow() to display it properly
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img

    def shift_channel(self, c, amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c