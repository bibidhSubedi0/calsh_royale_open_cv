import numpy as np
import win32gui, win32ui, win32con
import pyautogui
import numpy as np
import cv2


class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name):
        # find the handle for the window we want to capture
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y
        
    def get_screenshot(self):
        # Take a screenshot of the region defined by your window coordinates (self.cropped_x, self.cropped_y, self.w, self.h)
        screenshot = pyautogui.screenshot(region=(self.cropped_x, self.cropped_y, self.w, self.h))
        
        # Convert to a numpy array and handle the color format (RGB to BGR for OpenCV compatibility)
        img = np.array(screenshot)
        
        # Convert RGB to BGR (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img
    # def get_screenshot(self):

    #     # get the window image data
    #     wDC = win32gui.GetWindowDC(self.hwnd)
    #     dcObj = win32ui.CreateDCFromHandle(wDC)
    #     cDC = dcObj.CreateCompatibleDC()
    #     dataBitMap = win32ui.CreateBitmap()
    #     dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
    #     cDC.SelectObject(dataBitMap)
    #     cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

    #     # convert the raw data into a format opencv can read
    #     #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
    #     signedIntsArray = dataBitMap.GetBitmapBits(True)
    #     img = np.fromstring(signedIntsArray, dtype='uint8')
    #     img.shape = (self.h, self.w, 4)

    #     # free resources
    #     dcObj.DeleteDC()
    #     cDC.DeleteDC()
    #     win32gui.ReleaseDC(self.hwnd, wDC)
    #     win32gui.DeleteObject(dataBitMap.GetHandle())

    #     img = img[...,:3]

    #     img = np.ascontiguousarray(img)

    #     return img

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    def list_window_names(self):
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)