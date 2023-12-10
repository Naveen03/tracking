import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

def farnerback_from_pairimage(img1, img2):
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    #Create a mask in HSV Color Space.
    hsv = np.zeros_like(im1)
    # Sets image saturation to maximum.
    hsv[..., 1] = 255

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print("flow.shape : ", flow.shape)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print("angle.shape : ", angle.shape)

    # Sets image hue according to the optical flow direction
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow magnitude (normalized)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def farneback_from_video(image):
    global idx, im1
    if idx==0:
        im1 = image
        idx+=1
        return im1
    else:
        im2 = image
        gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(im1)
        hsv[..., 1] = 255
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        idx+=1
        im1= image.copy()
        return bgr
    

idx = 0
im1 = cv2.imread("C:/Users/navee/Documents/datsets/opticalflow1/image0.jpg")
im2 = cv2.imread("C:/Users/navee/Documents/datsets/opticalflow1/image1.jpg")
video_file = "C:/Users/navee/Documents/datsets/opticalflow1/hand_movement2.mp4"

flow_map = farnerback_from_pairimage(im1, im2)
cv2.imwrite("output/flow_farneback/outflow.png", flow_map)

clip = VideoFileClip(video_file).subclip(0,10)
white_clip = clip.fl_image(farneback_from_video)
white_clip.write_videofile("output/flow_farneback/output_hand.mp4",audio=False)