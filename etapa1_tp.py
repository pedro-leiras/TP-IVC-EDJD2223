import cv2
import numpy as np

cap = cv2.VideoCapture()
image_inverted = 0
window_camera_name = "Camera"
hsv_color = np.array([])

def start(game):
    cv2.namedWindow(window_camera_name)
    camera(game)

def getRGB(event, x, y,flags,param):
    global hsv_color
    if event == cv2.EVENT_LBUTTONDOWN:
        colorsB = image_inverted[y, x, 0]
        colorsG = image_inverted[y, x, 1]
        colorsR = image_inverted[y, x, 2]
        colors = image_inverted[y, x]
        hsv_value = np.uint8([[[colorsB, colorsG, colorsR]]])
        hsv_color = cv2.cvtColor(hsv_value, cv2.COLOR_BGR2HSV)
        print("HSV : ", hsv_color)
        print("Color: ", colors)


def camera(game):
    global image_inverted
    global hsv_color

    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    image_inverted = image[:, ::-1, :]
    cv2.imshow(window_camera_name, image_inverted)

    if hsv_color.size == 0:
        cv2.setMouseCallback(window_camera_name, getRGB)
    else:
        cv2.setMouseCallback(window_camera_name, lambda *args : None)
        image_hsv = cv2.cvtColor(image_inverted, cv2.COLOR_BGR2HSV)
        h = hsv_color[:, :, 0]
        s = hsv_color[:, :, 1]
        v = hsv_color[:, :, 2]
        HSV_MIN = np.array([h - 30, s - 30, v - 30])
        HSV_MAX = np.array([h + 30, s + 30, v + 30])
        mask = cv2.inRange(image_hsv, HSV_MIN, HSV_MAX)
        output = cv2.bitwise_and(image_inverted, image_inverted, mask=mask)
        cv2.imshow('output', output)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = np.zeros(image_inverted.shape, dtype=np.uint8)
        cv2.drawContours(image=img_contours, contours=contours, contourIdx=-1, color=1, thickness=-1,
                         hierarchy=hierarchy, maxLevel=1)

        cv2.imshow("Contours", img_contours * 255)
        img_contour0 = np.zeros(image_inverted.shape, dtype=np.uint8)
        cv2.drawContours(image=img_contour0, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=-1)
        cv2.imshow("Contour 0", img_contour0)

        cnt = contours[0]
        M = cv2.moments(cnt)
        print(M)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print("cx:", cx, ";  cy:", cy)
        area = cv2.contourArea(cnt)
        print("area:", area)
        perimeter = cv2.arcLength(cnt, True)
        print("perimeter:", perimeter)

    cv2.waitKey(1)
    game.after(1, camera, game)
