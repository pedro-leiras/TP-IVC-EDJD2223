import cv2
import numpy as np

cap = cv2.VideoCapture()
image_inverted = np.array([])
window_camera_name = "Camera"
hsv_color = np.array([])

def start(game):
    cv2.namedWindow(window_camera_name)
    camera(game)

def getHSVColorFromMouseClick(event, x, y,flags,param):
    global hsv_color, image_inverted

    if event == cv2.EVENT_LBUTTONDOWN:
        B = image_inverted[y, x, 0]
        G = image_inverted[y, x, 1]
        R = image_inverted[y, x, 2]
        bgr = image_inverted[y, x]
        bgr_array = np.uint8([[[B, G, R]]])
        hsv_color = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
        print("HSV : ", hsv_color)
        print("BGR: ", bgr)


def camera(game):
    global image_inverted, hsv_color

    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    image_inverted = image[:, ::-1, :]
    cv2.imshow(window_camera_name, image_inverted)

    if hsv_color.size == 0:
        cv2.setMouseCallback(window_camera_name, getHSVColorFromMouseClick)
    else:
        cv2.setMouseCallback(window_camera_name, lambda *args : None)
        image_hsv = cv2.cvtColor(image_inverted, cv2.COLOR_BGR2HSV)
        hsv_color_copy = hsv_color.copy()
        h = hsv_color_copy[:, :, 0].copy()
        s = hsv_color_copy[:, :, 1].copy()
        v = hsv_color_copy[:, :, 2].copy()

        if h <= 30:
            h[:, 0] = 0
        else:
            h -= 30

        if s <= 40:
            s[:, 0] = 0
        else:
            s -= 40

        if v <= 40:
            v[:, 0] = 0
        else:
            v -= 40

        HSV_MIN = np.array([h, s, v])
        h = hsv_color_copy[:, :, 0].copy()
        s = hsv_color_copy[:, :, 1].copy()
        v = hsv_color_copy[:, :, 2].copy()

        if h >= 150:
            h[:, 0] = 180
        else:
            h += 30

        if s >= 215:
            s[:, 0] = 255
        else:
            s += 40

        if v >= 215:
            v[:, 0] = 255
        else:
            v += 40

        HSV_MAX = np.array([h, s, v])

        mask = cv2.inRange(image_hsv, HSV_MIN, HSV_MAX)
        output = cv2.bitwise_and(image_inverted, image_inverted, mask=mask)
        cv2.imshow('output', output)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = np.zeros(image_inverted.shape, dtype=np.uint8)
        cv2.drawContours(image=img_contours, contours=contours, contourIdx=-1, color=1, thickness=-1,
                         hierarchy=hierarchy, maxLevel=1)

        if contours:
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img_contours, (cx, cy), 7, (0, 0, 255), -1)
                print(f"x: {cx} y: {cy}")
                x = image_inverted.shape
                if(cx < x[0]/2):
                    game.paddle.move(-10)
                else:
                    game.paddle.move(10)
        cv2.imshow("Contours", img_contours * 255)

    cv2.waitKey(1)
    game.after(1, camera, game)
