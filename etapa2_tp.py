import cv2
import numpy as np

window_camera = "Camera"

selection = None
drag_start = None
track_window = None
xPrevious = None
hist = None


def start(game):
    cap = cv2.VideoCapture()
    cv2.namedWindow(window_camera)
    cv2.setMouseCallback(window_camera, mouseSelectArea)
    camera(cap, game)


def mouseSelectArea(event, x, y, flags, param):
    global drag_start, track_window, selection

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        track_window = None
    if drag_start:
        xmin = min(x, drag_start[0])
        ymin = min(y, drag_start[1])
        xmax = max(x, drag_start[0])
        ymax = max(y, drag_start[1])
        selection = (xmin, ymin, xmax, ymax)
    if event == cv2.EVENT_LBUTTONUP:
        drag_start = None
        track_window = (xmin, ymin, xmax - xmin, ymax - ymin)


def camera(cap, game):
    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    image_inverted = image[:, ::-1, :]
    frame = image_inverted.copy()

    processImage(frame, game)
    showImages(frame)

    cv2.waitKey(1)
    game.after(1, camera, cap, game)


def processImage(frame, game):
    global selection, track_window, drag_start, hist

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    if selection:
        x0, y0, x1, y1 = selection
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist = hist.reshape(-1)

        frame_roi = frame[y0:y1, x0:x1]
        cv2.bitwise_not(frame_roi, frame_roi)
        frame[mask == 0] = 0

    if track_window and track_window[2] > 0 and track_window[3] > 0:
        selection = None
        prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        prob &= mask
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        track_box, track_window = cv2.CamShift(prob, track_window, term_crit)

        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        movePaddle(game, x)


def movePaddle(game, x):
    global xPrevious

    if xPrevious:
        if abs(xPrevious - x) > 2:
            if x < xPrevious:
                game.paddle.move(-10)
            elif x > xPrevious:
                game.paddle.move(10)
    xPrevious = x


def showImages(frame):
    global window_camera
    cv2.imshow(window_camera, frame)