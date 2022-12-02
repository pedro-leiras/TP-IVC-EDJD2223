import cv2
import numpy as np

window_camera = "Camera"
window_mask = "Mask"

selection = None
dragStart = None
trackWindow = None
xPrevious = None
hist = None
backproj = None


def start(game):
    cap = cv2.VideoCapture()
    cv2.namedWindow(window_camera)
    cv2.setMouseCallback(window_camera, mouseSelectArea)
    camera(cap, game)


def camera(cap, game):
    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    image_inverted = image[:, ::-1, :]
    frame = image_inverted.copy()

    processImage(frame, game)
    showImages(frame, backproj)

    cv2.waitKey(1)
    game.after(1, camera, cap, game)


def mouseSelectArea(event, x, y, flags, param):
    global dragStart, trackWindow, selection

    if event == cv2.EVENT_LBUTTONDOWN:
        dragStart = (x, y)
        trackWindow = None
    if dragStart:
        xmin = min(x, dragStart[0])
        ymin = min(y, dragStart[1])
        xmax = max(x, dragStart[0])
        ymax = max(y, dragStart[1])
        selection = (xmin, ymin, xmax, ymax)
    if event == cv2.EVENT_LBUTTONUP:
        dragStart = None
        trackWindow = (xmin, ymin, xmax - xmin, ymax - ymin)


def processImage(frame, game):
    global selection, trackWindow, dragStart, hist, backproj

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # mascara de segmentacao com as margens de hsv
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    #se existir uma area selecionada com o rato
    if selection:
        x0, y0, x1, y1 = selection
        hsv_roi = hsv[y0:y1, x0:x1] #o hsv dentro dos pontos da área selecionada
        mask_roi = mask[y0:y1, x0:x1] #a mascara dentro dos pontos da área selecionada
        hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180]) #obtem o histograma
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX) #transforma os valores em 0 ou 255
        hist = hist.reshape(-1) #altera a forma do array para 1 dimensao

        frame_roi = frame[y0:y1, x0:x1]
        cv2.bitwise_not(frame_roi, frame_roi) #permite ilustrar a area que esta a ser selecionada pelo rato

    if trackWindow and trackWindow[2] > 0 and trackWindow[3] > 0: #se existir área selecionada e a sua largura e altura for maior que 0
        selection = None #evita voltar a fazer o histograma feito acima
        backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1) #procura na imagem por features atraves do histograma feito acima
        backproj &= mask #aplica a mascara na imagem apenas com as features
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        track_box, trackWindow = cv2.CamShift(backproj, trackWindow, term_crit)  #aplica o camshift na imagem com as features

        #desenha e une com os pontos da feature obtidos no camshift
        pts = cv2.boxPoints(track_box)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, (0,255,0), 2)

        x, y, w, h = trackWindow
        movePaddle(game, x)


def movePaddle(game, x):
    global xPrevious

    if xPrevious:
        #margem de erro (a diferença positiva entre o x anterior e x atual)
        if abs(xPrevious - x) > 2:
            if x < xPrevious:
                game.paddle.move(-10)
            elif x > xPrevious:
                game.paddle.move(10)
    xPrevious = x


def showImages(frame, backgroundMask):
    global window_camera
    cv2.imshow(window_camera, frame)
    if backgroundMask is not None:
        cv2.imshow(window_mask, backgroundMask)