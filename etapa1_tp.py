import cv2
import numpy as np

window_camera = "Camera"
window_contours = "Contours"
window_threshed = "Image Threshed"
hsv_color = None
h_margin = 30
s_margin = 40
v_margin = 40


def start(game):
    cap = cv2.VideoCapture()
    cv2.namedWindow(window_camera)
    camera(cap, game)


def camera(cap, game):
    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    image_inverted = image[:, ::-1, :]

    image_contours, image_threshed = processImage(image_inverted, game)
    showImages(image_inverted, image_contours, image_threshed)

    cv2.waitKey(1)
    game.after(1, camera, cap, game)


def getHSVColorFromMouseClick(event, x, y, flags, frame):
    global hsv_color
    if event == cv2.EVENT_LBUTTONDOWN:
        B = frame[y, x, 0]
        G = frame[y, x, 1]
        R = frame[y, x, 2]
        bgr = np.uint8([[[B, G, R]]])
        hsv_color = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        print("HSV: ", hsv_color)
        print("BGR: ", bgr)


def processImage(frame, game):
    global hsv_color, window_camera

    # se ainda nao houver uma cor escolhida pelo utilizador atraves do rato
    # o programa nao processa a imagem da camara
    if hsv_color is None:
        cv2.setMouseCallback(window_camera, getHSVColorFromMouseClick, frame)
    else:
        # desativa a funcao de recolher a cor com o rato
        cv2.setMouseCallback(window_camera, lambda *args : None)

        hsv_min = getHSVMin()
        hsv_max = getHSVMax()

        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # mascara de segmentacao com as margens de hsv
        mask = cv2.inRange(image_hsv, hsv_min, hsv_max)
        # aplica na imagem da camara a mascara
        image_threshed = cv2.bitwise_and(frame, frame, mask=mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_contours = np.zeros(frame.shape, dtype=np.uint8)
        cv2.drawContours(image_contours, contours, -1, (0,255,0), 3)

        if contours:
            # iteracao para determinar o maior contorno
            largerContour = contours[0]
            for contour in contours:
                if largerContour.size < contour.size:
                    largerContour = contour

            M = cv2.moments(largerContour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # desenhar um circulo no centro do maior contorno
                cv2.circle(image_contours, (cx, cy), 7, (0, 0, 255), -1)
                print(f"x: {cx} y: {cy}")
                movePaddle(cx, frame, image_contours, game)

        return image_contours, image_threshed

    return None, None


# retorna uma a cor hsv com valores inferiores ao escolhido pelo utilizador
def getHSVMin():
    global hsv_color, h_margin, s_margin, v_margin

    h_min = hsv_color[:, :, 0].copy()
    s_min = hsv_color[:, :, 1].copy()
    v_min = hsv_color[:, :, 2].copy()

    #evita que o h fique abaixo de 0
    if h_min <= h_margin:
        h_min[:, 0] = 0
    else:
        h_min -= h_margin

    #evita que o s fique abaixo de 0
    if s_min <= s_margin:
        s_min[:, 0] = 0
    else:
        s_min -= s_margin

    #evita que o v fique abaixo de 0
    if v_min <= v_margin:
        v_min[:, 0] = 0
    else:
        v_min -= v_margin

    return np.array([h_min, s_min, v_min])


# retorna uma a cor hsv com valores superiores ao escolhido pelo utilizador
def getHSVMax():
    global hsv_color, h_margin, s_margin, v_margin

    h_max = hsv_color[:, :, 0].copy()
    s_max = hsv_color[:, :, 1].copy()
    v_max = hsv_color[:, :, 2].copy()

    #evita que o h fique acima de 180
    if h_max >= (180 - h_margin):
        h_max[:, 0] = 180
    else:
        h_max += h_margin

    #evita que o s fique acima de 255
    if s_max >= (255 - s_margin):
        s_max[:, 0] = 255
    else:
        s_max += s_margin

    #evita que o v fique acima de 255
    if v_max >= (255 - v_margin):
        v_max[:, 0] = 255
    else:
        v_max += v_margin

    return np.array([h_max, s_max, v_max])


def movePaddle(cx, frame, image_contours, game):
    image_width = frame.shape[1]
    image_height = frame.shape[0]
    # um quarto da largura da imagem da camara
    qhalf_width = int(image_width / 4)

    # 20% da largura da imagem da camara
    deathzone = int(qhalf_width * 0.2)

    #desenhar linhas no ecra para referencia
    # um quarto da largura
    cv2.line(image_contours, (qhalf_width, 0), (qhalf_width, image_height), (0, 0, 255), 5)
    # metade da largura menos metade da deathzone (inicio deathzone)
    cv2.line(image_contours, (int(qhalf_width * 2 - deathzone / 2), 0),
             (int(qhalf_width * 2 - deathzone / 2), image_height), (0, 0, 255), 5)
    # metade da largura mais metade da deathzone (fim deathzone)
    cv2.line(image_contours, (int(qhalf_width * 2 + deathzone / 2), 0),
             (int(qhalf_width * 2 + deathzone / 2), image_height), (0, 0, 255), 5)
    # tres quartos da largura
    cv2.line(image_contours, (int(qhalf_width * 3), 0), (int(qhalf_width * 3), image_height), (0, 0, 255), 5)

    #se o centro do contorno estiver no primeiro quarto de largura
    if cx < qhalf_width:
        game.paddle.move(-8)
    # se o centro do contorno estiver no segundo quarto de largura
    elif (cx >= qhalf_width) & (cx < int(qhalf_width*2 - deathzone/2)):
        game.paddle.move(-4)
    # se o centro do contorno estiver na deathzone
    elif (cx >= int(qhalf_width*2 - deathzone/2)) & (cx <= int(qhalf_width*2 + deathzone/2)):
        game.paddle.move(0)
    # se o centro do contorno estiver no terceiro quarto de largura
    elif (cx > int(qhalf_width*2 + deathzone/2)) & (cx <= qhalf_width*3):
        game.paddle.move(4)
    # se o centro do contorno estiver no ultimo quarto de largura
    else:
        game.paddle.move(8)


def showImages(frame, image_contours, image_threshed):
    global window_camera, window_contours, window_threshed

    cv2.imshow(window_camera, frame)
    if image_contours is not None:
        cv2.imshow(window_contours, image_contours)
    if image_threshed is not None:
        cv2.imshow(window_threshed, image_threshed)
