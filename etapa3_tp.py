import cv2

window_camera = "Camera"
window_faces = "Face detection"
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
image_faces = None


def start(game):
    cap = cv2.VideoCapture()
    cv2.namedWindow(window_camera)
    camera(cap, game)


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
    global image_faces
    wMax = 0
    hMax = 0
    faces = face_cascade.detectMultiScale(frame)
    image_faces = frame.copy()
    if len(faces) != 0: #se existir alguma cara detetada
        for (x, y, w, h) in faces:
            if w*h > wMax*hMax: #se a area do retangulo onde tem a cara for a maior
                wMax = w
                hMax = h
                xMax = x
                yMax = y
        #apenas desenha o retangulo na detecao com maior area
        cv2.rectangle(image_faces, (xMax, yMax), (xMax + wMax, yMax + hMax), (0, 255, 0), 2)
        movePaddle(game, xMax)


def movePaddle(game, faceX):
    coords = game.paddle.get_position() #obtem as cordenadas do paddle
    #x = valor que deve ser somado/subtraido à cordenada X do paddle para ficar na mesma posicao X que a cara
    x = faceX - coords[0]
    game.paddle.move(x)


def showImages(frame):
    global window_camera, window_faces, image_faces
    cv2.imshow(window_camera, frame)
    if image_faces is not None:
        cv2.imshow(window_faces, image_faces)
