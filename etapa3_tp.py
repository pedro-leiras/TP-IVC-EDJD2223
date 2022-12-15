import cv2

window_camera = "Camera"
window_faces = "Faces detection"
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
    faces = face_cascade.detectMultiScale(frame)
    image_faces = frame.copy()
    if len(faces) != 0:
        xMax = faces[0][0]
        yMax = faces[0][1]
        wMax = faces[0][2]
        hMax = faces[0][3]
        for (x, y, w, h) in faces:
            if w*h > wMax*hMax:
                wMax = w
                hMax = h
                xMax = x
                yMax = y
        cv2.rectangle(image_faces, (xMax, yMax), (xMax + wMax, yMax + hMax), (0, 255, 0), 2)
        movePaddle(game, xMax)


def movePaddle(game, objectX):
    coords = game.paddle.get_position()
    x = objectX - coords[0]
    game.paddle.move(x)


def showImages(frame):
    global window_camera, window_faces, image_faces
    cv2.imshow(window_camera, frame)
    if image_faces is not None:
        cv2.imshow(window_faces, image_faces)