import cv2

cap = cv2.VideoCapture()
image_inverted = 0
window_camera_name = "Camera"


def start(game):
    cv2.namedWindow(window_camera_name)
    camera(game)

def getRGB(event, x, y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rgb = image_inverted[y, x]
        print("Color: ", rgb)


def camera(game):
    global image_inverted

    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    image_inverted = image[:, ::-1, :]
    cv2.imshow(window_camera_name, image_inverted)
    cv2.setMouseCallback(window_camera_name,getRGB)
    cv2.waitKey(1)
    game.after(1, camera, game)


