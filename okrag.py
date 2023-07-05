import cv2
import numpy as np

path = 'celny.mp4'
film = cv2.VideoCapture(path)

while True:
    ret, frame = film.read()
    if not ret:
        print('wtf')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('gray',gray)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp = 0.1, minDist=10000, param1 = 10, param2 = 30, minRadius = 100, maxRadius = 150)

    if circles is not None:
        circles = np.round(circles[0, :].astype(int))

    for (x, y, r) in circles:
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
    cv2.imshow('Ball', frame)
    if cv2.waitKey(0) == ord('f'):
       break
    cv2.destroyAllWindows()