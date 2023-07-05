import cv2
import numpy as np
import math

film = cv2.VideoCapture('celny.mp4')
#tablica punktów
X = []
Y = []
#lista wszystkich pixeli na osi x 
listX = [i for i in range(0, 1420)]
#funkcja znajdująca piłkę 
def BallContours(frame, mask, treshhold):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        field = cv2.contourArea(cnt)
        if field >= treshhold:
            cv2.drawContours(frame, cnt, -1, (0, 0, 255), 5)
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            cx = x + w // 2
            cy = y + h // 2
            X.append(cx)
            Y.append(cy)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            #print(cnt[index][0][0])
    
    return frame
#barwa w hsv piłki 
hsvBall = {'hmin': 36, 'smin': 61, 'vmin': 22, 'hmax': 92, 'smax': 190, 'vmax': 255}
#{'hmin': 67, 'smin': 63, 'vmin': 11, 'hmax': 92, 'smax': 255, 'vmax': 139}
#normalna_wiczor{'hmin': 0, 'smin': 34, 'vmin': 0, 'hmax': 15, 'smax': 100, 'vmax': 108}
#CELNY{'hmin': 59, 'smin': 54, 'vmin': 0, 'hmax': 90, 'smax': 255, 'vmax': 255} # znaleziona barwa piłki

while True:
    ret, frame = film.read()
    if not ret and not X:
        print("Nie można otworzyć filmu")
        break
    elif not ret:
        break
    
    #przycięcię klatki tak aby był tylko kosz 
    height, width, channels = frame.shape
    #print(height, width, channels)
    frame = frame[0:700, 500:width]
    cv2.imwrite('klatka.png', frame)
    #utworzenie maski
    mask = []
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerBoundry = np.array([hsvBall['hmin'], hsvBall['smin'], hsvBall['vmin']])
    upperBoundry = np.array([hsvBall['hmax'], hsvBall['smax'], hsvBall['vmax']])
    mask = cv2.inRange(frameHSV, lowerBoundry, upperBoundry)

    #narysowanie toru lotu pilki
    contours = BallContours(frame, mask, 700)
    for index, point in enumerate(zip(X, Y)):
        cv2.circle(contours, (point[0], point[1]), 10, (0, 0, 0), cv2.FILLED)
        if index > 0:
            cv2.line(contours, (X[index], Y[index]), (X[index - 1], Y[index - 1]), (255, 0, 0))
        else:
            cv2.line(contours, (X[index], Y[index]), (X[index], Y[index]), (255, 0, 0))
    
    # Y = Ax^2 + Bx + C # PREDYKCJA
    if X:
        a, b, c = np.polyfit(X, Y, 2)
        
        for x in listX:
            y = int(a * x * x + b * x + c)
            cv2.circle(contours, (x,y), 2, (0, 0, 125), cv2.FILLED)

        # koordynaty kosza x: 315 - 440 y: 615
        #obliczamy x 
            if c >= 615:
                x = int((-b - math.sqrt(b ** 2 - (4 * a * (c - 615)))) / (2 * a))

                if 315 <= x <= 440:
                    cv2.putText(contours, 'BASKET', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                else:
                    cv2.putText(contours, 'MISS', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            else:
                cv2.putText(contours, 'MISS', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

# Końcowe okno
    imgColor = cv2.resize(mask, (0,0), None, 0.7, 0.7)
    
    
    cv2.imshow("Ball", contours)
    #aby przerwać pokazywanie klatek naciśnij "f"
    if cv2.waitKey(10) == ord('f'):
       break

film.release()
cv2.destroyAllWindows()