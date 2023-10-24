import cv2
import numpy as np
import math

while True:
    path = input('Podaj ścieźkę do filmiku\n')
    if isinstance(path, str):
        break

#odczyt filmu
film = cv2.VideoCapture(path)
#listy które zbierają położenie środka piłki X,Y do predykcji toru lotu 
# X_line, Y_line do zwykłego lotu piłki
X = []
Y = []
X_line = []
Y_line = []
#oś X
axisX = [i for i in range(0, 2000)]
#główna pętla do przetwarzenia filmu
while True:
    ret, frame = film.read()
    if not ret and not X:
        print('Nie można otworzyć filmu ze ścieżki: ', path)
        break
    if frame is None:
        print('Koniec filmu')
        break
    
    #zebranie wysokości i długości klatki
    height, width, _ = frame.shape
    #print(height, width) 
    #wyciecie kaltki w taki sposób aby nam nie przeszkadzały inne elementy w wykryciu piłki
    ball = frame[400:800, 700:width]
    gray = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=10, param2=25, minRadius=20, maxRadius=38)

    #narysowanie okręgu na klatce
    if circles is not None:
        circles = np.round(circles[0, :].astype(int))
        
        for (x, y, r) in circles:
            cv2.circle(ball, (x, y), r, (0, 255, 0), 2)
            #dodanie do listy koordynatów środka piłki i narysowanie go
            X_line.append(x)
            Y_line.append(y)
            cv2.circle(ball, (x, y), 10, (255, 255, 255), cv2.FILLED)
            #program tworzy predykcję tylko do czasu gdy piłka leci w stronę kosza, po odbiciu piłki
            # od konstrukcji kosza nie zbiera punktów 
            if X == []:
                X.append(x)
                Y.append(y)
            elif x <= X[len(X) - 1]:
                X.append(x)
                Y.append(y)

    #wykrycie tablicy 
    roi = ball[:, :100]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_low = np.array([0, 0, 50])
    white_high = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, white_low, white_high)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    backboard = max(contours, key = cv2.contourArea)
    x1, y1, w, h = cv2.boundingRect(backboard)
    cv2.rectangle(roi, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
    #print(x1, y1, w, h)
    ball[:, :100] = roi
      
        #narysowanie linii

    for index, point in enumerate(zip(X_line, Y_line)):
        if index > 0:
            cv2.line(ball, (X_line[index], Y_line[index]), (X_line[index - 1], Y_line[index - 1],), (255, 0, 0), 7)
        else:
            cv2.line(ball, (X_line[index], Y_line[index]), (X_line[index], Y_line[index],), (255, 0, 0), 7)
    #cv2.imwrite('finale_obrazek.png', ball)

#predykcja
#Tor lotu piłki jest parabolą a więc można go przedstawić jako:
# y = a * x ^ 2 + b * x + c
    if X:
        a, b, c = np.polyfit(X, Y, 2)
        for x in axisX:
            y = int(a * x * x + b * x + c)
            cv2.circle(ball, (x, y), 2, (0, 0, 125, cv2.FILLED))

        #wyliczenie czy piłka wpadnie do kosza na podstawie koordynatów tablicy tablicy
        if c >= y1 + h - 25 and b * b >= (4 * a * (c - y1 + h - 25)):
            x = int((-b - math.sqrt(b * b - (4 * a * (c - y1 + h - 25)))) / (2 * a))
            if x1 + w + 40 <= x <= x1 + w + 40 + 100:
                cv2.putText(frame, 'BASKET', (200, 400), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'MISS', (200, 400), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'MISS', (200, 400), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        frame[400:800, 700:width] = ball
        
    # Końcowe okno
    frame = cv2.resize(frame, None, fx = 0.7, fy = 0.7)
    cv2.imshow("Ball", frame)
    #przerwanie klatek naciśnij "f"
    if cv2.waitKey(100) == ord('f'):
       break

film.release()
cv2.destroyAllWindows()
