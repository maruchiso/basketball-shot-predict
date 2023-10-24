import cv2


video = cv2.VideoCapture('finale/end.mp4')

start = 75.0
end = 78.0

video.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
output_video = cv2.VideoWriter('finale/end_celny.mp4', fourcc, 30.0, (width, height))

while True:

    ret, frame = video.read()

    current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if ret and current_time <= end:
        output_video.write(frame)
    
        if cv2.waitKey(1) & 0xFF == ord('f'):
            break
    else:
        break

video.release()
output_video.release()
cv2.destroyAllWindows()
