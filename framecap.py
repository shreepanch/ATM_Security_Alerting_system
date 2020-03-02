import os
import numpy as np
import cv2

cap = cv2.VideoCapture('video.avi')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('error too create path')



currentframe = 0
while (True):
    ret, frame = cap.read()
    name = 'D:\atmsystem\data\frame' + str(currentframe) + '.jpg '
    print('creating......' + name)
    cv2.imwrite(name, frame)
    currentframe += 1
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
