import cv2
import pyttsx3
import time
import numpy as np

engine = pyttsx3.init('sapi5')
voices = engine.getProperty("voices")
print(voices)
engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Video Capture
capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture("demo.mov")

# History, Threshold, DetectShadows
# fgbg = cv2.createBackgroundSubtractorMOG2(50, 200, True)
fgbg = cv2.createBackgroundSubtractorMOG2(300, 400, True)

# Keeps track of what frame we're on
frameCount = 0
currentframe=0
while (1):
    # Return Value and the current frame
    ret, frame = capture.read()

    name = './date/frame' + str(currentframe) + '.jpg '
    print('creating......' + name)
    cv2.imwrite(name , frame)
    currentframe += 1
    #cv2.imshow('frame', frame)

    #  Check if a current frame actually exist
    if not ret:
        break

    frameCount += 1
    # Resize the frame
    resizedFrame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

    # Get the foreground mask
    fgmask = fgbg.apply(resizedFrame)

    # Count all the non zero pixels within the mask
    count = np.count_nonzero(fgmask)

    print('Frame: %d, Pixel Count: %d' % (frameCount, count))

    # Determine how many pixels do you want to detect to be considered "movement"
    # if (frameCount > 1 and cou`nt > 5000):
    if (frameCount > 1 and count > 5000):
        speak('object is moving')
        print('object are moving')
        cv2.putText(resizedFrame, 'boject is moving', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    cv2.imshow('Frame', resizedFrame)
    cv2.imshow('Mask', fgmask)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
