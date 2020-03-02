import numpy as np
import pyttsx3
import os
import cv2
import time

first_frame = 0
#gethering the resource for the voice massege
engine = pyttsx3.init('sapi5')
voices = engine.getProperty("voices")
print(voices)
engine.setProperty('voice', voices[1].id)

#define the voices to be speak
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

#gethering the face area from the cascadeclassification
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

#define the video formate which are to be save
filename = 'video.avi'
frames_per_seconds = 24.0
my_res = '740'


def change_res(cap, width, height):
    cap.set(5, width)
    cap.set(6, height)


STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080": (1920, 1080),
    "4k": (3840, 2160)
}


def get_dim(cap, res=('1080p')):
    width, height = STD_DIMENSIONS['480p']
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

#acces the camera for the inpute
cap = cv2.VideoCapture(0)
dims = get_dim(cap, res=my_res)
out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dim(cap, my_res))
dims = get_dim(cap, res=my_res)


fgbg = cv2.createBackgroundSubtractorMOG2(300, 400, True)

frameCount = 0
currentframe = 0
count=0

while True:
    ret, frame = cap.read()
    # print(frame)

    def motion():
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #cv2.imshow('Original', frame)
        #cv2.imshow('frame', frame)
        count=0

        #capture 6 photos if moving object are arais in the servilent

        while (count <6 ):
            cv2.imwrite("frame" + str(count) + ".jpg", grey)
            count = count + 1
            time.sleep(1)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#check the initial frame or first frame has face or not
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2.0, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        rol_gray = gray[y:y + h, x:x + w]
        rol_color = frame[y:y + h, x:x + h]
        img_item = "my-image.png"
        cv2.imwrite(img_item, rol_gray)
        cv2.imwrite(img_item, rol_color)



    #  Check if a current frame actually exist
    if not ret:
        break

    frameCount += 1
    # Resize the frame
    resizedFrame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    # Get the foreground mask
    fgmask = fgbg.apply(resizedFrame)

    # Count all the non zero pixels within the mask
    count = np.count_nonzero(fgmask)

    # print('Frame: %d, Pixel Count: %d' % (frameCount, count))

    # Determine how many pixels do you want to detect to be considered "movement"
    # if (frameCount > 1 and cou`nt > 5000):
    if (frameCount > 1 and count > 1000):

        #speak('object is moving')

        #motion()

        # print('object are moving')
        cv2.putText(resizedFrame, 'object is moving', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        faces = face_cascade.detectMultiScale(resizedFrame, scaleFactor=2.0, minNeighbors=5)

        #for the checking the faces is empty or not


        if faces is None:
            print("there is no face")
        else:
            print("there is face present")

        for (x, y, w, h) in faces:
            print(x, y, w, h)
            rol_gray = gray[y:y + h, x:x + w]
            rol_color = frame[y:y + h, x:x + h]
            img_item = "image1.png"

            cv2.imwrite(img_item, rol_gray)
            cv2.imwrite(img_item, rol_color)
#show the frame or videos in the system display
    cv2.imshow('Frame', resizedFrame)
    cv2.imshow('gray', fgmask)
#save the video capture by camera
    out.write(frame)

#exit the frame when press the exit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
