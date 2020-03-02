
from tkinter import *
from tkinter import filedialog
import imutils,pyttsx3
import numpy as np
import os,cv2,time
from imutils.video import VideoStream
import argparse

from speek import speak


class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("ATM SECURITY SYSTEM")

        menu = Menu(self.master)
        self.master.config(menu=menu)

        # menu
        file = Menu(menu)
        file.add_command(label="Save")
        file.add_command(label="Exit", command=self.close)
        menu.add_cascade(label="File", menu=file)

        edit = Menu(menu)
        edit.add_command(label="Redo")
        edit.add_command(label="Undo")
        menu.add_cascade(label="Edit", menu=edit)

        # Graphics window
        #imageFrame = Frame(root, width=400, height=400)
        #imageFrame.grid(row=4, column=0, padx=10, pady=2)

        #live feed label
        livefeed = Label(root, text="Turn On the Live Feed~")
        livefeed.grid(row=1, column=0)

        #Live Feed
        var = IntVar()
        live_check_button = Checkbutton(root, text = "Live Feed", variable= var, command = self.detect)
        live_check_button.grid(row=2, column=0)
        print(var)

        #Select the video feed from your Device
        pic_button = Button(root, text="Get the Video File path", fg="blue", width=20, command=self.add_vid)
        pic_button.grid(row=1, column=2)

        #Add the selected Video to the Screen
        add = Button(root, text="Add", fg="green", width=12, command=self.load_vid)
        add.grid(row=2, column=2)

        # for the puting text on the frame
        livefeed = Label(root, text="helmet detection~")
        livefeed.grid(row=1, column=4)

        #add the button for the helmet detection
        add =Button(root, text="detect",fg='green',command=self.detecthelmet)
        add.grid(row=2,column=4)

        # About the Live Feed
        canvas = Canvas(root, width=400, height=400, bg='#ffffff')
        canvas.grid(row=4, column=0, rowspan=8, columnspan= 3)

        # showing the Picture that is taken
        canvas = Canvas(root, width=300, height=300, bg='#ffffff')
        canvas.grid(row=4, column=4, rowspan=2)



        withhelmet = Label(root, text="Waring a Helmet or")
        withhelmet.grid(row=6, column=4)

        withouthelmet = Label(root, text="Not Wearing a Helmet")
        withouthelmet.grid(row=7, column=4)

        wt = Label(root, text="SECURITY ON ATM SYSTEM")
        wt.grid(row=6, column=0)

        wot = Label(root, text="USING HAAR CASCADE ALGORITHM")
        wot.grid(row=7, column=0)

    global myframe
    global facecover

    def myframe(h):
        tx= Label(root,text="OBJECT ARE MOVING::"+str(h),fg='blue')
        tx.grid(row=4, column=0)
        return 0
    def facecover(h):
        tx = Label(root, text="OBJECT HAS ::" + str(h),fg='blue')
        tx.grid(row=5, column=0)


        # close = Button(root, text="Close", fg="green", width=12, command=self.close)
        # close.grid(row=1, column=8)

    def add_vid(self):
        global vid_path
        vid_path = filedialog.askopenfilename(initialdir="/", title="Select an image file",
                                              filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))

    def load_vid(self):
        cap = cv2.VideoCapture(0)

        while (cap.isOpened()):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cap.release()
        cv2.destroyAllWindows()

    def close(self):
        exit()

    def detecthelmet(self):
        import numpy as np
        import cv2

        face_cascade = cv2.CascadeClassifier('cc/cascade.xml')

        img = cv2.imread('44.jpg')
        # cv2.imshow('img', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = face_cascade.detectMultiScale(gray, 12, 50)
        flag=0
        for (x, y, w, h) in mask:
            im = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow('im', im)

            print(x, y, w, h)

            if 1>x:
                wot = Label(root, text="HELMET IS NOT DETECTED",fg='blue')
                wot.grid(row=5, column=4)
            else:
                wot = Label(root, text="HELMET IS  DETECTED",fg='blue')
                wot.grid(row=4, column=4)
                flag=1
        if flag==1:
            for i in range(3):
                speak("sir please remove your helmets please")
                time.sleep(3)
            wot = Label(root, text="WARNING::object is not removing the helmet",bg="#FFFFAA",fg='red')
            wot.grid(row=5, column=4)

            #canvas = Canvas(root, width=300, height=300, bg='#ffffff')
            #canvas.grid(row=4, column=4, rowspan=2)
            #canvas.pack()
            #ig = PhotoImage(file='D:\\atmsystem\\testimage\\44.png')
            #canvas.create_image(300,300,anchor=NW,image=ig)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect(self):

        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--delay", type=float,
                        help="Amount of time in seconds to dzzelay the program before starting.")
        ap.add_argument("-m", "--min-area", type=int, default=500,
                        help="Minimum area in pixels difference to be considered actual motion.")
        ap.add_argument("-t", "--thresh", default=25, type=int,
                        help="Level of threshold intensity.")
        ap.add_argument("-v", "--video-path",
                        help="Path to video file. If not provided the default video recording device on your system will be used.")
        args = vars(ap.parse_args())

        print("Program starting.\n")

        if args.get("video_path", None) is not None:
            try:
                print("Attempting to access the video at path: {}".format(args["video_path"]))
                video_stream = cv2.VideoCapture(args["video_path"])
                print("Successfully accessed video.")
            except:
                print("Could not access the specified video. Please make sure you are "
                      + "providing an absolute path to file.")
        else:
            try:
                print("Attempting to access the default video recording device.")
                video_stream = VideoStream(src=0).start()
                time.sleep(2.0)
                print("Successfully connected to the default recording device.")
            except:
                print("Could not access the default recording device. Please make sure "
                      + "you have a device capable of recording video configured on your system.")
        print()

        if args.get("delay", None) is not None:
            print("Starting delay of: {} seconds".format(args["delay"]))
            time.sleep(args["delay"])
            print("Delay complete.")

        # Init variable to hold first frame of video. This will be used as a reference.
        # The motion detection algorithm utilizes the background of the initial frame
        # to compare all consecutive frames to in order to detect motion
        initial_frame = None

        print("Starting motion detection")
        print("Enter 'q' at any time to terminate")

        first_frame = 0

        # gethering the resource for the voice massege

        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty("voices")
        print(voices)
        engine.setProperty('voice', voices[1].id)

        # define the voices to be speak
        def speak(audio):
            engine.say(audio)
            engine.runAndWait()
            time.sleep(3)

        # define the video formate which are to be save
        filename = 'video.avi'
        frames_per_seconds = 20.0
        my_res = '740'

        # formating the frame to be save
        def change_res(cap, width, height):
            cap.set(5, width)
            cap.set(6, height)

        # saving video quality or formate
        STD_DIMENSIONS = {
            "480p": (640, 480),
            "720p": (1280, 720),
            "1080": (1920, 1080),
            "4k": (3840, 2160)
        }

        # function for the selection the formate to be define
        def get_dim(cap, res=('1080p')):
            width, height = STD_DIMENSIONS['480p']
            if res in STD_DIMENSIONS:
                width, height = STD_DIMENSIONS[res]
            change_res(cap, width, height)
            return width, height

        # define the video type on the storage or saving formate
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        # seving file and formate selection
        def get_video_type(filename):
            filename, ext = os.path.splitext(filename)
            if ext in VIDEO_TYPE:
                return VIDEO_TYPE[ext]
            return VIDEO_TYPE['avi']

        # acces the camera for the inpute
        cap = cv2.VideoCapture(0)
        dims = get_dim(cap, res=my_res)
        out = cv2.VideoWriter(filename, get_video_type(filename), 20, get_dim(cap, my_res))
        dims = get_dim(cap, res=my_res)

        # baground subltraction for the motion detection on the frame
        fgbg = cv2.createBackgroundSubtractorMOG2(300, 400, True)

        # initilazing the parameter
        frameCount = 0
        currentframe = 0
        count = 0

        while True:

            # Set initial status to vacant.
            status = 'Area vacant.'

            # Grab current frame
            frame = video_stream.read()
            frame = frame if args.get("video_path", None) is None else frame[1]

            # If frame is none we have reached the end of the video
            if frame is None:
                break

                # Preprocess frame:
                # Resize to have a width of 500px. Improves speed without sacrificing accuracy
            frame = imutils.resize(frame, width=600)
            # Convert to grayscale as the background subtraction algorithm utilizes
            # black & white pixel data
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply guassian blur to smooth out image data and reduce irrelevant misleading
            # data from noise, scratches, etc.
            blurred_frame = cv2.GaussianBlur(grayscale_frame, (21, 21), 0)

            if initial_frame is None:
                initial_frame = grayscale_frame
                continue

            # Calculate the absolute difference between the current frame and the comparison

            # print(frame)

            def motion():
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # cv2.imshow('Original', frame)
                # cv2.imshow('frame', frame)
                count = 0

                # capture 6 photos if moving object are arais in the servilent

                while (count < 6):
                    cv2.imwrite("frame" + str(count) + ".jpg", grey)
                    count = count + 1
                    time.sleep(1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # check the initial frame or first frame has face or not
            face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=2.0, minNeighbors=5)
            for (x, y, w, h) in faces:
                # print(x, y, w, h)
                rol_gray = gray[y:y + h, x:x + w]
                rol_color = frame[y:y + h, x:x + h]
                img_item = "my-image.png"
                cv2.imwrite(img_item, rol_gray)
                cv2.imwrite(img_item, rol_color)

            #  Check if a current frame actually exist

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

            # text in the frame for the detecting the object is moving or not

            if (frameCount > 1 and count > 1000):

                myframe("YES")



                # text in the frame for the detecting the object is moving or note

                # speak('object is moving')

                # motion()

                print('object are moving')

                # if object has largaer then the threshold value then show object is moving in the grame
                cv2.putText(resizedFrame, 'object is moving', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

                # load the datebet for the object face detection on the servelent camera

                # faces = face_cascade.detectMultiScale(resizedFrame, scaleFactor=2.0, minNeighbors=5)
                face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
                # face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_upperbody.xml')
                eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye_tree_eyeglasses.xml')

                # change the frame into gray scale ande set the flage for the face is present or not

                gray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
                flage = 0

                # set the sceler factor  and no of neighbors of the face valu

                faces = face_cascade.detectMultiScale(
                    resizedFrame,
                    scaleFactor=1.3,
                    minNeighbors=7,
                    # Min size for valid detection, changes according to video size or body size in the video.
                    flags=cv2.CASCADE_SCALE_IMAGE

                )

                for (x, y, w, h) in faces:

                    img = cv2.rectangle(resizedFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    roi_gray = resizedFrame[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]

                    eyes = eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.3,
                        minNeighbors=7,
                        # Min size for valid detection, changes according to video size or body size in the video.
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                        eye_roi_gray = roi_color[ey:ey + eh, ex:ex + ew]
                        eye_roi_color = img[ey:ey + eh, ex:ex + ew]

                        # detection of noise present or not
                        nose_cascade = cv2.CascadeClassifier('cascades/data/nose.xml')
                        nose = nose_cascade.detectMultiScale(
                            eye_roi_color,
                            scaleFactor=1.3,
                            minNeighbors=7,
                            # Min size for valid detection, changes according to video size or body size in the video.
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )

                        for (nx, ny, nw, nh) in nose:
                            cv2.rectangle(eye_roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
                            nose_roi_gray = eye_roi_color[ey:ey + eh, ex:ex + ew]
                            nose_roi_color = img[ey:ey + eh, ex:ex + ew]

                    if 1 > x:
                        print('no face')
                        facecover("HAS NO FACE")

                    else:
                        flage = 1
                        print('has face')
                        facecover("HAS FACE")

                if flage == 0:

                    print('hass no face')




                    # if there is no face then load helmet detaset

                    # for the detection of upper body of humman which are in the servilent camera
                    # and that is the input for the ferther process
                    haar_upper_body_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_upperbody.xml")

                    # Uncomment this for real-time webcam detection
                    # If you have more than one webcam & your 1st/original webcam is occupied,
                    # you may increase the parameter to 1 or respectively to detect with other webcams, depending on which one you wanna use.

                    upper_body = haar_upper_body_cascade.detectMultiScale(
                        resizedFrame,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50),
                        # Min size for valid detection, changes according to video size or body size in the video.
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    # Draw a rectangle around the upper bodies
                    for (x, y, w, h) in upper_body:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        # creates green color rectangle with a thickness size of 1
                        cv2.putText(frame, "Upper Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0),
                                    2)
                        # creates green color text with text size of 0.5 & thickness size of 2

                    # if there is no face then load helmet detaset

                    helmet_cascede = cv2.CascadeClassifier('cc/cascade.xml')
                    helmet = helmet_cascede.detectMultiScale(
                        resizedFrame,
                        scaleFactor=12,
                        minNeighbors=50,
                        # Min size for valid detection, changes according to video size or body size in the video.
                        flags=cv2.CASCADE_SCALE_IMAGE

                    )

                    # detection helmet weaaring or not

                    for (xh, yh, wh, hh) in helmet:
                        img = cv2.rectangle(resizedFrame, (xh, yh), (xh + wh, yh + hh), (255, 0, 0), 2)

                        roi_gray = resizedFrame[yh:yh + hh, xh:xh + wh]
                        roi_color = img[yh:yh + hh, xh:xh + wh]
                        #speak('sir  please remove your helmet')

                    # detect mask wearing or not
                    # load data for the mask wearing or not

                    mask_cascade = cv2.CascadeClassifier('cascades/data/mask.xml')
                    # face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_upperbody.xml')

                    # data for the eye of the user
                    eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye_tree_eyeglasses.xml')

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    mask = mask_cascade.detectMultiScale(
                        resizedFrame,
                        scaleFactor=2.0,
                        minNeighbors=7,
                        # Min size for valid detection, changes according to video size or body size in the video.
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    for (x, y, w, h) in mask:

                        img = cv2.rectangle(resizedFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        roi_gray = resizedFrame[y:y + h, x:x + w]
                        roi_color = img[y:y + h, x:x + w]

                        # detect the eye on the face if mask was wearing the person

                        eyes = eye_cascade.detectMultiScale(
                            roi_gray,
                            scaleFactor=7,
                            minNeighbors=20,
                            # Min size for valid detection, changes according to video size or body size in the video.
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )

                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                            eye_roi_gray = roi_color[ey:ey + eh, ex:ex + ew]
                            eye_roi_color = img[ey:ey + eh, ex:ex + ew]

                # for the checking the faces is empty or not
                # if face is present then the capture the face

                for (x, y, w, h) in faces:
                    print(x, y, w, h)
                    rol_gray = gray[y:y + h, x:x + w]
                    rol_color = frame[y:y + h, x:x + h]
                    img_item = "image1.png"

                    # save the face on the physical storege device

                    cv2.imwrite(img_item, rol_gray)

                    cv2.imwrite(img_item, rol_color)




            # show the frame or videos in the system display

            cv2.imshow('Frame', resizedFrame)
            cv2.imshow('gray', fgmask)

            # save the video capture by camera

            out.write(frame)

            # terminet the frame if the key is q given thrrogh user or admine

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        if args.get("video_path", None) is not None:
            video_stream.release()
        else:
            video_stream.stop()

        # terminet the other packege
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Program terminating")


root = Tk()

if __name__ == "__main__":
    app = Window(root)

root.mainloop()


