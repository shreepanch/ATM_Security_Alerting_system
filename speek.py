import pyttsx3
import os
import time

engine = pyttsx3.init('sapi5')
voices = engine.getProperty("voices")
print(voices)
engine.setProperty('voice', voices[1].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

#for i in range(3):
#    speak("please remove your helmets or mask")
 #   time.sleep(5)


#music_dr = 'D:\\Music\\New folder 1'
#song=os.listdir(music_dr)
#print(song)
#for i in range (1):
  # os.startfile(os.path.join(music_dr,song[i]))
   #time.sleep(11)


if __name__ == '__main__':
    pass

