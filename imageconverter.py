
import os
from PIL import Image,ImageEnhance

BASE_DIR = r'D:\atmsystem\p'
saving_dir = r'D:\atmsystem\pos'
photos = os.listdir(BASE_DIR)

print(photos)

for photo in photos:
    full_path = os.path.join(BASE_DIR, photo)
    saving_path=os.path.join(saving_dir,photo)
    photo_obj = Image.open(full_path)
    photo_obj = photo_obj.convert('L')
    my=photo_obj.resize((100,100),Image.ANTIALIAS)
    my.show()
    my.save(saving_path)