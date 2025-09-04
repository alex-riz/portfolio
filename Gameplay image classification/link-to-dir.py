import requests
import os
from PIL import Image
#from resize import resizeImg

def resizeImg(path):
    dirs = os.listdir( path )
    count  = 0
    for item in dirs:
        count = count + 1
        if os.path.isfile(path+item):
            im = Image.open(path+item)
           # f, e = os.path.splitext(path+item)
            imResize = im.resize((800, 800), Image.ANTIALIAS)
            imResize.save( './input_models/pic%i.png' %count, 'PNG', optimize = True, quality=90)
            os.remove(path + item)
       # if 'png' in item:
        #    os.remove(path+item)       #sterg png

#file1 = open('urls.txt', 'r') #fisier -> aici bagam url-uri date de ei
#Lines = file1.readlines()

##count  = 0
#dirs = os.listdir( path )
#for line in dirs:
    #count  = count + 1

    #with open('/mnt/c/Users/40727/OneDrive/Documents/bestem22/input_models/imagine%i.png' %count, 'wb') as handle: #salvez imagine ca png -> grija la path
    #    response = requests.get(line.strip(), stream=True)
    #    if not response.ok:
    #        print(response)
    #    for block in response.iter_content(1024):
     #       if not block:
      #          break
       #     handle.write(block)

resizeImg("//mnt/c/Users/40727/OneDrive/Documents/bestem22/input_models/")     #le fac resize + salvez ca jpg -> schimba path eventual