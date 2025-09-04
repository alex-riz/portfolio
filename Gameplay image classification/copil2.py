import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tensorflow as tf
from PIL import Image

model = load_model("my_model3.h5")

vs = open('/mnt/c/Users/40727/OneDrive/Documents/bestem22/pepppavsassassin.txt', 'r')
for line in vs:
    print(line)


line = "="
for i in range(30):
   line = line + "="
line = line + "TESTING IMAGES"   
for i in range(30):
   line = line + "="
print(line)   
line =""
for i in range(75):
   line = line + "-"
print(line)

for cnt in range (1,22): #cate teste avem
    image = load_img('input_models/pic%i.png' %cnt, target_size=(800,800))
    input = img_to_array(image)
    input = np.array([input])
    value = model.predict(input)

    image = Image.open('input_models/pic%i.png' %cnt)

    saturation = 0

    for i in range(1, 400):
        for j in range (1, 400):

            colors = image.getpixel((i, j))
            r = colors[0] / 255
            g = colors[1] /255
            b = colors[2] / 255
        
            lum = 0.5 * (max(r, g, b) + min(r, g, b))

            if(lum != 1): 
                saturation += max(r, g, b) - min(r, g, b)

    saturation /= (399 * 399)

    #print(saturation)

    # if(value >= 0.5 or saturation >= 0.27):
    #     print("Imaginea " + str(cnt) + " este din Peppa Pig")
    # else:
    #     print("Imaginea " + str(cnt) + " este din Assassin's Creed")
    
    if((value >= 0.5) ^ (saturation >= 0.27)):
        if(saturation >= 0.27):
            print("Imaginea " + str(cnt) + " este din Peppa Pig")
        else:
            print("Imaginea " + str(cnt) + " este din Assassin's Creed")
    else:
        if(value >= 0.5):
            print("Imaginea " + str(cnt) + " este din Peppa Pig")
        else:
            print("Imaginea " + str(cnt) + " este din Assassin's Creed")  
         


