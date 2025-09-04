import imageio
import PIL
from PIL import Image
import requests
from io import BytesIO
from PIL import ImageFilter
from PIL import ImageEnhance
from IPython.display import display
import numpy as np

for i in range (1, 7):

    image = Image.open('pigs/pig%i.png' %i)

    saturation = 0

    for i in range(150, 160):
        for j in range (150, 160):

            colors = image.getpixel((i, j))
            r = colors[0] / 255
            g = colors[1] /255
            b = colors[2] / 255
        
            lum = 0.5 * (max(r, g, b) + min(r, g, b))

            if(lum != 1): 
                saturation += max(r, g, b) - min(r, g, b)

    saturation /= 100

    print(saturation)
        


