import requests
import random
import os
import shutil
from pickletools import optimize
from PIL import Image
import os, sys


def shuffle(path, n, dst_path):
    dirs = os.listdir( path )
    i = 0
    while i < n:
        curent = random.choice(dirs)
        try:
            shutil.move(path + curent, dst_path + curent)
            i+=1
        except:
            pass


shuffle("/mnt/c/Users/40727/OneDrive/Documents/bestem22/GameImages/training/assassin/", 706, "/mnt/c/Users/40727/OneDrive/Documents/bestem22/GameImages/tests/assassin/")
shuffle("/mnt/c/Users/40727/OneDrive/Documents/bestem22/GameImages/training/peppa/", 356, "/mnt/c/Users/40727/OneDrive/Documents/bestem22/GameImages/tests/peppa/")