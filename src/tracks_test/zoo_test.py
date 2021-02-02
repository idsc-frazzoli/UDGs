from tracks import straightLineR2L, winti_002
from PIL import Image
from numpy import asarray
import numpy as np
from matplotlib import pyplot as plt
import os

path = os.path.dirname(__file__)

def test_load_tracks():
    picpath = os.path.join(path, "road06.png")
    img1 = Image.open(picpath)
    data1 = asarray(img1)
    img2 = Image.fromarray(data1)
    # image2.show()
    # img1.show()
    print(img2.format, img2.size, img2.mode)
    track = straightLineR2L
    data2 = asarray(track.background*255)
    img = Image.fromarray(data2.astype(np.uint8))
    img.show()
    print(img.format, img.size, img.mode)
    print("so far so good")