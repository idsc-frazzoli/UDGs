from map import straightLineW2E
from PIL import Image
from numpy import asarray
import numpy as np
from matplotlib import pyplot as plt
import os

from map.utils import load_scenario

path = os.path.dirname(__file__)

def test_load_tracks():
    picpath = os.path.join(path, "road06.png")
    img1 = Image.open(picpath)
    data1 = asarray(img1)
    img2 = Image.fromarray(data1)
    # image2.show()
    # img1.show()
    print(img2.format, img2.size, img2.mode)
    lane = straightLineW2E
    data2 = asarray(lane.background*255)
    img = Image.fromarray(data2.astype(np.uint8))
    img.show()
    print(img.format, img.size, img.mode)
    print("so far so good")


def test_load_scenario():
    load_scenario()