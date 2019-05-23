#!python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # To grab the images and extract useful information
# requires python -m pip install pillow on python 3.6.8

import matplotlib.pyplot as plt

print("start")

# sample open one image file
filename = "images/cat.jpg"
print(filename)
oneimagefile = Image.open(filename)
print("type of " + filename + str(type(oneimagefile)))
print("Image format: {}\nImage mode: {}\nImage size: {}"
      .format(
          oneimagefile.format,
          oneimagefile.mode,
          oneimagefile.size))
# create an array object from the image
imgarr = np.array(oneimagefile)
print("Image array shape: {}"
      .format(
          imgarr.shape
      ))

print("print image")
for h in range(oneimagefile.size[1]):
    line_s = ""
    for w in range(oneimagefile.size[0]):
        rgb_s = ""
        pixel = imgarr[h,w]
        rgb_s = "{}{}{} ".format(
            hex(pixel[0]).lstrip("0x"),
            hex(pixel[1]).lstrip("0x"),
            hex(pixel[2]).lstrip("0x")
        )
        line_s = line_s + rgb_s
    # print(line_s)

print("finished printing image")

plt.imshow(oneimagefile)
plt.show() 
