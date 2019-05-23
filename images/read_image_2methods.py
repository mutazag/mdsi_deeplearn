#!python 


import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
import pandas as pd

filename = "images/cat.jpg"

#method 1
img1 = Image.open(filename)
#method 2 
img2 = plt.imread(filename)

# https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm
df = pd.DataFrame()

df = df.append(
    [
        ("method", "PIL.Image.open()", "plt.imread"), 
        ("type", type(img1), type(img2)), 
        ("size", img1.size, img2.shape)
    ]
)

print(df)
df.columns = ["property", "img1", "img2"]

print()
print(df)


plt.subplot(2,1,1)
plt.imshow(img1)
plt.title("cats")
plt.subplot(2,1,2)
plt.imshow(img2)
plt.title("cats2")
plt.show()

