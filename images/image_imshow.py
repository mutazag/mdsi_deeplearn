
# http://cs231n.github.io/python-numpy-tutorial/ 

import numpy as np
import pandas as pd

from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

cat_image = imread('images/cat.jpg')
img_tinted = cat_image * [1, 0.5, 0.9]
# img_tinted = cat_image[:,:,2]

# Show the original image
plt.subplot(2, 3, 1)
plt.imshow(cat_image)

# Show the tinted image
plt.subplot(2, 3, 2)
# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))

plt.subplot(2, 3, 3)
plt.imshow(np.uint8(cat_image * [.5,.5,.5]))
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(np.uint8(cat_image[:,:,0]))
plt.subplot(2, 3, 5)
plt.imshow(np.uint8(cat_image[:,:,1]))
plt.subplot(2, 3, 6)
plt.imshow(np.uint8(cat_image[:,:,2]))


plt.gca().axes.get_xaxis().set_visible(False)
# plt.gca().axes.set_visible(False)
plt.show()

print('END')