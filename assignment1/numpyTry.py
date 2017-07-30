#import numpy as np

#a = np.array([0,1,2,3, 4 , 5 , 6, 7, 8, 9, 10])
#print (type(a), a.shape, a[0], a[1], a[2])
#a[0] = 5
#print (a[:11:2])

#a = np.array([[1,2,3, 4], [ 5 , 6, 7, 8], [9, 10,11,12]])
#b = a[:, 1:3]

#print(b[0,1])
#print(b)
#b[0,0] = 77
#print(b[1,1])
#print (a%2==0)

#from scipy.misc import imread, imsave, imresize


# Read an JPEG image into a numpy array
#img = imread('/home/daryl/Downloads/sunflower.jpg')
#print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
#img_tinted = img * [0.5, 0.9, 0.8]

# Resize the tinted image to be 300 by 300 pixels.
#img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
#imsave('/home/daryl/Downloads/sunflower tinted.jpg', img_tinted)
#!/usr/bin/env python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('/home/daryl/Downloads/sunflower.jpg')
img_tinted = img * [0.8, 0.9, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()

