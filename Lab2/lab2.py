import numpy as np
import matplotlib.pyplot as plt
import cv2

img = plt.imread('./lenna.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
fig = plt.figure()
plt.imshow(img, cmap='gray')

width = img.shape[0]
high = img.shape[1]
print(img.shape)

n = 3
level_w = np.zeros_like(img, np.float64)
level_h = np.zeros_like(img, np.float64)
for i in range(n):
    for i in range(high):
        for count, j in enumerate(range(0, width, 2)):
            level_w[i,count] = ((img[i,j] + img[i,j+1]) / 2.)
            level_w[i,width/2+count] = ((img[i,j] - img[i,j+1]) / 2.)

    for i in range(width):
        for count, j in enumerate(range(0, high, 2)):
            level_h[count, i] = ((level_w[j, i] + level_w[j+1, i]) / 2.)
            level_h[high/2+count, i] = ((level_w[j, i] - level_w[j+1, i]) / 2.)
    img = level_h
    high = high / 2
    width = width / 2
fig = plt.figure()
plt.imshow(level_h, cmap='gray')    

return_h = img
return_w = np.zeros_like(img, np.float64)
for i in range(n):
    width = width * 2
    high = high * 2

    for i in range(width):
        for count, j in enumerate(range(0, high, 2)):
            return_h[j, i] = (img[count, i] + img[count+high/2, i]) / 2.
            return_h[j+1, i] = (img[count, i] - img[count+high/2, i]) / 2.

    for i in range(high):
        for count, j in enumerate(range(0, width, 2)):
            return_w[i, j] = (return_h[i, count] + return_h[i, count+width/2]) / 2.
            return_w[i, j+1] = (return_h[i, count] - return_h[i, count+width/2]) / 2.
    img = return_w



fig = plt.figure()
plt.imshow(img, cmap='gray')
plt.show()


