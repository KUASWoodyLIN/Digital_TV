import numpy as np
import matplotlib.pyplot as plt
import cv2

img = plt.imread('./lenna.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
plt.figure('Original image')
plt.imshow(img, cmap='gray')

width = img.shape[0]
high = img.shape[1]
print "Image shape: " + str(img.shape)

# n level compress
n = 4

# Compress process
compress_w = np.zeros_like(img)
compress_h = np.zeros_like(img)
for i in range(n):
  compress_w[:high, :width/2] = (img[:high, :width:2] + img[:high, 1:width:2]) / 2
  compress_w[:high, width/2:width] = (img[:high, :width:2] - img[:high, 1:width:2]) / 2

  compress_h[:high/2, :width] = (compress_w[:high:2, :width] + compress_w[1:high:2, :width]) / 2
  compress_h[high/2:high, :width] = (compress_w[:high:2, :width] - compress_w[1:high:2, :width]) / 2
  img = compress_h
  high = high / 2
  width = width / 2

plt.figure('Compress image')
plt.imshow(compress_h, cmap='gray')
plt.imsave('Compress_image.png', compress_h, cmap='gray')


# Decompress process
decompress_h = np.copy(img)
for i in range(n):
  high = high * 2
  width = width * 2
  decompress_h[:high:2, :width] = img[:high/2, :width] + img[high/2:high, :width]
  decompress_h[1:high:2, :width] = img[:high/2, :width] - img[high/2:high, :width]

  decompress_w = np.copy(compress_h)
  decompress_w[:high, :width:2] = decompress_h[:high, :width/2] + decompress_h[:high, width/2:width]
  decompress_w[:high, 1:width:2] = decompress_h[:high, :width/2] - decompress_h[:high, width/2:width]
  img = decompress_w

plt.figure('Decompress image')
plt.imshow(img, cmap='gray')
plt.imsave('Decompress_image.png', img, cmap='gray')


plt.show()


