import PIL
from PIL import Image
import numpy as np
import pandas as pd
import sys
import cv2
from matplotlib import pyplot as plt

image1 = sys.argv[1]
image2 = sys.argv[2]
image3 = sys.argv[3]

color = "F"
img1 = np.asarray(Image.open(image1).convert(color))
img2 = np.asarray(Image.open(image2).convert(color))
img3 = np.asarray(Image.open(image3).convert(color))
if img2.max()<img1.max():
	lim = img1.max()
else:
	lim = img2.max()
if lim < img3.max():
	lim = img3.max()
cmap = "gray"
inter="bilinear"
minval=0
lim = 10
print((abs(img3-img2)).max())
print((abs(img3-img2)).min())
plt.figure("astra")
plt.imshow(img1, cmap = cmap, interpolation=inter,vmin=minval,vmax=lim)
plt.figure("photoneo")
plt.imshow(img2, cmap = cmap, interpolation=inter,vmin=minval,vmax=lim)
plt.figure("trained")
plt.imshow(img3, cmap = cmap, interpolation=inter,vmin=minval,vmax=lim)
#plt.figure("loss")
#plt.imshow(abs(img3-img2), cmap = "jet", interpolation=inter,vmin=0,vmax=20)
#plt.colorbar()
#plt.figure("loss2")
#plt.imshow(abs(img3-img2), cmap = "jet", interpolation=inter,vmin=0,vmax=200)
#plt.colorbar()
#plt.figure("loss3")
#plt.imshow(abs(img3-img2), cmap = "jet", interpolation=inter,vmin=0,vmax=900)
plt.colorbar()
plt.show()
cv2.waitKey(0)
plt.close('all')
