import cv2
import numpy as np
import os

def gaborFilters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F) #getting Gabor features
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

gabor_filter = gaborFilters()

filename = 'data/Fake'
for root, dirs, files in os.walk(filename):
    for fdata in files:
        image_file = root+"/"+fdata;
        img = cv2.imread(image_file)
        img = process(img,gabor_filter)
        cv2.imwrite('Gabor/train/Fake/'+fdata, img)


filename = 'data/Real'
for root, dirs, files in os.walk(filename):
    for fdata in files:
        image_file = root+"/"+fdata;
        img = cv2.imread(image_file)
        img = process(img,gabor_filter)
        cv2.imwrite('Gabor/train/Real/'+fdata, img)        
