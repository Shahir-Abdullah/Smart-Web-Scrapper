import cv2
import numpy as np
from selenium import webdriver
from time import sleep
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import re
import string 




#screenshot by selenium
filename = "images/input_image.png"

driver = webdriver.Firefox()
driver.get('https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/')
sleep(1)

driver.get_screenshot_as_file(filename)
driver.quit()

'''
# Load image, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread(filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Create rectangular structuring element and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=4)

# Find contours and draw rectangle
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

cv2.imshow('thresh', thresh)
cv2.imshow('dilate', dilate)
cv2.imshow('image', image)


cv2.waitKey()
'''