# USAGE
# python opencv_text_detection_image.py --image images/bangla.png --east frozen_east_text_detection.pb
# or
# python opencv_text_detection_image.py --image images/Book.jpg --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import shutil
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-east", "--east", type=str,
                help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=1280,
                help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=1280,
                help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]
pH = H
pW = W
# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
    # extract the scores (probabilities), followed by the geometrical
    # data used to derive potential bounding box coordinates that
    # surround text
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    # loop over the number of columns
    for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        if scoresData[x] < args["min_confidence"]:
            continue

        # compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        # extract the rotation angle for the prediction and then
        # compute the sin and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # use the geometry volume to derive the width and height of
        # the bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # add the bounding box coordinates and probability score to
        # our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)
para_image = orig.copy()


count = 1
# change box width and height -> positive will add pixels and vice-versa
box_width_padding = 3
box_height_padding = 3

temp_image = orig.copy()

# delete output folder
try:
    shutil.rmtree('output')
except Exception as e:
    do = "nothing"

# create empty output folder
uncreated = 1
while (uncreated):
    try:
        os.mkdir('output')
        uncreated = 0
    except Exception as e:
        do = "nothing"

# define crop object


class Crop(object):
    def __init__(self, startX, startY, endX, endY):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY

    def __eq__(self, other):
        diff = abs(self.startY - other.startY)
        if (diff <= 10):
            return self.startX == other.startX
        else:
            False

    def __lt__(self, other):
        diff = abs(self.startY - other.startY)
        if (diff <= 10):
            return self.startX < other.startX
        else:
            return self.startY < other.startY


croppedList = []
para_regions = []
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW) - box_width_padding
    startY = int(startY * rH) - box_height_padding
    endX = int(endX * rW) + box_width_padding
    endY = int(endY * rH) + box_height_padding
    para_regions.append((startX, startY, endX, endY))
    # draw the bounding box on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # paragraph detector
    #cv2.rectangle(para_image, (startX, startY), (endX, endY), (255, 255, 255), -1)
    # append to croppedList to sort the images
    croppedList.append(Crop(startX, startY, endX, endY))

croppedList = sorted(croppedList)

# paragraph detection


def present(x, y):

    for i in para_regions:
        sx = i[0]
        sy = i[1]
        ex = i[2]
        ey = i[3]

        if x >= sx and x <= ex and y >= sy and y <= ey:
            return True
    return False


for i in range(0, pW):
    for j in range(0, pH):

        if present(i, j) == False:

            para_image[j][i] = [255, 255, 255]

'''
para_image = cv2.GaussianBlur(para_image, (7,7), 0)

sharpened_para_image = cv2.cvtColor(para_image, cv2.COLOR_BGR2GRAY)

# Create kernel

kernel = np.array([[0, -1, 0], 
                   [-1, 5,-1], 
                   [0, -1, 0]])
# Sharpen image
sharpened_para_image = cv2.filter2D(sharpened_para_image, -1, kernel)
'''


'''

for img in croppedList:
    roi = temp_image[img.startY:img.endY, img.startX:img.endX]
    cv2.imwrite("output/" + str(count) + ".jpg", roi)
    count = count + 1
'''


# show the output image
cv2.imwrite("output/Text Detection.jpg", orig)
cv2.imwrite("output/Paragraph_detection.jpg", para_image)

# Load source / input image as grayscale, also works on color images...
imgIn = cv2.imread("output/Paragraph_detection.jpg", cv2.IMREAD_GRAYSCALE)
#cv2.imshow("Original", imgIn)


# Create the identity filter, but with the 1 shifted to the right!
kernel = np.zeros((9, 9), np.float32)
kernel[4, 4] = 2.0  # Identity, times two!

# Create a box filter:
boxFilter = np.ones((9, 9), np.float32) / 81.0

# Subtract the two:
kernel = kernel - boxFilter


# Note that we are subject to overflow and underflow here...but I believe that
# filter2D clips top and bottom ranges on the output, plus you'd need a
# very bright or very dark pixel surrounded by the opposite type.

custom = cv2.filter2D(imgIn, -1, kernel)
cv2.imwrite("output/Paragraph_detection.jpg", custom)
#cv2.imshow("Sharpen", custom)
# cv2.waitKey(0)



############################## paragraph detection section ##################################

filename = "output/Paragraph_detection.jpg"
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
